"""
PyTorch datasets for the ElephantVoices hackathon.

Two datasets live in this module:

* ElephantSpectrogramDataset
      Stage-1 baseline. One sample per annotated call; the chunk is positioned
      with random pre-roll jitter during training, and the target is the
      rectangular rumble mask. When a `NoiseBank` is supplied, each training
      sample is augmented with SNR-jitter: a pure mechanical-noise chunk
      sampled from the negative space of the dataset is mixed into the
      training window at a random SNR between -5 and +5 dB. This teaches the
      U-Net to stay robust across different background noise levels.

* InstanceMixDataset
      Stage-2 synthetic 2-caller mixtures. Each __getitem__ picks a random
      pair of annotations from DIFFERENT source files, loads both chunks,
      power-normalises them, mixes them with a random loudness offset, and
      emits three tensors:
          spec        [1, F, T]   log-magnitude of the mixture
          mask        [1, F, T]   union rectangular rumble mask
          instance_y  [K, F, T]   per-source Ideal Binary Mask labels
      The IBM labels give the Deep Clustering loss clean ground truth: bin i
      belongs to whichever source is loudest at that bin, restricted to the
      rumble mask (non-rumble bins get no label).

Both datasets share the full-pipeline STFT hyperparameters (TARGET_SR = 4000,
N_FFT = 8192, HOP_LENGTH = 1024, CHUNK_SECONDS = 8.0) so the U-Net sees the
same input shape (1, 2048, 32) in both stages.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from data_engine import (
    CallAnnotation,
    NoiseBank,
    load_raven_annotations,
    mix_at_snr,
)


# ---------------------------------------------------------------------------
# Hyperparameters (shared across dataset, training, and inference)
# ---------------------------------------------------------------------------

TARGET_SR: int = 4000
N_FFT: int = 8192
HOP_LENGTH: int = 1024
CHUNK_SECONDS: float = 8.0
FREQ_CROP_HZ: float = 1000.0
MASK_LOW_HZ: float = 10.0
MASK_HIGH_HZ: float = 1000.0
TOP_DB: float = 80.0

CHUNK_SAMPLES: int = int(round(CHUNK_SECONDS * TARGET_SR))
N_FRAMES: int = 1 + CHUNK_SAMPLES // HOP_LENGTH
FREQS: np.ndarray = librosa.fft_frequencies(sr=TARGET_SR, n_fft=N_FFT)
_FREQ_CROP_BINS_RAW: int = int(np.searchsorted(FREQS, FREQ_CROP_HZ))
FREQ_CROP_BINS: int = (_FREQ_CROP_BINS_RAW // 16) * 16

SNR_JITTER_RANGE: Tuple[float, float] = (-5.0, 5.0)
SNR_JITTER_PROB: float = 0.75

# Backward-compat re-exports: older code (`backend/inference.py`, `train.py`)
# imports `load_annotations` from this module. Keep the name alive as an
# alias for the Raven-aware parser in data_engine.
load_annotations = load_raven_annotations


# ---------------------------------------------------------------------------
# Spectrogram + mask helpers
# ---------------------------------------------------------------------------

def _normalise_db(mag: np.ndarray) -> np.ndarray:
    """Amplitude -> dB(ref=1) -> clipped to [-TOP_DB, 0] -> rescaled to [0, 1]."""
    mag_db = librosa.amplitude_to_db(mag, ref=1.0, top_db=None)
    mag_db = np.clip(mag_db, -TOP_DB, 0.0)
    return ((mag_db + TOP_DB) / TOP_DB).astype(np.float32)


def _magnitude_stft(y: np.ndarray) -> np.ndarray:
    """Full-resolution magnitude STFT of a waveform (no cropping)."""
    return np.abs(
        librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True)
    ).astype(np.float32)


def _compute_spec(y: np.ndarray) -> np.ndarray:
    """Waveform -> cropped, normalised log-magnitude spectrogram."""
    mag = _magnitude_stft(y)
    return _normalise_db(mag[:FREQ_CROP_BINS, :N_FRAMES])


def _time_to_frame(t_seconds: float) -> int:
    """Map a time in seconds (relative to chunk start) to an STFT frame index."""
    return int(round(t_seconds * TARGET_SR / HOP_LENGTH))


def _freq_to_bin(hz: float) -> int:
    """Map a frequency to a bin index in the cropped spectrogram."""
    return int(np.searchsorted(FREQS, hz))


def _rectangle_mask(
    windowed_calls: Sequence[Tuple[float, float, Optional[float], Optional[float]]],
) -> np.ndarray:
    """Build a binary rectangular rumble mask of shape (FREQ_CROP_BINS, N_FRAMES).

    Each entry in `windowed_calls` is (rel_start, rel_end, low_hz, high_hz)
    where the times are in seconds relative to the chunk start and both
    frequency bounds may be None (falling back to the global rumble band).
    Overlapping rectangles are OR'd together.
    """
    mask = np.zeros((FREQ_CROP_BINS, N_FRAMES), dtype=np.float32)
    for rel_start, rel_end, low_hz, high_hz in windowed_calls:
        f0 = max(0, min(_time_to_frame(rel_start), N_FRAMES))
        f1 = max(0, min(_time_to_frame(rel_end), N_FRAMES))
        if f1 <= f0:
            continue
        low = low_hz if low_hz is not None else MASK_LOW_HZ
        high = high_hz if high_hz is not None else MASK_HIGH_HZ
        b0 = max(0, min(_freq_to_bin(low), FREQ_CROP_BINS))
        b1 = max(0, min(_freq_to_bin(high), FREQ_CROP_BINS))
        if b1 <= b0:
            continue
        mask[b0:b1, f0:f1] = 1.0
    return mask


def _unit_power(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Scale a waveform so its RMS power equals 1.0."""
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2) + eps))
    return (x / rms).astype(np.float32)


# ---------------------------------------------------------------------------
# Baseline dataset: mask-only with SNR jitter
# ---------------------------------------------------------------------------

class ElephantSpectrogramDataset(Dataset):
    """One sample per annotated call.

    Output:
        spec [1, FREQ_CROP_BINS, N_FRAMES] log-magnitude spectrogram in [0, 1]
        mask [1, FREQ_CROP_BINS, N_FRAMES] binary rectangular rumble mask

    Training-mode augmentations:
        * random pre-roll jitter positions the call somewhere in the 8-second window
        * SNR jitter mixes a pure-noise chunk from the NoiseBank into the signal
          at a target SNR drawn uniformly from [-5, +5] dB (when noise_bank set)
    """

    def __init__(
        self,
        annotations: Union[str, os.PathLike, Sequence[CallAnnotation]],
        audio_dir: Union[str, os.PathLike],
        train: bool = True,
        cache_audio: bool = True,
        noise_bank: Optional[NoiseBank] = None,
        snr_jitter_prob: float = SNR_JITTER_PROB,
        snr_range: Tuple[float, float] = SNR_JITTER_RANGE,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.train = bool(train)
        self.cache_audio = bool(cache_audio)
        self.noise_bank = noise_bank
        self.snr_jitter_prob = float(snr_jitter_prob)
        self.snr_range = snr_range
        self._audio_cache: Dict[str, np.ndarray] = {}

        if isinstance(annotations, (str, os.PathLike)):
            self.annotations: List[CallAnnotation] = load_raven_annotations(annotations)
        else:
            self.annotations = list(annotations)

        self._by_file: Dict[str, List[CallAnnotation]] = {}
        for ann in self.annotations:
            self._by_file.setdefault(ann.file_name, []).append(ann)

    def __len__(self) -> int:
        return len(self.annotations)

    def _load_audio(self, file_name: str) -> np.ndarray:
        if self.cache_audio and file_name in self._audio_cache:
            return self._audio_cache[file_name]
        y, _ = librosa.load(self.audio_dir / file_name, sr=TARGET_SR, mono=True)
        y = y.astype(np.float32)
        if self.cache_audio:
            self._audio_cache[file_name] = y
        return y

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ann = self.annotations[index]

        call_len = ann.duration_s
        max_pre_roll = max(CHUNK_SECONDS - call_len, 0.0)
        if self.train and max_pre_roll > 0:
            pre_roll = random.uniform(0.0, max_pre_roll)
        else:
            pre_roll = min(1.0, max_pre_roll)
        chunk_start = max(ann.start_s - pre_roll, 0.0)
        chunk_end = chunk_start + CHUNK_SECONDS

        y = self._load_audio(ann.file_name)
        s0 = int(round(chunk_start * TARGET_SR))
        s1 = s0 + CHUNK_SAMPLES
        chunk = y[s0:s1]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        chunk = chunk.astype(np.float32)

        # SNR jitter: inject pure mechanical noise at a random level.
        if (
            self.train
            and self.noise_bank is not None
            and not self.noise_bank.is_empty()
            and random.random() < self.snr_jitter_prob
        ):
            noise_chunk = self.noise_bank.sample()
            target_snr_db = random.uniform(*self.snr_range)
            chunk = mix_at_snr(chunk, noise_chunk, target_snr_db)

        spec = _compute_spec(chunk)

        windowed_calls: List[Tuple[float, float, Optional[float], Optional[float]]] = []
        for other in self._by_file.get(ann.file_name, ()):
            if other.end_s <= chunk_start or other.start_s >= chunk_end:
                continue
            rel_start = max(other.start_s - chunk_start, 0.0)
            rel_end = min(other.end_s - chunk_start, CHUNK_SECONDS)
            windowed_calls.append((rel_start, rel_end, other.low_hz, other.high_hz))
        mask = _rectangle_mask(windowed_calls)

        spec_t = torch.from_numpy(spec).unsqueeze(0)
        mask_t = torch.from_numpy(mask).unsqueeze(0)
        return spec_t, mask_t


# ---------------------------------------------------------------------------
# Stage-2 dataset: synthetic 2-caller mixtures
# ---------------------------------------------------------------------------

class InstanceMixDataset(Dataset):
    """Dynamically-synthesised mixtures of two calls from different files.

    Output tuple:
        spec        [1, FREQ_CROP_BINS, N_FRAMES] log-magnitude of the mixture
        mask        [1, FREQ_CROP_BINS, N_FRAMES] union rumble mask
        instance_y  [K=2, FREQ_CROP_BINS, N_FRAMES] Ideal Binary Mask labels

    The IBM labelling is the standard supervision signal for Deep Clustering:
    for each T-F bin inside the rumble mask, whichever source has the larger
    isolated magnitude "wins" and gets labelled 1 in its channel, the other
    gets 0. This gives hard, exclusive assignments suitable for affinity
    training even though the two sources can overlap in time and frequency
    in the mixture.
    """

    def __init__(
        self,
        annotations: Union[str, os.PathLike, Sequence[CallAnnotation]],
        audio_dir: Union[str, os.PathLike],
        num_sources: int = 2,
        length: Optional[int] = None,
        cache_audio: bool = True,
        gain_jitter_db: Tuple[float, float] = (-3.0, 3.0),
    ) -> None:
        if num_sources != 2:
            raise ValueError("InstanceMixDataset currently only supports K=2 sources")
        self.audio_dir = Path(audio_dir)
        self.num_sources = num_sources
        self.cache_audio = bool(cache_audio)
        self.gain_jitter_db = gain_jitter_db
        self._audio_cache: Dict[str, np.ndarray] = {}

        if isinstance(annotations, (str, os.PathLike)):
            self.annotations: List[CallAnnotation] = load_raven_annotations(annotations)
        else:
            self.annotations = list(annotations)

        self._by_file: Dict[str, List[int]] = {}
        for idx, ann in enumerate(self.annotations):
            self._by_file.setdefault(ann.file_name, []).append(idx)
        self._file_names: List[str] = list(self._by_file.keys())

        if len(self._file_names) < 2:
            raise RuntimeError(
                "InstanceMixDataset needs annotations from at least 2 files; "
                f"found {len(self._file_names)}."
            )

        self._length = int(length) if length is not None else len(self.annotations)

    def __len__(self) -> int:
        return self._length

    def _load_audio(self, file_name: str) -> np.ndarray:
        if self.cache_audio and file_name in self._audio_cache:
            return self._audio_cache[file_name]
        y, _ = librosa.load(self.audio_dir / file_name, sr=TARGET_SR, mono=True)
        y = y.astype(np.float32)
        if self.cache_audio:
            self._audio_cache[file_name] = y
        return y

    def _extract_chunk(
        self, ann: CallAnnotation
    ) -> Tuple[np.ndarray, float, float]:
        """Return (chunk, rel_start, rel_end) where rel times are relative to chunk start."""
        call_len = ann.duration_s
        max_pre_roll = max(CHUNK_SECONDS - call_len, 0.0)
        pre_roll = random.uniform(0.0, max_pre_roll) if max_pre_roll > 0 else 0.0
        chunk_start = max(ann.start_s - pre_roll, 0.0)
        y = self._load_audio(ann.file_name)
        s0 = int(round(chunk_start * TARGET_SR))
        s1 = s0 + CHUNK_SAMPLES
        chunk = y[s0:s1]
        if len(chunk) < CHUNK_SAMPLES:
            chunk = np.pad(chunk, (0, CHUNK_SAMPLES - len(chunk)))
        rel_start = max(ann.start_s - chunk_start, 0.0)
        rel_end = min(ann.end_s - chunk_start, CHUNK_SECONDS)
        return chunk.astype(np.float32), rel_start, rel_end

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Pick a pair of DIFFERENT files so the two callers are never the same
        # elephant (which would collapse to a trivial clustering task).
        file_a, file_b = random.sample(self._file_names, 2)
        ann_a = self.annotations[random.choice(self._by_file[file_a])]
        ann_b = self.annotations[random.choice(self._by_file[file_b])]

        raw_a, rel_a_start, rel_a_end = self._extract_chunk(ann_a)
        raw_b, rel_b_start, rel_b_end = self._extract_chunk(ann_b)

        unit_a = _unit_power(raw_a)
        unit_b = _unit_power(raw_b)

        gain_db = random.uniform(*self.gain_jitter_db)
        scale_b = float(10.0 ** (gain_db / 20.0))
        mix = unit_a + scale_b * unit_b

        # Peak-normalise mixture to keep samples in [-1, 1]; apply the same
        # normalisation factor to each source so that isolated_a + isolated_b
        # exactly equals the final mixture.
        peak = float(np.max(np.abs(mix)))
        if peak > 0.98:
            norm = 0.98 / peak
            src_a = (unit_a * norm).astype(np.float32)
            src_b = (scale_b * unit_b * norm).astype(np.float32)
            mix = (mix * norm).astype(np.float32)
        else:
            src_a = unit_a.astype(np.float32)
            src_b = (scale_b * unit_b).astype(np.float32)
            mix = mix.astype(np.float32)

        mag_a = _magnitude_stft(src_a)[:FREQ_CROP_BINS, :N_FRAMES]
        mag_b = _magnitude_stft(src_b)[:FREQ_CROP_BINS, :N_FRAMES]
        mag_mix = _magnitude_stft(mix)[:FREQ_CROP_BINS, :N_FRAMES]

        # Rectangular rumble mask: union of both calls' annotation rectangles.
        mask_rect_a = _rectangle_mask([(rel_a_start, rel_a_end, ann_a.low_hz, ann_a.high_hz)])
        mask_rect_b = _rectangle_mask([(rel_b_start, rel_b_end, ann_b.low_hz, ann_b.high_hz)])
        rumble_mask = np.maximum(mask_rect_a, mask_rect_b).astype(np.float32)

        # Ideal Binary Mask instance labels restricted to the rumble region.
        # Each bin exclusively assigned to whichever isolated source is loudest.
        a_dominant = mag_a >= mag_b
        assignable = rumble_mask > 0
        instance_y = np.zeros(
            (self.num_sources, FREQ_CROP_BINS, N_FRAMES), dtype=np.float32
        )
        instance_y[0] = (a_dominant & assignable).astype(np.float32)
        instance_y[1] = ((~a_dominant) & assignable).astype(np.float32)

        spec = _normalise_db(mag_mix)

        spec_t = torch.from_numpy(spec).unsqueeze(0)
        mask_t = torch.from_numpy(rumble_mask).unsqueeze(0)
        y_t = torch.from_numpy(instance_y)
        return spec_t, mask_t, y_t


# ---------------------------------------------------------------------------
# Discovery helpers (used by train.py for auto-locating the spreadsheet)
# ---------------------------------------------------------------------------

def _discover_sheet(root: Path) -> Optional[Path]:
    """Recursively find the first CSV/TSV/XLSX/XLS/TXT sheet under `root`.

    Skips build artefacts: node_modules, CMake build dirs, and anything inside
    `backend/cpp_ext/build`.
    """
    skip_parts = {"node_modules", "build", ".claude"}
    patterns = ("*.csv", "*.tsv", "*.txt", "*.xlsx", "*.xls")
    for pat in patterns:
        for candidate in sorted(root.rglob(pat)):
            parts = set(candidate.parts)
            if parts & skip_parts:
                continue
            if "CMakeLists" in candidate.name or candidate.name.startswith("CMake"):
                continue
            return candidate
    return None


def _synthesize_dummy_annotations(audio_dir: Path) -> List[CallAnnotation]:
    """Fallback annotations for smoke-testing before the real CSV lands.

    Pretends the first 5 wavs each contain a 2-4 second call so the loader
    can be exercised end-to-end without blocking on the spreadsheet.
    """
    wavs = sorted(audio_dir.glob("*.wav"))[:5]
    return [CallAnnotation(file_name=w.name, start_s=2.0, end_s=4.0) for w in wavs]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from torch.utils.data import DataLoader

    ROOT = Path(__file__).resolve().parent.parent
    AUDIO_DIR = ROOT / "Audio Files (04-10-2026)"

    sheet = _discover_sheet(ROOT)
    if sheet is not None:
        print(f"[dataset] Using annotations from {sheet}")
        annotations = load_raven_annotations(sheet)
    else:
        print("[dataset] No spreadsheet found - using synthesized dummy annotations.")
        annotations = _synthesize_dummy_annotations(AUDIO_DIR)

    # Baseline dataset with SNR jitter
    noise_bank = NoiseBank(AUDIO_DIR, annotations, chunk_samples=CHUNK_SAMPLES)
    print(f"[dataset] noise bank: {'empty' if noise_bank.is_empty() else 'ready'}")

    base = ElephantSpectrogramDataset(
        annotations, AUDIO_DIR, train=True, noise_bank=noise_bank
    )
    base_loader = DataLoader(base, batch_size=min(4, len(base)), shuffle=True)
    specs, masks = next(iter(base_loader))
    print(f"[dataset] baseline  specs={tuple(specs.shape)} masks={tuple(masks.shape)} "
          f"pos_frac={masks.mean():.4f}")

    # Instance mixture dataset (if at least 2 source files)
    file_count = len({a.file_name for a in annotations})
    if file_count >= 2:
        inst = InstanceMixDataset(annotations, AUDIO_DIR, num_sources=2)
        inst_loader = DataLoader(inst, batch_size=min(4, len(inst)), shuffle=True)
        specs, masks, y = next(iter(inst_loader))
        print(f"[dataset] instance  specs={tuple(specs.shape)} "
              f"masks={tuple(masks.shape)} y={tuple(y.shape)} "
              f"y_sum_per_src={y.sum(dim=(0, 2, 3)).tolist()}")
    else:
        print(f"[dataset] instance  skipped (need >=2 files, got {file_count})")
    print("[dataset] OK")
