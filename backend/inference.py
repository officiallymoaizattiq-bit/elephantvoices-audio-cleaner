"""
Elephant-rumble isolation inference pipeline.

Two-stage approach:
    1. Baseline U-Net masks the 10-1000 Hz rumble band (noise removal)
    2. Annotation-driven harmonic tracking separates overlapping callers

The neural network handles what it's good at (frequency-domain noise masking).
Classical DSP handles what it's good at (harmonic source separation using the
known physics: elephant harmonics are integer multiples of F0 and don't cross
within one call).

NOTE on hop-length mismatch:
    The U-Net was trained with HOP_LENGTH=1024 (from dataset.py).
    The harmonic separator now operates at HOP_LENGTH=256 for finer temporal
    resolution.  The noise mask produced by the U-Net is therefore 4x coarser
    along the time axis than what the separator expects.  We upsample the mask
    by 4x (scipy.ndimage.zoom) before handing it to the separator.

NOTE on soft masking:
    Previous versions thresholded the sigmoid output to a hard binary mask.
    We now pass the raw sigmoid probabilities through as a soft mask.  This
    preserves gradient information and yields smoother noise suppression.
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
from pathlib import Path
from typing import List, Optional, Tuple

import librosa
import numpy as np
import scipy.ndimage
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "model"))

from data_engine import CallAnnotation, load_raven_annotations  # noqa: E402
from dataset import FREQ_CROP_BINS, HOP_LENGTH, N_FFT, TARGET_SR, TOP_DB  # noqa: E402
from harmonic_separator import process_file_with_annotations  # noqa: E402
from unet import UNet  # noqa: E402

# Optional C++ acceleration for mask application
try:
    from fast_mask_ext import apply_mask as _cpp_apply_mask  # noqa: F401

    _HAS_FAST_MASK = True
except ImportError:
    _HAS_FAST_MASK = False

# ---------------------------------------------------------------------------
# Hop-length constants
# ---------------------------------------------------------------------------
# The U-Net was trained at this hop (imported from dataset.py as HOP_LENGTH).
UNET_HOP = HOP_LENGTH  # 1024

# The harmonic separator now runs at a finer hop for better temporal resolution.
SEPARATOR_HOP = 256

# The ratio between the two hops — used to upsample the mask along the time axis.
HOP_UPSAMPLE_FACTOR = UNET_HOP // SEPARATOR_HOP  # 4

# ---------------------------------------------------------------------------
# Weights + model (baseline only — no embedding head needed)
# ---------------------------------------------------------------------------

WEIGHTS_DIR = PROJECT_ROOT / "backend" / "weights"
BASELINE_WEIGHTS = WEIGHTS_DIR / "unet_baseline.pth"
LEGACY_WEIGHTS = WEIGHTS_DIR / "unet_latest.pth"
UNET_POOLING_FACTOR = 16

_MODEL_CACHE: Optional[Tuple[UNet, torch.device]] = None
_ANNOTATIONS_CACHE: Optional[List[CallAnnotation]] = None


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_weights() -> Path:
    if BASELINE_WEIGHTS.exists():
        return BASELINE_WEIGHTS
    if LEGACY_WEIGHTS.exists():
        return LEGACY_WEIGHTS
    raise FileNotFoundError(
        "No trained weights found. Run `python model/train.py` first."
    )


def get_model() -> Tuple[UNet, torch.device]:
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        device = pick_device()
        weights_path = _resolve_weights()
        model = UNet(in_channels=1, out_channels=1, embedding_dim=0).to(device)
        state = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state, strict=False)
        model.eval()
        _MODEL_CACHE = (model, device)
        print(f"[infer] loaded {weights_path.name} on {device}")
    return _MODEL_CACHE


def get_annotations() -> List[CallAnnotation]:
    global _ANNOTATIONS_CACHE
    if _ANNOTATIONS_CACHE is None:
        csv_files = list(PROJECT_ROOT.glob("*.csv"))
        if csv_files:
            _ANNOTATIONS_CACHE = load_raven_annotations(csv_files[0])
            print(f"[infer] loaded {len(_ANNOTATIONS_CACHE)} annotations from {csv_files[0].name}")
        else:
            _ANNOTATIONS_CACHE = []
            print("[infer] no CSV found — overlap detection disabled")
    return _ANNOTATIONS_CACHE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_db(mag: np.ndarray) -> np.ndarray:
    mag_db = librosa.amplitude_to_db(mag, ref=1.0, top_db=None)
    mag_db = np.clip(mag_db, -TOP_DB, 0.0)
    return ((mag_db + TOP_DB) / TOP_DB).astype(np.float32)


def _pad_time(mag: np.ndarray, multiple: int) -> Tuple[np.ndarray, int]:
    T = mag.shape[1]
    pad = (-T) % multiple
    if pad > 0:
        mag = np.pad(mag, ((0, 0), (0, pad)), mode="edge")
    return mag, T


def _upsample_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Upsample a (F, T) mask along the time axis by *factor*.

    Uses scipy.ndimage.zoom with order=1 (bilinear) to smoothly
    interpolate the soft mask values.  The frequency axis is left
    unchanged (zoom factor 1.0).
    """
    if factor == 1:
        return mask
    return scipy.ndimage.zoom(mask, zoom=(1.0, float(factor)), order=1).astype(np.float32)


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

def _get_noise_mask(
    y: np.ndarray, model: UNet, device: torch.device,
) -> np.ndarray:
    """Run the baseline U-Net and return a soft (FREQ_CROP_BINS, T) mask.

    The mask contains raw sigmoid probabilities (no hard threshold).
    It is computed at the U-Net's training hop (UNET_HOP = 1024).
    """
    complex_stft = librosa.stft(y, n_fft=N_FFT, hop_length=UNET_HOP, center=True)
    mag = np.abs(complex_stft).astype(np.float32)
    mag_norm = _normalise_db(mag[:FREQ_CROP_BINS])
    padded, orig_T = _pad_time(mag_norm, UNET_POOLING_FACTOR)
    spec = torch.from_numpy(padded).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(spec)
        if isinstance(logits, tuple):
            logits = logits[0]
        mask = torch.sigmoid(logits)[0, 0].cpu().numpy()

    # Return soft mask — no thresholding.  Trim to the original time extent.
    return mask[:, :orig_T].astype(np.float32)


def _apply_mask_to_stft(
    complex_stft: np.ndarray, noise_mask: np.ndarray,
) -> np.ndarray:
    """Element-wise multiply *complex_stft* by a real-valued soft mask.

    Uses the C++ fast_mask_ext when available; otherwise falls back to NumPy.
    The mask is zero-padded/cropped to match the STFT shape.
    """
    F_stft, T_stft = complex_stft.shape
    F_m, T_m = noise_mask.shape
    full_mask = np.zeros((F_stft, T_stft), dtype=np.float32)
    fmin = min(F_m, F_stft)
    tmin = min(T_m, T_stft)
    full_mask[:fmin, :tmin] = noise_mask[:fmin, :tmin]

    if _HAS_FAST_MASK:
        return _cpp_apply_mask(complex_stft, full_mask)
    return full_mask.astype(complex_stft.dtype) * complex_stft


def clean_audio(
    wav_path,
    model: UNet,
    device: torch.device,
    annotations: List[CallAnnotation],
) -> Tuple[List[np.ndarray], int]:
    """Clean one wav file: noise removal + optional harmonic separation.

    Returns (waveforms, K) where K is the number of detected callers.
    """
    y, _ = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    y = y.astype(np.float32)

    # Step 1: get the soft noise mask from the baseline U-Net (at UNET_HOP=1024)
    noise_mask_1024 = _get_noise_mask(y, model, device)

    # Step 2: upsample the mask from UNET_HOP frames to SEPARATOR_HOP frames
    # so it aligns with the harmonic separator's finer-resolution STFT.
    noise_mask_256 = _upsample_mask(noise_mask_1024, HOP_UPSAMPLE_FACTOR)

    # Step 3: find annotations for this file
    file_name = wav_path.name
    file_anns = [a for a in annotations if a.file_name == file_name]

    if not file_anns:
        # No annotations — apply noise mask at the U-Net's native hop and return
        complex_stft = librosa.stft(y, n_fft=N_FFT, hop_length=UNET_HOP, center=True)
        masked_stft = _apply_mask_to_stft(complex_stft, noise_mask_1024)
        cleaned = librosa.istft(
            masked_stft, hop_length=UNET_HOP, n_fft=N_FFT, length=len(y),
        )
        print(f"[clean_audio] no annotations for {file_name} -> K=1 (noise removal only)")
        return [cleaned.astype(np.float32)], 1

    # Step 4: harmonic separation handles both solo and overlap cases.
    # Pass the upsampled (256-hop) mask so the separator can apply it
    # at its own temporal resolution.
    result = process_file_with_annotations(
        y, TARGET_SR, file_anns, noise_mask=noise_mask_256,
    )
    K = len(result.caller_waveforms)
    f0_str = ", ".join(
        f"{f:.1f}Hz" if f else "?" for f in result.caller_f0s[:K]
    )
    n_overlaps = len(result.overlap_groups)
    print(
        f"[clean_audio] {file_name}: K={K}, "
        f"overlaps={n_overlaps}, F0s=[{f0_str}]"
    )
    return result.caller_waveforms, K


def process_audio(input_path: str, output_dir: str, original_filename: str = "") -> List[str]:
    """End-to-end pipeline. Returns list of output wav paths.

    `original_filename` is the user's original wav filename (e.g.
    '04-040920-02_vehicle_1.wav'). The API passes this so the harmonic
    separator can match it against CSV annotations even though the upload
    was saved to a temp path with a UUID name.
    """
    model, device = get_model()
    annotations = get_annotations()

    wav_path = Path(input_path)
    if original_filename:
        # Create a wrapper path that has the original name for annotation matching
        # but points to the actual file on disk
        class _NamedPath:
            def __init__(self, real_path: str, display_name: str) -> None:
                self._real = Path(real_path)
                self.name = display_name

            def __str__(self) -> str:
                return str(self._real)

            def __fspath__(self) -> str:
                return str(self._real)

        wav_path = _NamedPath(input_path, original_filename)

    outputs, K = clean_audio(wav_path, model, device, annotations)

    output_dir_p = Path(output_dir)
    output_dir_p.mkdir(parents=True, exist_ok=True)

    written: List[str] = []
    if K == 1 and len(outputs) == 1:
        p = output_dir_p / "cleaned_output.wav"
        sf.write(p, outputs[0], TARGET_SR)
        written.append(str(p))
    else:
        for idx, wave in enumerate(outputs, 1):
            p = output_dir_p / f"cleaned_caller_{idx}.wav"
            sf.write(p, wave, TARGET_SR)
            written.append(str(p))
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    model, device = get_model()
    annotations = get_annotations()

    audio_dir = PROJECT_ROOT / "Audio Files (04-10-2026)"
    if len(sys.argv) >= 2:
        input_wav = Path(sys.argv[1])
        if not input_wav.is_absolute():
            input_wav = PROJECT_ROOT / input_wav
    else:
        wavs = sorted(audio_dir.glob("*.wav"))
        if not wavs:
            raise FileNotFoundError(f"No .wav files in {audio_dir}")
        input_wav = wavs[0]

    print(f"[infer] input: {input_wav.name}")
    demo_out = PROJECT_ROOT / "backend" / "outputs_demo"
    demo_out.mkdir(exist_ok=True)
    for p in demo_out.glob("cleaned_*.wav"):
        p.unlink()

    paths = process_audio(str(input_wav), str(demo_out))
    print(f"[infer] {len(paths)} output(s):")
    for p in paths:
        info = sf.info(p)
        print(f"  {os.path.basename(p)} | {info.samplerate}Hz {info.duration:.2f}s")


if __name__ == "__main__":
    main()
