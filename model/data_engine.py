"""
Data engineering primitives for the elephant-rumble pipeline.

Exports:
    CallAnnotation         : immutable record for one elephant-call interval
    load_raven_annotations : parses CSV / TSV / XLSX / Raven-Pro selection tables
    find_noise_intervals   : returns per-file "negative space" intervals (no calls)
    NoiseBank              : samples pure-mechanical-noise waveform chunks at runtime
    mix_at_snr             : adds two waveforms at a target SNR in dB

Everything operates in plain numpy + librosa so it can be shared between the
PyTorch dataset classes and any non-torch consumers. Downstream modules
(`dataset.py`, `train.py`, `backend/inference.py`) own the torch wrapping.

Design notes worth knowing:

* Column-alias resolution is case-insensitive and covers the three likely
  handoff formats: generic user CSV (`file_name` / `start_time` / `end_time`),
  Raven-Pro selection tables (`Begin File` / `Begin Time (s)` / `End Time (s)`
  / optional `Low Freq (Hz)` / `High Freq (Hz)`), and their xlsx variants.

* `find_noise_intervals` pads each annotated call by a safety margin before
  subtracting it from the file's timeline so we don't accidentally mine
  rumble spillover from imperfect annotation boundaries.

* `NoiseBank` samples weighted by interval duration so a 30-second silence
  contributes 30x more noise chunks than a 1-second gap. Pure-noise audio is
  cached in memory (same trick as `ElephantSpectrogramDataset`).

* `mix_at_snr` uses the standard definition
      SNR_dB = 10 * log10(power_signal / power_noise)
  and scales the noise in place to hit that target. Positive dB means signal
  dominates; -5 to +5 means "roughly balanced, slightly noisy".
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import pandas as pd


TARGET_SR: int = 4000


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

_FILE_ALIASES = [
    "file_name", "filename", "file", "wav",
    "sound_file", "sound file", "soundfile",
    "begin file", "begin path",
]
_START_ALIASES = [
    "start_time", "start", "begin time (s)", "begin_time", "start time (s)",
]
_END_ALIASES = [
    "end_time", "end", "end time (s)", "end_time_s",
]
_LOW_FREQ_ALIASES = ["low freq (hz)", "low_freq", "fmin"]
_HIGH_FREQ_ALIASES = ["high freq (hz)", "high_freq", "fmax"]


@dataclass(frozen=True)
class CallAnnotation:
    """One elephant-call interval within one audio file."""
    file_name: str
    start_s: float
    end_s: float
    low_hz: Optional[float] = None
    high_hz: Optional[float] = None

    @property
    def duration_s(self) -> float:
        return max(self.end_s - self.start_s, 0.0)


def _resolve_column(df: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    """Case-insensitive column lookup; returns None if nothing matches."""
    lower = {c.lower().strip(): c for c in df.columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    return None


def load_raven_annotations(sheet_path: Union[str, os.PathLike]) -> List[CallAnnotation]:
    """Parse a CSV / TSV / XLSX annotations file into a list of CallAnnotation.

    Supports both comma-separated and tab-separated .csv files (Raven Pro
    selection tables are tab-separated despite the .txt extension, and users
    sometimes export them as .csv that is secretly tab-delimited).
    """
    path = Path(sheet_path)
    suffix = path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix in (".tsv", ".txt"):
        df = pd.read_csv(path, sep="\t")
    else:
        df = pd.read_csv(path)
        # If comma parsing collapsed everything into one column, re-read as TSV.
        if df.shape[1] <= 1:
            df = pd.read_csv(path, sep="\t")

    file_col = _resolve_column(df, _FILE_ALIASES)
    start_col = _resolve_column(df, _START_ALIASES)
    end_col = _resolve_column(df, _END_ALIASES)
    if not (file_col and start_col and end_col):
        raise KeyError(
            f"Could not resolve file / start / end columns in {path.name}. "
            f"Columns found: {list(df.columns)}"
        )
    low_col = _resolve_column(df, _LOW_FREQ_ALIASES)
    high_col = _resolve_column(df, _HIGH_FREQ_ALIASES)

    out: List[CallAnnotation] = []
    for _, row in df.iterrows():
        raw_name = str(row[file_col]).strip()
        # Raven's "Begin Path" may be an absolute path; use basename for portability.
        fname = os.path.basename(raw_name)
        out.append(
            CallAnnotation(
                file_name=fname,
                start_s=float(row[start_col]),
                end_s=float(row[end_col]),
                low_hz=(
                    float(row[low_col])
                    if low_col is not None and pd.notna(row[low_col])
                    else None
                ),
                high_hz=(
                    float(row[high_col])
                    if high_col is not None and pd.notna(row[high_col])
                    else None
                ),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Negative-space mining
# ---------------------------------------------------------------------------

def find_noise_intervals(
    file_annotations: Sequence[CallAnnotation],
    audio_duration_s: float,
    min_duration_s: float = 2.0,
    safety_margin_s: float = 0.5,
) -> List[Tuple[float, float]]:
    """Return non-overlapping (start, end) intervals free of annotated calls.

    Each call is padded by `safety_margin_s` on both sides before being
    subtracted from the timeline so that rumble energy spilling past the
    annotated boundary (which is common - annotators draw boxes around the
    visible harmonic stack, not the exact signal envelope) doesn't leak into
    the mined noise chunks.
    """
    if audio_duration_s <= 0:
        return []

    sorted_calls = sorted(file_annotations, key=lambda c: c.start_s)

    busy: List[List[float]] = []
    for call in sorted_calls:
        s = max(0.0, call.start_s - safety_margin_s)
        e = min(audio_duration_s, call.end_s + safety_margin_s)
        if e <= s:
            continue
        if busy and s <= busy[-1][1]:
            busy[-1][1] = max(busy[-1][1], e)
        else:
            busy.append([s, e])

    negative: List[Tuple[float, float]] = []
    cursor = 0.0
    for s, e in busy:
        if s - cursor >= min_duration_s:
            negative.append((cursor, s))
        cursor = e
    if audio_duration_s - cursor >= min_duration_s:
        negative.append((cursor, audio_duration_s))

    return negative


# ---------------------------------------------------------------------------
# Noise bank
# ---------------------------------------------------------------------------

class NoiseBank:
    """Pure-mechanical-noise chunk sampler.

    At construction time, scans every file referenced in `annotations` and
    computes its negative-space intervals via `find_noise_intervals`. At
    sample time, picks an interval (weighted by duration), picks a random
    offset inside it, and returns a waveform chunk of the requested length
    resampled to `target_sr`.

    The sampled audio is pure mechanical noise (cars, airplanes, generators)
    with zero elephant content, which makes it safe to mix back into training
    samples as SNR-jitter augmentation.
    """

    def __init__(
        self,
        audio_dir: Union[str, os.PathLike],
        annotations: Sequence[CallAnnotation],
        chunk_samples: int,
        target_sr: int = TARGET_SR,
        min_interval_s: float = 2.0,
        safety_margin_s: float = 0.5,
        cache_audio: bool = True,
    ) -> None:
        self.audio_dir = Path(audio_dir)
        self.chunk_samples = int(chunk_samples)
        self.target_sr = int(target_sr)
        self.min_interval_s = float(min_interval_s)
        self.safety_margin_s = float(safety_margin_s)
        self.cache_audio = bool(cache_audio)
        self._audio_cache: Dict[str, np.ndarray] = {}

        by_file: Dict[str, List[CallAnnotation]] = {}
        for ann in annotations:
            by_file.setdefault(ann.file_name, []).append(ann)

        self._pool: List[Tuple[str, float, float]] = []
        for file_name, file_anns in by_file.items():
            wav_path = self.audio_dir / file_name
            if not wav_path.exists():
                continue
            try:
                duration = float(librosa.get_duration(path=str(wav_path)))
            except Exception:
                continue
            intervals = find_noise_intervals(
                file_anns,
                audio_duration_s=duration,
                min_duration_s=self.min_interval_s,
                safety_margin_s=self.safety_margin_s,
            )
            for s, e in intervals:
                self._pool.append((file_name, s, e))

        if self._pool:
            durations = np.array([e - s for _, s, e in self._pool], dtype=np.float64)
            self._weights = (durations / durations.sum()).astype(np.float64)
        else:
            self._weights = np.array([], dtype=np.float64)

    def is_empty(self) -> bool:
        return len(self._pool) == 0

    def _load_audio(self, file_name: str) -> np.ndarray:
        if self.cache_audio and file_name in self._audio_cache:
            return self._audio_cache[file_name]
        wav_path = self.audio_dir / file_name
        y, _ = librosa.load(wav_path, sr=self.target_sr, mono=True)
        y = y.astype(np.float32)
        if self.cache_audio:
            self._audio_cache[file_name] = y
        return y

    def sample(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Draw a pure-noise chunk of length `chunk_samples`.

        If the bank is empty (no negative space anywhere in the dataset)
        returns a zero waveform so callers can keep their training loop
        branch-free.
        """
        if self.is_empty():
            return np.zeros(self.chunk_samples, dtype=np.float32)
        if rng is None:
            rng = np.random.default_rng()

        idx = int(rng.choice(len(self._pool), p=self._weights))
        file_name, start_s, end_s = self._pool[idx]

        chunk_seconds = self.chunk_samples / self.target_sr
        upper = max(start_s, end_s - chunk_seconds)
        if upper <= start_s:
            offset_s = start_s
        else:
            offset_s = float(rng.uniform(start_s, upper))

        y = self._load_audio(file_name)
        s0 = int(round(offset_s * self.target_sr))
        s1 = s0 + self.chunk_samples
        chunk = y[s0:s1]
        if len(chunk) < self.chunk_samples:
            chunk = np.pad(chunk, (0, self.chunk_samples - len(chunk)))
        return chunk.astype(np.float32)


# ---------------------------------------------------------------------------
# SNR mixing
# ---------------------------------------------------------------------------

def _signal_power(x: np.ndarray, eps: float = 1e-12) -> float:
    """Mean-square power of a real-valued waveform."""
    return float(np.mean(x.astype(np.float64) ** 2) + eps)


def mix_at_snr(
    signal: np.ndarray,
    noise: np.ndarray,
    target_snr_db: float,
) -> np.ndarray:
    """Add `noise` to `signal` so the resulting SNR equals `target_snr_db`.

    SNR is defined as 10*log10(power_signal / power_noise). We scale the
    noise waveform (never the signal) so that the signal's perceived
    amplitude is preserved and only the mechanical background moves.
    Returns a fresh float32 array - neither input is mutated.
    """
    if signal.shape != noise.shape:
        raise ValueError(
            f"mix_at_snr: shape mismatch signal={signal.shape} noise={noise.shape}"
        )
    p_sig = _signal_power(signal)
    p_noise = _signal_power(noise)
    target_noise_power = p_sig / (10.0 ** (target_snr_db / 10.0))
    scale = float(np.sqrt(target_noise_power / p_noise))
    mixed = signal.astype(np.float32) + (scale * noise).astype(np.float32)
    return mixed.astype(np.float32)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    sig = rng.standard_normal(32000).astype(np.float32)
    noise = rng.standard_normal(32000).astype(np.float32)

    for snr_db in (-5.0, 0.0, 5.0):
        mixed = mix_at_snr(sig, noise, snr_db)
        measured_db = 10.0 * np.log10(_signal_power(sig) / _signal_power(mixed - sig))
        print(f"[data_engine] target {snr_db:+.1f} dB  -> measured {measured_db:+.2f} dB")

    # Exercise the annotation dataclass and negative-space finder on a
    # synthetic "file" with two calls inside a 60-second timeline.
    fake_anns = [
        CallAnnotation("fake.wav", 5.0, 8.0),
        CallAnnotation("fake.wav", 20.0, 23.0),
    ]
    intervals = find_noise_intervals(fake_anns, audio_duration_s=60.0, min_duration_s=2.0)
    print(f"[data_engine] negative space intervals: {intervals}")
    print("[data_engine] OK")
