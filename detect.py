"""
Precision-First Elephant Rumble Detector.

Architecture: Multiplicative binary-mask gating (NOT additive scoring).

Stage 1 -- STFT on raw magnitude spectrogram (no PCEN, no background
           subtraction -- these were boosting noise artifacts).

Stage 2 -- THREE independent binary masks per frame:
    Mask A: pYIN voicing gate (HARD REQUIREMENT, primary discriminator)
    Mask B: Harmonic salience contrast (elephant vs mechanical band)
    Mask C: F0 stability (rejects generators and random noise)

Stage 3 -- Fused mask = A AND (B OR C)
           Voicing is REQUIRED.  Then either harmonic contrast OR F0
           stability must also pass.

Stage 4 -- Morphological segmentation with duration filtering.

Stage 5 -- Confidence scoring via mean voiced_prob per detection.

Usage:
    python detect.py                        # default audio folder
    python detect.py /path/to/audio_dir     # custom folder
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np
import pandas as pd
from scipy.ndimage import label as nd_label
from scipy.ndimage import binary_closing, binary_opening

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_SR: int = 4000
N_FFT: int = 8192
HOP_LENGTH: int = 256
FREQ_RES: float = TARGET_SR / N_FFT  # ~0.488 Hz/bin

# pYIN parameters
F0_MIN: float = 8.0
F0_MAX: float = 50.0
PYIN_FRAME_LENGTH: int = 2048

# Mask A: Voicing gate
VOICING_THRESHOLD: float = 0.2

# Mask B: Harmonic salience contrast
N_HARMONICS: int = 15
MAX_HARMONIC_HZ: float = 1000.0
HARMONIC_DECAY: float = 0.7
F0_STEP: float = 0.25
CONTRAST_F0_SPLIT: float = 50.0
CONTRAST_MIN_RATIO: float = 1.3  # elephant_sal > 1.3 * mechanical_sal

# Mask C: F0 stability
STABILITY_WINDOW_S: float = 1.0
STABILITY_MIN_STD: float = 0.2
STABILITY_MAX_STD: float = 6.0

# Segmentation
SEG_MORPH_CLOSE_S: float = 1.0
SEG_MORPH_OPEN_S: float = 0.4
SEG_MIN_DURATION_S: float = 1.0
SEG_MERGE_GAP_S: float = 2.0
SEG_MAX_DURATION_S: float = 12.0

# Confidence floor
CONFIDENCE_FLOOR: float = 0.10


# ---------------------------------------------------------------------------
# Stage 1: Raw STFT spectrogram
# ---------------------------------------------------------------------------

def _raw_spectrogram(y: np.ndarray) -> np.ndarray:
    """Compute raw magnitude spectrogram. No PCEN, no background subtraction."""
    S = np.abs(
        librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True)
    ).astype(np.float64)
    return S


# ---------------------------------------------------------------------------
# Stage 2: Three independent binary masks
# ---------------------------------------------------------------------------

def _mask_a_voicing(y: np.ndarray, sr: int, n_frames: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mask A: pYIN voicing gate (HARD REQUIREMENT).

    Returns:
        mask_a:      (n_frames,) bool -- True if voiced_prob > threshold
        voiced_prob: (n_frames,) float -- raw voicing probability
        f0_pyin:     (n_frames,) float -- estimated F0 (NaN if unvoiced)
    """
    try:
        f0, vflag, vprob = librosa.pyin(
            y, fmin=F0_MIN, fmax=F0_MAX, sr=sr,
            frame_length=PYIN_FRAME_LENGTH, hop_length=HOP_LENGTH,
        )
    except Exception:
        z = np.zeros(n_frames)
        return np.zeros(n_frames, dtype=bool), z, np.full(n_frames, np.nan)

    if f0 is None:
        f0 = np.full(n_frames, np.nan)
    if vprob is None:
        vprob = np.zeros(n_frames)

    f0 = np.asarray(f0, dtype=np.float64)
    vprob = np.asarray(vprob, dtype=np.float64)

    # Pad or trim to n_frames
    if len(vprob) < n_frames:
        vprob = np.pad(vprob, (0, n_frames - len(vprob)))
        f0 = np.pad(f0, (0, n_frames - len(f0)), constant_values=np.nan)
    else:
        vprob = vprob[:n_frames]
        f0 = f0[:n_frames]

    mask_a = vprob > VOICING_THRESHOLD
    return mask_a, vprob, f0


def _mask_b_harmonic_contrast(S: np.ndarray) -> np.ndarray:
    """Mask B: Harmonic salience contrast on RAW spectrogram (NOT normalized).

    Computes Klapuri-style harmonic salience for elephant band (8-50 Hz F0)
    and mechanical band (50-100 Hz F0). Frame passes if elephant salience
    exceeds mechanical salience by CONTRAST_MIN_RATIO.

    Uses ABSOLUTE salience values -- no per-file normalization.

    Returns:
        mask_b: (n_frames,) bool
    """
    n_freq, n_frames = S.shape

    # Candidate F0 grid covering both bands
    candidates = np.arange(F0_MIN, 100.0 + F0_STEP * 0.5, F0_STEP)
    harmonics = np.arange(1, N_HARMONICS + 1, dtype=np.float64)

    # Harmonic bin indices: (n_candidates, n_harmonics)
    hz = candidates[:, None] * harmonics[None, :]
    bins = np.round(hz / FREQ_RES).astype(np.intp)
    weights = HARMONIC_DECAY ** (harmonics - 1)

    valid = (hz <= MAX_HARMONIC_HZ) & (bins < n_freq) & (bins >= 0)
    w2d = np.where(valid, weights[None, :], 0.0)
    safe_bins = np.clip(bins, 0, n_freq - 1)

    # Salience: (n_candidates, n_frames)
    mag = S[safe_bins, :]  # (n_candidates, n_harmonics, n_frames)
    salience = np.einsum("cht,ch->ct", mag, w2d)

    # Split into elephant and mechanical bands
    elephant_mask_idx = candidates <= CONTRAST_F0_SPLIT
    mechanical_mask_idx = candidates > CONTRAST_F0_SPLIT

    sal_elephant = salience[elephant_mask_idx, :]     # (n_el, n_frames)
    sal_mechanical = salience[mechanical_mask_idx, :]  # (n_mech, n_frames)

    best_el_sal = sal_elephant.max(axis=0)
    best_mech_sal = sal_mechanical.max(axis=0) if sal_mechanical.shape[0] > 0 else np.zeros(n_frames)

    # Best F0 in elephant band (for output)
    best_el_idx = np.argmax(sal_elephant, axis=0)
    el_candidates = candidates[elephant_mask_idx]
    best_f0 = el_candidates[best_el_idx]

    # Contrast gate: elephant salience must exceed mechanical by ratio
    mask_b = best_el_sal > (CONTRAST_MIN_RATIO * best_mech_sal)

    return mask_b, best_f0


def _mask_c_f0_stability(f0_contour: np.ndarray, sr: int) -> np.ndarray:
    """Mask C: F0 stability via rolling std.

    Frame passes if rolling std of F0 is in [STABILITY_MIN_STD, STABILITY_MAX_STD].
    This rejects:
      - Generators: near-zero std (constant RPM)
      - Random noise: very high std (>5 Hz)

    Returns:
        mask_c: (n_frames,) bool
    """
    n = len(f0_contour)
    w = max(1, int(round(STABILITY_WINDOW_S * sr / HOP_LENGTH)))

    f0_valid = np.where(np.isfinite(f0_contour), f0_contour, 0.0)
    is_finite = np.isfinite(f0_contour).astype(np.float64)

    kernel = np.ones(w)
    count = np.convolve(is_finite, kernel, mode="same")
    count = np.maximum(count, 1.0)

    s1 = np.convolve(f0_valid * is_finite, kernel, mode="same")
    s2 = np.convolve(f0_valid ** 2 * is_finite, kernel, mode="same")

    mean = s1 / count
    var = np.maximum(s2 / count - mean ** 2, 0.0)
    std = np.sqrt(var)

    # Need at least 3 voiced frames in the window for a meaningful std
    std = np.where(count > 2, std, 0.0)

    mask_c = (std >= STABILITY_MIN_STD) & (std <= STABILITY_MAX_STD)
    return mask_c


# ---------------------------------------------------------------------------
# Stage 3: Fused binary mask
# ---------------------------------------------------------------------------

def _fuse_masks(mask_a: np.ndarray, mask_b: np.ndarray, mask_c: np.ndarray) -> np.ndarray:
    """Multiplicative gating: final = A AND (B OR C).

    Voicing is REQUIRED. Then either harmonic contrast OR stability must pass.
    """
    return mask_a & (mask_b | mask_c)


# ---------------------------------------------------------------------------
# Stage 4: Segmentation
# ---------------------------------------------------------------------------

def _segment_mask(
    mask: np.ndarray,
    voiced_prob: np.ndarray,
    f0s: np.ndarray,
    times: np.ndarray,
) -> List[dict]:
    """Convert binary mask to discrete call detections.

    Steps:
        1. Morphological closing (fill gaps within calls)
        2. Morphological opening (remove speckles)
        3. Connected components -> intervals
        4. Merge intervals closer than merge_gap
        5. Duration filtering (min/max)
        6. Confidence scoring via mean(voiced_prob)
    """
    if mask.size == 0:
        return []

    frame_dt = HOP_LENGTH / TARGET_SR

    # Step 1: Morphological closing
    close_frames = max(1, int(round(SEG_MORPH_CLOSE_S / frame_dt)))
    struct_close = np.ones(close_frames, dtype=bool)
    mask = binary_closing(mask, structure=struct_close).astype(bool)

    # Step 2: Morphological opening
    open_frames = max(1, int(round(SEG_MORPH_OPEN_S / frame_dt)))
    struct_open = np.ones(open_frames, dtype=bool)
    mask = binary_opening(mask, structure=struct_open).astype(bool)

    # Step 3: Connected components
    labeled, n_regions = nd_label(mask)
    if n_regions == 0:
        return []

    raw_regions: List[Tuple[int, int]] = []
    for rid in range(1, n_regions + 1):
        frames = np.where(labeled == rid)[0]
        raw_regions.append((int(frames[0]), int(frames[-1] + 1)))

    # Convert to seconds
    def _f2t(fi: int) -> float:
        fi = min(fi, len(times) - 1)
        return float(times[fi])

    sec_regions = [(_f2t(s), _f2t(e), s, e) for s, e in raw_regions]

    # Step 4: Merge close regions
    merged: List[Tuple[float, float, int, int]] = []
    for s, e, si, ei in sec_regions:
        if merged and (s - merged[-1][1]) <= SEG_MERGE_GAP_S:
            merged[-1] = (merged[-1][0], e, merged[-1][2], ei)
        else:
            merged.append((s, e, si, ei))

    # Step 5: Duration filter
    filtered = [(s, e, si, ei) for s, e, si, ei in merged
                if (e - s) >= SEG_MIN_DURATION_S]

    # Step 6: Split long regions at valleys
    final: List[Tuple[float, float, int, int]] = []
    for s, e, si, ei in filtered:
        if (e - s) <= SEG_MAX_DURATION_S:
            final.append((s, e, si, ei))
        else:
            _split_region(voiced_prob, times, si, ei, SEG_MAX_DURATION_S, final)

    # Step 7: Build output with confidence scoring
    detections: List[dict] = []
    for s, e, si, ei in final:
        # Confidence = mean voicing probability in this region
        vp_slice = voiced_prob[si:ei]
        confidence = float(vp_slice.mean()) if vp_slice.size > 0 else 0.0

        # Only keep detections above confidence floor
        if confidence < CONFIDENCE_FLOOR:
            continue

        f0_slice = f0s[si:ei]
        valid_f0 = f0_slice[np.isfinite(f0_slice) & (f0_slice > 0)]
        median_f0 = float(np.median(valid_f0)) if valid_f0.size > 0 else 0.0

        detections.append({
            "start_s": round(s, 4),
            "end_s": round(e, 4),
            "call_type": "rumble",
            "confidence": round(confidence, 4),
            "median_f0_hz": round(median_f0, 2),
        })

    return detections


def _split_region(
    score: np.ndarray, times: np.ndarray,
    si: int, ei: int, max_dur: float,
    output: List[Tuple[float, float, int, int]],
) -> None:
    """Recursively split a long region at its deepest valley."""
    s = float(times[min(si, len(times) - 1)])
    e = float(times[min(ei, len(times) - 1)])
    if (e - s) <= max_dur or (ei - si) < 4:
        output.append((s, e, si, ei))
        return
    margin = max(1, (ei - si) // 10)
    lo, hi = si + margin, ei - margin
    if hi <= lo:
        output.append((s, e, si, ei))
        return
    valley = lo + int(np.argmin(score[lo:hi]))
    _split_region(score, times, si, valley, max_dur, output)
    _split_region(score, times, valley, ei, max_dur, output)


# ---------------------------------------------------------------------------
# Top-level API
# ---------------------------------------------------------------------------

def detect_calls_in_file(wav_path: str) -> List[dict]:
    """Detect elephant rumble calls in a single WAV file."""
    try:
        y, sr = librosa.load(str(wav_path), sr=TARGET_SR, mono=True)
    except Exception as exc:
        print(f"[detect] WARNING: could not load {wav_path}: {exc}")
        return []
    if y.size == 0:
        return []

    # Stage 1: Raw STFT
    S_raw = _raw_spectrogram(y)
    n_freq, n_frames = S_raw.shape

    if n_frames == 0:
        return []

    # Stage 2: Three independent binary masks
    mask_a, voiced_prob, f0_pyin = _mask_a_voicing(y, sr, n_frames)
    mask_b_result = _mask_b_harmonic_contrast(S_raw)
    mask_b, best_f0 = mask_b_result
    mask_c = _mask_c_f0_stability(f0_pyin, sr)

    # Align lengths (pYIN may differ slightly from STFT)
    n = min(n_frames, len(mask_a), len(mask_b), len(mask_c))
    mask_a = mask_a[:n]
    mask_b = mask_b[:n]
    mask_c = mask_c[:n]
    voiced_prob = voiced_prob[:n]
    f0_pyin = f0_pyin[:n]
    best_f0 = best_f0[:n]

    # Stage 3: Fused binary mask (multiplicative gating)
    fused_mask = _fuse_masks(mask_a, mask_b, mask_c)

    # Stage 4+5: Segmentation and confidence scoring
    times = librosa.frames_to_time(np.arange(n), sr=sr, hop_length=HOP_LENGTH)

    return _segment_mask(fused_mask, voiced_prob, best_f0, times)


def detect_all_files(audio_dir) -> pd.DataFrame:
    """Run detector on all .wav files in a directory."""
    columns = [
        "Selection", "Sound_file", "Start_time", "End_time",
        "Call_type", "Confidence", "Median_F0_Hz",
    ]
    audio_path = Path(audio_dir)
    if not audio_path.is_dir():
        return pd.DataFrame(columns=columns)

    wav_files = sorted(audio_path.glob("*.wav"))
    if not wav_files:
        return pd.DataFrame(columns=columns)

    rows: List[dict] = []
    sel = 0
    for wf in wav_files:
        print(f"[detect] Processing {wf.name} ...", end=" ", flush=True)
        dets = detect_calls_in_file(str(wf))
        print(f"{len(dets)} detection(s)")
        for d in dets:
            sel += 1
            rows.append({
                "Selection": sel,
                "Sound_file": wf.name,
                "Start_time": d["start_s"],
                "End_time": d["end_s"],
                "Call_type": d["call_type"],
                "Confidence": d["confidence"],
                "Median_F0_Hz": d["median_f0_hz"],
            })

    return pd.DataFrame(rows, columns=columns) if rows else pd.DataFrame(columns=columns)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    default_dir = root / "Audio Files (04-10-2026)"

    audio_dir = sys.argv[1] if len(sys.argv) > 1 else str(default_dir)
    if not Path(audio_dir).is_dir():
        print(f"[detect] Directory not found: {audio_dir}")
        sys.exit(1)

    print(f"[detect] Scanning: {audio_dir}")
    results = detect_all_files(audio_dir)

    if results.empty:
        print("[detect] No detections.")
    else:
        print(f"\n[detect] {len(results)} total detection(s):\n")
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 140)
        pd.set_option("display.max_rows", 200)
        print(results.to_string(index=False))
        out = root / "predictions.csv"
        results.to_csv(out, index=False)
        print(f"\n[detect] Saved to {out}")
    print("[detect] Done.")
