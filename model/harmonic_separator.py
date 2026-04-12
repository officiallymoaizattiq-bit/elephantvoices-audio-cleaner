"""
Harmonic-tracking multi-caller separator for elephant rumbles.

Uses classical signal processing grounded in the stated physics:
    "Within a single call, the harmonics will never cross each other.
     If there are two different callers, the harmonics of the two calls
     will often cross." — Dr. Mickey Pardo, ElephantVoices

Algorithm:
    1. Check annotations for temporally overlapping calls in the same file.
    2. For each caller in an overlap group, find their "solo" portion
       (the part of their call that does NOT overlap with any other call).
    3. Estimate each caller's fundamental frequency (F0) from their solo
       portion using pYIN (tuned for 8-50 Hz elephant range).
       For engulfed calls with no solo portion, use Klapuri's iterative
       multipitch estimation on the mixed signal.
    4. In the overlap region, construct a TIME-VARYING harmonic comb for
       each caller: the F0 contour is estimated per-frame via pYIN in the
       solo portion and extrapolated into the overlap region, so vibrato
       and onset transients are tracked.
    5. Build per-caller Wiener masks: mask_k = (comb_k * |S|) / (sum_j comb_j * |S| + eps).
       This allocates energy proportionally — if two harmonics from different
       callers land on the same bin, each gets a fraction.
    6. Apply each Wiener mask to the COMPLEX STFT (preserves original phase —
       no Griffin-Lim needed) and ISTFT to reconstruct per-caller audio.

For solo regions (96.3% of annotated call time), no harmonic separation is
needed — the baseline U-Net already handles noise removal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np

from data_engine import CallAnnotation

TARGET_SR = 4000
N_FFT = 8192
HOP_LENGTH = 256  # 0.064s time resolution (was 1024 = 0.256s)
FREQ_RESOLUTION = TARGET_SR / N_FFT  # ~0.488 Hz per bin
MAX_HARMONIC_HZ = 1000.0
HARMONIC_BANDWIDTH_HZ = 2.0  # Gaussian sigma for each harmonic peak
F0_MIN = 8.0
F0_MAX = 50.0
KLAPURI_MAX_SOURCES = 4  # max callers to detect in iterative estimation
KLAPURI_MAX_HARMONICS = 20  # harmonics to sum for salience function


@dataclass
class OverlapGroup:
    """A set of annotations that overlap in time within one file."""
    annotations: List[CallAnnotation]
    group_start: float  # earliest start_s in the group
    group_end: float    # latest end_s in the group


@dataclass
class SeparationResult:
    """Output of separate_overlapping_callers."""
    caller_waveforms: List[np.ndarray]   # one waveform per caller
    caller_f0s: List[Optional[float]]    # estimated F0 per caller (None if failed)
    overlap_groups: List[OverlapGroup]
    sample_rate: int


def find_overlap_groups(
    file_annotations: Sequence[CallAnnotation],
) -> List[OverlapGroup]:
    """Find groups of temporally overlapping annotations in one file.

    Two annotations overlap if the start of one falls before the end of the
    other. Connected overlaps are merged into groups (e.g., if A overlaps B
    and B overlaps C, all three form one group even if A and C don't directly
    overlap). Returns only groups with 2+ members.
    """
    if len(file_annotations) < 2:
        return []

    sorted_anns = sorted(file_annotations, key=lambda a: a.start_s)
    groups: List[OverlapGroup] = []
    current = [sorted_anns[0]]
    current_end = sorted_anns[0].end_s

    for ann in sorted_anns[1:]:
        if ann.start_s < current_end:
            current.append(ann)
            current_end = max(current_end, ann.end_s)
        else:
            if len(current) >= 2:
                groups.append(OverlapGroup(
                    annotations=list(current),
                    group_start=current[0].start_s,
                    group_end=current_end,
                ))
            current = [ann]
            current_end = ann.end_s

    if len(current) >= 2:
        groups.append(OverlapGroup(
            annotations=list(current),
            group_start=current[0].start_s,
            group_end=current_end,
        ))

    return groups


def _find_solo_interval(
    ann: CallAnnotation, group: OverlapGroup
) -> Optional[Tuple[float, float]]:
    """Find the largest non-overlapping sub-interval of `ann` within its group.

    The "solo" portion is where this caller is vocalizing but no other caller
    in the group is. We use this to estimate the caller's F0 cleanly.
    """
    others = [a for a in group.annotations if a is not ann]
    if not others:
        return (ann.start_s, ann.end_s)

    # Build a list of "busy" intervals from other callers
    busy = sorted([(a.start_s, a.end_s) for a in others])

    # Merge overlapping busy intervals
    merged_busy: List[Tuple[float, float]] = []
    for s, e in busy:
        if merged_busy and s < merged_busy[-1][1]:
            merged_busy[-1] = (merged_busy[-1][0], max(merged_busy[-1][1], e))
        else:
            merged_busy.append((s, e))

    # Find gaps within ann's interval where no other caller is active
    solo_intervals: List[Tuple[float, float]] = []
    cursor = ann.start_s
    for bs, be in merged_busy:
        if bs > cursor and bs <= ann.end_s:
            gap_start = max(cursor, ann.start_s)
            gap_end = min(bs, ann.end_s)
            if gap_end > gap_start:
                solo_intervals.append((gap_start, gap_end))
        cursor = max(cursor, be)
    # After all busy intervals
    if cursor < ann.end_s:
        solo_intervals.append((max(cursor, ann.start_s), ann.end_s))

    if not solo_intervals:
        return None

    # Return the longest solo interval
    return max(solo_intervals, key=lambda iv: iv[1] - iv[0])


def _estimate_f0(
    y: np.ndarray, sr: int, start_s: float, end_s: float,
    fmin: float = F0_MIN, fmax: float = F0_MAX,
) -> Optional[float]:
    """Estimate the median fundamental frequency in a time interval."""
    s0 = int(round(start_s * sr))
    s1 = int(round(end_s * sr))
    chunk = y[s0:s1]
    if len(chunk) < sr * 0.3:
        return None

    try:
        f0, voiced, prob = librosa.pyin(
            chunk, fmin=fmin, fmax=fmax, sr=sr, frame_length=2048,
        )
        f0_valid = f0[~np.isnan(f0)]
        if f0_valid.size >= 2:
            return float(np.median(f0_valid))
    except Exception:
        pass

    # Fallback: find the strongest spectral peak in the F0 range
    S = np.abs(librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mean_spec = S.mean(axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    band = (freqs >= fmin) & (freqs <= fmax)
    if band.any() and mean_spec[band].max() > 0:
        peak_idx = np.argmax(mean_spec[band])
        return float(freqs[band][peak_idx])

    return None


def _estimate_f0_contour(
    y: np.ndarray, sr: int, start_s: float, end_s: float,
    fmin: float = F0_MIN, fmax: float = F0_MAX,
) -> Optional[np.ndarray]:
    """Estimate a per-frame F0 contour in a time interval using pYIN.

    Returns an array of F0 values (one per STFT frame within the interval),
    with NaN gaps filled by linear interpolation and edge extrapolation.
    Returns None if not enough voiced frames are found.
    """
    s0 = int(round(start_s * sr))
    s1 = int(round(end_s * sr))
    chunk = y[s0:s1]
    if len(chunk) < sr * 0.3:
        return None

    try:
        f0_raw, voiced, prob = librosa.pyin(
            chunk, fmin=fmin, fmax=fmax, sr=sr,
            frame_length=2048, hop_length=HOP_LENGTH,
        )
    except Exception:
        return None

    if f0_raw is None or len(f0_raw) == 0:
        return None

    voiced_mask = ~np.isnan(f0_raw)
    if voiced_mask.sum() < 2:
        return None

    # Interpolate NaN gaps
    contour = f0_raw.copy()
    frame_indices = np.arange(len(contour))
    voiced_indices = frame_indices[voiced_mask]
    voiced_values = contour[voiced_mask]

    # np.interp handles extrapolation at the edges automatically (clamps to edge values)
    contour = np.interp(frame_indices, voiced_indices, voiced_values)

    return contour.astype(np.float64)


def _klapuri_iterative_f0(
    y: np.ndarray, sr: int, start_s: float, end_s: float,
    n_sources: int = 2,
    fmin: float = F0_MIN, fmax: float = F0_MAX,
    n_harmonics: int = KLAPURI_MAX_HARMONICS,
) -> List[float]:
    """Klapuri's iterative multipitch estimation for engulfed calls.

    When a call has zero solo portion (completely inside another call),
    the standard pYIN approach on the mixed signal is unreliable. Instead:
      1. Compute the harmonic salience function across candidate F0s.
      2. Find the strongest F0 peak.
      3. Subtract that F0's harmonic series from the magnitude spectrum.
      4. Find the next strongest F0 in the residual.
      5. Each detected F0 becomes one caller.

    The salience function for a candidate F0:
      salience(f0) = sum over harmonics h=1..H of:
          mag[round(h*f0/freq_resolution)] * weight[h]
      where weight[h] = 1/h (higher harmonics get less weight).

    Args:
        y: full waveform
        sr: sample rate
        start_s: start of the region to analyze
        end_s: end of the region to analyze
        n_sources: number of F0s to extract
        fmin: minimum candidate F0 in Hz
        fmax: maximum candidate F0 in Hz
        n_harmonics: number of harmonics to sum for salience

    Returns:
        List of detected F0 values (may be fewer than n_sources if
        residual energy is too low to reliably detect another source).
    """
    s0 = int(round(start_s * sr))
    s1 = int(round(end_s * sr))
    chunk = y[s0:s1]

    if len(chunk) < sr * 0.2:
        return []

    # Compute magnitude spectrum (average across frames)
    S = np.abs(librosa.stft(chunk, n_fft=N_FFT, hop_length=HOP_LENGTH))
    mag = S.mean(axis=1).astype(np.float64)  # average magnitude spectrum
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    freq_res = freqs[1] - freqs[0] if len(freqs) > 1 else FREQ_RESOLUTION

    # Build candidate F0 grid: 0.1 Hz resolution
    candidate_f0s = np.arange(fmin, fmax + 0.05, 0.1)

    # Precompute harmonic weights: 1/h
    harmonic_weights = np.array([1.0 / h for h in range(1, n_harmonics + 1)])

    residual_mag = mag.copy()
    detected_f0s: List[float] = []

    for _ in range(min(n_sources, KLAPURI_MAX_SOURCES)):
        # Compute salience for each candidate F0
        salience = np.zeros(len(candidate_f0s), dtype=np.float64)

        for ci, f0_candidate in enumerate(candidate_f0s):
            s = 0.0
            for h in range(1, n_harmonics + 1):
                harmonic_hz = h * f0_candidate
                if harmonic_hz > MAX_HARMONIC_HZ:
                    break
                bin_idx = int(round(harmonic_hz / freq_res))
                if 0 <= bin_idx < len(residual_mag):
                    s += residual_mag[bin_idx] * harmonic_weights[h - 1]
            salience[ci] = s

        # Find peak salience
        if salience.max() <= 0:
            break

        best_idx = int(np.argmax(salience))
        best_f0 = float(candidate_f0s[best_idx])

        # Check that the detected F0 is meaningfully strong
        # (at least 10% of the original peak salience to avoid noise)
        if detected_f0s:
            # Recompute salience on original mag for the first detected F0
            # to establish a baseline
            first_salience = 0.0
            for h in range(1, n_harmonics + 1):
                harmonic_hz = h * detected_f0s[0]
                if harmonic_hz > MAX_HARMONIC_HZ:
                    break
                bin_idx = int(round(harmonic_hz / freq_res))
                if 0 <= bin_idx < len(mag):
                    first_salience += mag[bin_idx] * harmonic_weights[h - 1]

            if first_salience > 0 and salience[best_idx] < 0.1 * first_salience:
                break

        detected_f0s.append(best_f0)

        # Subtract this F0's harmonic series from the residual magnitude
        bandwidth_bins = max(1, int(round(HARMONIC_BANDWIDTH_HZ / freq_res)))
        for h in range(1, n_harmonics + 1):
            harmonic_hz = h * best_f0
            if harmonic_hz > MAX_HARMONIC_HZ:
                break
            center_bin = int(round(harmonic_hz / freq_res))
            lo = max(0, center_bin - bandwidth_bins)
            hi = min(len(residual_mag), center_bin + bandwidth_bins + 1)
            # Zero out the harmonic region so the next iteration
            # doesn't pick up the same source
            residual_mag[lo:hi] = 0.0

    return detected_f0s


def _build_harmonic_comb(
    f0: float,
    freqs: np.ndarray,
    n_frames: int,
    max_hz: float = MAX_HARMONIC_HZ,
    bandwidth: float = HARMONIC_BANDWIDTH_HZ,
) -> np.ndarray:
    """Build a (F, T) mask with Gaussian peaks at each harmonic of f0.

    This is the STATIC version: the mask is the same across all time frames.
    Used as a fallback when no per-frame F0 contour is available.
    Each harmonic is modeled as exp(-0.5 * ((f - h*f0) / bandwidth)^2).
    """
    mask_1d = np.zeros(len(freqs), dtype=np.float64)
    n_harmonics = int(max_hz / f0)
    for h in range(1, n_harmonics + 1):
        harmonic_hz = h * f0
        dist = np.abs(freqs - harmonic_hz)
        gaussian = np.exp(-0.5 * (dist / bandwidth) ** 2)
        mask_1d = np.maximum(mask_1d, gaussian)
    mask_2d = np.tile(mask_1d[:, np.newaxis], (1, n_frames))
    return mask_2d.astype(np.float32)


def _build_timevarying_harmonic_comb(
    f0_contour: np.ndarray,
    freqs: np.ndarray,
    n_frames: int,
    max_hz: float = MAX_HARMONIC_HZ,
    bandwidth: float = HARMONIC_BANDWIDTH_HZ,
) -> np.ndarray:
    """Build a time-varying (F, T) harmonic comb mask from a per-frame F0 contour.

    Unlike _build_harmonic_comb which uses a single static F0, this version
    uses a different F0 for each time frame, tracking vibrato and onset
    transients accurately.

    Args:
        f0_contour: array of length n_frames with the F0 (Hz) for each frame.
                    Values should be positive; any NaN/zero entries get the
                    median F0 as fallback.
        freqs: frequency bin centers from librosa.fft_frequencies
        n_frames: number of STFT time frames
        max_hz: highest harmonic frequency to include
        bandwidth: Gaussian sigma for each harmonic peak (Hz)

    Returns:
        (F, T) mask array of dtype float32.
    """
    n_freq = len(freqs)
    mask = np.zeros((n_freq, n_frames), dtype=np.float64)

    # Sanitize the contour: replace NaN/zero/negative with median
    contour = f0_contour.copy()
    valid = contour[np.isfinite(contour) & (contour > 0)]
    if len(valid) == 0:
        # Complete fallback: use 20 Hz (middle of elephant range)
        contour[:] = 20.0
    else:
        fallback_f0 = float(np.median(valid))
        bad_mask = ~np.isfinite(contour) | (contour <= 0)
        contour[bad_mask] = fallback_f0

    # If contour is shorter or longer than n_frames, resample it
    if len(contour) != n_frames:
        old_indices = np.linspace(0, 1, len(contour))
        new_indices = np.linspace(0, 1, n_frames)
        contour = np.interp(new_indices, old_indices, contour)

    for t in range(n_frames):
        f0_t = contour[t]
        n_harmonics = int(max_hz / f0_t)
        frame_mask = np.zeros(n_freq, dtype=np.float64)
        for h in range(1, n_harmonics + 1):
            harmonic_hz = h * f0_t
            dist = np.abs(freqs - harmonic_hz)
            gaussian = np.exp(-0.5 * (dist / bandwidth) ** 2)
            frame_mask = np.maximum(frame_mask, gaussian)
        mask[:, t] = frame_mask

    return mask.astype(np.float32)


def _extrapolate_f0_contour(
    solo_contour: np.ndarray,
    solo_start_frame: int,
    solo_end_frame: int,
    target_start_frame: int,
    target_end_frame: int,
    n_total_frames: int,
) -> np.ndarray:
    """Extrapolate a solo-region F0 contour to cover the full STFT frame range.

    The solo contour gives us ground-truth F0 per frame where the caller is
    alone. We extend it into the overlap region using:
      - Linear extrapolation based on the trend of the last/first few frames
        of the solo contour (clamped to F0_MIN..F0_MAX to avoid runaway).
      - For frames far from the solo region, we clamp to the edge value.

    Returns an array of length n_total_frames.
    """
    full_contour = np.full(n_total_frames, np.nan, dtype=np.float64)

    # Place the solo contour in its correct frame range
    solo_len = len(solo_contour)
    actual_start = max(solo_start_frame, 0)
    actual_end = min(solo_start_frame + solo_len, n_total_frames)
    fill_start = actual_start - solo_start_frame
    fill_end = fill_start + (actual_end - actual_start)
    if actual_end > actual_start and fill_end > fill_start:
        full_contour[actual_start:actual_end] = solo_contour[fill_start:fill_end]

    # Determine the valid (non-NaN) range
    valid_mask = ~np.isnan(full_contour)
    if not valid_mask.any():
        full_contour[:] = 20.0  # middle of elephant range as ultimate fallback
        return full_contour

    valid_indices = np.where(valid_mask)[0]
    first_valid = valid_indices[0]
    last_valid = valid_indices[-1]
    valid_values = full_contour[valid_mask]

    # Compute a linear trend from the solo contour for extrapolation
    if len(valid_values) >= 3:
        # Use the last few frames for forward extrapolation
        trend_window = min(10, len(valid_values))

        # Forward trend (for extrapolating after solo)
        end_indices = valid_indices[-trend_window:]
        end_values = full_contour[end_indices]
        fwd_slope = 0.0
        if len(end_indices) >= 2:
            poly = np.polyfit(end_indices.astype(np.float64), end_values, 1)
            fwd_slope = poly[0]

        # Backward trend (for extrapolating before solo)
        start_indices = valid_indices[:trend_window]
        start_values = full_contour[start_indices]
        bwd_slope = 0.0
        if len(start_indices) >= 2:
            poly = np.polyfit(start_indices.astype(np.float64), start_values, 1)
            bwd_slope = poly[0]
    else:
        fwd_slope = 0.0
        bwd_slope = 0.0

    # Extrapolate backward (before the solo region)
    if first_valid > 0:
        base_val = full_contour[first_valid]
        for i in range(first_valid - 1, -1, -1):
            delta = (first_valid - i) * bwd_slope
            extrapolated = base_val - delta
            full_contour[i] = np.clip(extrapolated, F0_MIN, F0_MAX)

    # Extrapolate forward (after the solo region)
    if last_valid < n_total_frames - 1:
        base_val = full_contour[last_valid]
        for i in range(last_valid + 1, n_total_frames):
            delta = (i - last_valid) * fwd_slope
            extrapolated = base_val + delta
            full_contour[i] = np.clip(extrapolated, F0_MIN, F0_MAX)

    # Fill any internal NaN gaps by interpolation
    still_nan = np.isnan(full_contour)
    if still_nan.any():
        good = ~still_nan
        good_idx = np.where(good)[0]
        good_vals = full_contour[good]
        all_idx = np.arange(n_total_frames)
        full_contour = np.interp(all_idx, good_idx, good_vals)

    return full_contour


def separate_overlapping_callers(
    y: np.ndarray,
    sr: int,
    group: OverlapGroup,
) -> List[np.ndarray]:
    """Separate K overlapping callers using harmonic comb Wiener masking.

    Returns K waveforms, one per caller. Each waveform covers the full
    duration of `y` but only contains energy from its assigned caller
    within the overlap region. Outside the overlap, each caller gets
    its own annotation window from the original audio.

    Improvements over the static version:
      - Time-varying harmonic combs: F0 is estimated per-frame via pYIN
        in the solo portion and extrapolated into the overlap region.
      - Klapuri iterative F0 for engulfed calls: when a call has zero
        solo portion, multipitch estimation detects F0s from the mixture.
    """
    n_callers = len(group.annotations)
    duration = len(y) / sr

    # Compute STFT of the full waveform (needed for frame count and Klapuri)
    complex_stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True)
    mag = np.abs(complex_stft).astype(np.float32)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=N_FFT)
    n_freq, n_frames = complex_stft.shape
    fps = sr / HOP_LENGTH

    # --- Phase 1: Estimate F0 contours (or scalar F0) for each caller ---
    caller_f0_contours: List[Optional[np.ndarray]] = []
    caller_f0_scalars: List[Optional[float]] = []
    engulfed_indices: List[int] = []  # callers with no solo portion

    for k, ann in enumerate(group.annotations):
        solo = _find_solo_interval(ann, group)
        if solo and (solo[1] - solo[0]) >= 0.3:
            # Has a usable solo portion: estimate per-frame F0 contour
            contour = _estimate_f0_contour(y, sr, solo[0], solo[1])
            if contour is not None:
                # Extrapolate into the full waveform frame range
                solo_start_frame = int(round(solo[0] * fps))
                solo_end_frame = int(round(solo[1] * fps))
                full_contour = _extrapolate_f0_contour(
                    contour, solo_start_frame, solo_end_frame,
                    0, n_frames, n_frames,
                )
                caller_f0_contours.append(full_contour)
                caller_f0_scalars.append(float(np.median(contour)))
            else:
                # pYIN contour failed; try scalar estimate
                f0 = _estimate_f0(y, sr, solo[0], solo[1])
                caller_f0_contours.append(None)
                caller_f0_scalars.append(f0)
        else:
            # No solo portion (engulfed) or too short
            caller_f0_contours.append(None)
            caller_f0_scalars.append(None)
            engulfed_indices.append(k)

    # --- Phase 2: Klapuri iterative F0 for engulfed calls ---
    if engulfed_indices:
        # Determine how many F0s we need from Klapuri
        # Already-known F0s from non-engulfed callers
        known_f0s = [
            caller_f0_scalars[k] for k in range(n_callers)
            if k not in engulfed_indices and caller_f0_scalars[k] is not None
        ]

        # Run Klapuri on the overlap region
        n_needed = len(engulfed_indices)
        klapuri_f0s = _klapuri_iterative_f0(
            y, sr, group.group_start, group.group_end,
            n_sources=len(known_f0s) + n_needed,
        )

        # Remove any Klapuri F0s that are too close to already-known F0s
        # (within 3 Hz tolerance) so we don't assign the same F0 to two callers
        remaining_f0s: List[float] = []
        for kf0 in klapuri_f0s:
            is_duplicate = False
            for known in known_f0s:
                if abs(kf0 - known) < 3.0:
                    is_duplicate = True
                    break
            if not is_duplicate:
                remaining_f0s.append(kf0)

        # Assign remaining Klapuri F0s to engulfed callers
        for idx_i, k in enumerate(engulfed_indices):
            if idx_i < len(remaining_f0s):
                caller_f0_scalars[k] = remaining_f0s[idx_i]
            elif klapuri_f0s:
                # If we ran out of novel F0s, use the best available
                # (even if it's close to a known one — better than nothing)
                used_indices = set(range(min(idx_i, len(remaining_f0s))))
                for kf0 in klapuri_f0s:
                    already_used = False
                    for ui in used_indices:
                        if ui < len(remaining_f0s) and abs(kf0 - remaining_f0s[ui]) < 0.5:
                            already_used = True
                            break
                    if not already_used:
                        caller_f0_scalars[k] = kf0
                        break

    # --- Phase 3: Check if we have enough valid F0s ---
    valid_f0s = [f for f in caller_f0_scalars if f is not None]
    if len(valid_f0s) < 2:
        return [y.copy()]

    # --- Phase 4: Build per-caller harmonic comb masks (time-varying) ---
    combs = np.zeros((n_callers, n_freq, n_frames), dtype=np.float32)
    for k in range(n_callers):
        if caller_f0_contours[k] is not None:
            # Use time-varying comb from the per-frame F0 contour
            combs[k] = _build_timevarying_harmonic_comb(
                caller_f0_contours[k], freqs, n_frames,
            )
        elif caller_f0_scalars[k] is not None:
            # Use static comb from scalar F0 (Klapuri or fallback)
            combs[k] = _build_harmonic_comb(
                caller_f0_scalars[k], freqs, n_frames,
            )
        else:
            # No F0 — give this caller a flat mask (gets leftover energy)
            combs[k] = np.ones((n_freq, n_frames), dtype=np.float32) * 0.1

    # --- Phase 5: Wiener masking ---
    weighted = combs * mag[np.newaxis, :, :]  # (K, F, T)
    total = weighted.sum(axis=0, keepdims=True) + 1e-8  # (1, F, T)
    wiener_masks = weighted / total  # (K, F, T)

    # Determine which time frames fall in the overlap vs solo regions
    overlap_start_frame = int(round(group.group_start * fps))
    overlap_end_frame = int(round(group.group_end * fps))

    # Build per-caller output: Wiener mask in overlap, annotation window outside
    outputs: List[np.ndarray] = []
    for k, ann in enumerate(group.annotations):
        ann_start_frame = int(round(ann.start_s * fps))
        ann_end_frame = min(int(round(ann.end_s * fps)), n_frames)

        # Start with zeros
        caller_stft = np.zeros_like(complex_stft)

        # Solo region BEFORE overlap: use original STFT directly
        solo_before_start = ann_start_frame
        solo_before_end = min(overlap_start_frame, ann_end_frame)
        if solo_before_end > solo_before_start:
            caller_stft[:, solo_before_start:solo_before_end] = \
                complex_stft[:, solo_before_start:solo_before_end]

        # Overlap region: apply Wiener mask
        ov_start = max(ann_start_frame, overlap_start_frame)
        ov_end = min(ann_end_frame, overlap_end_frame)
        if ov_end > ov_start:
            caller_stft[:, ov_start:ov_end] = \
                wiener_masks[k, :, ov_start:ov_end] * complex_stft[:, ov_start:ov_end]

        # Solo region AFTER overlap: use original STFT directly
        solo_after_start = max(overlap_end_frame, ann_start_frame)
        solo_after_end = ann_end_frame
        if solo_after_end > solo_after_start:
            caller_stft[:, solo_after_start:solo_after_end] = \
                complex_stft[:, solo_after_start:solo_after_end]

        # Reconstruct waveform (ISTFT preserves original phase — no Griffin-Lim needed)
        caller_wav = librosa.istft(
            caller_stft, hop_length=HOP_LENGTH, n_fft=N_FFT, length=len(y),
        )
        outputs.append(caller_wav.astype(np.float32))

    return outputs


def process_file_with_annotations(
    y: np.ndarray,
    sr: int,
    file_annotations: Sequence[CallAnnotation],
    noise_mask: Optional[np.ndarray] = None,
) -> SeparationResult:
    """Full pipeline: noise removal + overlap detection + harmonic separation.

    Args:
        y: mono waveform at TARGET_SR
        sr: sample rate
        file_annotations: all annotations for this file from the CSV
        noise_mask: optional (F, T) mask from the U-Net for noise removal.
                    If provided, applied to the STFT before separation.

    Returns:
        SeparationResult with per-caller waveforms and metadata.
    """
    # Apply noise mask if provided (from the baseline U-Net)
    if noise_mask is not None:
        complex_stft = librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH, center=True)
        F_mask, T_mask = noise_mask.shape
        F_stft, T_stft = complex_stft.shape
        # Pad mask to full STFT shape
        full_mask = np.zeros((F_stft, T_stft), dtype=np.float32)
        full_mask[:min(F_mask, F_stft), :min(T_mask, T_stft)] = \
            noise_mask[:min(F_mask, F_stft), :min(T_mask, T_stft)]
        cleaned_stft = full_mask.astype(np.complex64) * complex_stft
        y = librosa.istft(cleaned_stft, hop_length=HOP_LENGTH, n_fft=N_FFT, length=len(y))
        y = y.astype(np.float32)

    # Find overlap groups
    overlap_groups = find_overlap_groups(file_annotations)

    if not overlap_groups:
        # No overlaps — single output (noise already removed above)
        return SeparationResult(
            caller_waveforms=[y],
            caller_f0s=[None],
            overlap_groups=[],
            sample_rate=sr,
        )

    # For each overlap group, separate callers via harmonic tracking
    all_waveforms: List[np.ndarray] = []
    all_f0s: List[Optional[float]] = []

    for group in overlap_groups:
        caller_wavs = separate_overlapping_callers(y, sr, group)
        for k, ann in enumerate(group.annotations):
            solo = _find_solo_interval(ann, group)
            f0 = None
            if solo and (solo[1] - solo[0]) >= 0.3:
                f0 = _estimate_f0(y, sr, solo[0], solo[1])
            all_f0s.append(f0)
        all_waveforms.extend(caller_wavs)

    # If no callers were successfully separated, return original
    if not all_waveforms:
        return SeparationResult(
            caller_waveforms=[y],
            caller_f0s=[None],
            overlap_groups=overlap_groups,
            sample_rate=sr,
        )

    return SeparationResult(
        caller_waveforms=all_waveforms,
        caller_f0s=all_f0s,
        overlap_groups=overlap_groups,
        sample_rate=sr,
    )


if __name__ == "__main__":
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from data_engine import load_raven_annotations

    project_root = Path(__file__).resolve().parent.parent
    audio_dir = project_root / "Audio Files (04-10-2026)"
    csv_path = list(project_root.glob("*.csv"))

    if not csv_path:
        print("[harmonic] No CSV found. Cannot test.")
        sys.exit(1)

    annotations = load_raven_annotations(csv_path[0])
    by_file: Dict[str, List[CallAnnotation]] = {}
    for ann in annotations:
        by_file.setdefault(ann.file_name, []).append(ann)

    # Find files with overlaps
    for fname, anns in by_file.items():
        groups = find_overlap_groups(anns)
        if not groups:
            continue

        wav_path = audio_dir / fname
        if not wav_path.exists():
            continue

        print(f"\n[harmonic] {fname}: {len(groups)} overlap group(s)")
        y, sr = librosa.load(wav_path, sr=TARGET_SR, mono=True)

        for gi, group in enumerate(groups):
            print(f"  group {gi}: {len(group.annotations)} callers, "
                  f"{group.group_start:.1f}-{group.group_end:.1f}s")
            for ann in group.annotations:
                solo = _find_solo_interval(ann, group)
                f0 = None
                if solo and (solo[1] - solo[0]) >= 0.3:
                    f0 = _estimate_f0(y, sr, solo[0], solo[1])
                solo_str = f"{solo[0]:.1f}-{solo[1]:.1f}s" if solo else "none"
                f0_str = f"{f0:.1f} Hz" if f0 else "unknown"
                print(f"    call {ann.start_s:.1f}-{ann.end_s:.1f}s | "
                      f"solo: {solo_str} | F0: {f0_str}")

        result = process_file_with_annotations(y, sr, anns)
        print(f"  -> {len(result.caller_waveforms)} output waveform(s)")
        break  # Just test the first file with overlaps
