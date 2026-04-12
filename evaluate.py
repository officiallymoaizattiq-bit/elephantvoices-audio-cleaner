"""
Evaluate the harmonic detector against ground truth annotations.

Pipeline:
    1. Load ground truth CSV (auto-discovered at project root)
    2. Run detect_all_files() on the Audio Files directory
    3. Save predictions to predictions.csv
    4. Match predictions to ground truth via:
       a. IoU-based matching (Hungarian algorithm) at thresholds 0.3, 0.5, 0.7
       b. Collar-based detection matching (±1.0 s)
       c. Frame-level evaluation (0.256 s resolution)
    5. Compute comprehensive metrics and print report

Usage:
    python evaluate.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import librosa
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

# ---------------------------------------------------------------------------
# Setup paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
AUDIO_DIR = PROJECT_ROOT / "Audio Files (04-10-2026)"
GT_CSV = next(
    p for p in sorted(PROJECT_ROOT.glob("*.csv"))
    if p.name != "predictions.csv"
)

sys.path.insert(0, str(PROJECT_ROOT))
from detect import detect_all_files

# Constants
SR = 4000
HOP = 1024
FRAME_RES = HOP / SR  # 0.256 s

IOU_THRESHOLDS = [0.3, 0.5, 0.7]
COLLAR_S = 1.0


# ---------------------------------------------------------------------------
# Temporal IoU
# ---------------------------------------------------------------------------

def temporal_iou(
    pred_start: float, pred_end: float, gt_start: float, gt_end: float
) -> float:
    """Compute temporal Intersection over Union between two intervals."""
    inter_start = max(pred_start, gt_start)
    inter_end = min(pred_end, gt_end)
    intersection = max(0.0, inter_end - inter_start)
    union = (pred_end - pred_start) + (gt_end - gt_start) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


# ---------------------------------------------------------------------------
# IoU-based matching — Hungarian algorithm
# ---------------------------------------------------------------------------

def iou_match_file(
    gt_intervals: List[Tuple[float, float]],
    pred_intervals: List[Tuple[float, float]],
    iou_threshold: float,
) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    """Match GT to predictions within a single file using the Hungarian algorithm.

    Returns:
        matches: list of (gt_idx, pred_idx, iou) for matched pairs
        unmatched_gt: list of gt indices with no match
        unmatched_pred: list of pred indices with no match
    """
    n_gt = len(gt_intervals)
    n_pred = len(pred_intervals)

    if n_gt == 0 and n_pred == 0:
        return [], [], []
    if n_gt == 0:
        return [], [], list(range(n_pred))
    if n_pred == 0:
        return [], list(range(n_gt)), []

    # Build cost matrix: cost = -IoU (we minimise cost, so maximise IoU)
    cost_matrix = np.zeros((n_gt, n_pred), dtype=np.float64)
    for i, (gs, ge) in enumerate(gt_intervals):
        for j, (ps, pe) in enumerate(pred_intervals):
            cost_matrix[i, j] = -temporal_iou(ps, pe, gs, ge)

    # Solve assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches: List[Tuple[int, int, float]] = []
    matched_gt = set()
    matched_pred = set()

    for gi, pi in zip(row_indices, col_indices):
        iou_val = -cost_matrix[gi, pi]
        if iou_val >= iou_threshold:
            matches.append((gi, pi, iou_val))
            matched_gt.add(gi)
            matched_pred.add(pi)

    unmatched_gt = [i for i in range(n_gt) if i not in matched_gt]
    unmatched_pred = [j for j in range(n_pred) if j not in matched_pred]

    return matches, unmatched_gt, unmatched_pred


# ---------------------------------------------------------------------------
# Collar-based detection matching
# ---------------------------------------------------------------------------

def collar_match_file(
    gt_intervals: List[Tuple[float, float]],
    pred_intervals: List[Tuple[float, float]],
    collar: float,
) -> Tuple[int, int, int]:
    """Collar-based detection matching for a single file.

    A prediction 'detects' a GT call if:
      - the prediction's midpoint falls within [gt_start - collar, gt_end + collar], OR
      - the GT's midpoint falls within [pred_start - collar, pred_end + collar]

    Returns (tp, fp, fn).
    """
    n_gt = len(gt_intervals)
    n_pred = len(pred_intervals)

    if n_gt == 0:
        return 0, n_pred, 0
    if n_pred == 0:
        return 0, 0, n_gt

    gt_detected = [False] * n_gt
    pred_used = [False] * n_pred

    for j, (ps, pe) in enumerate(pred_intervals):
        pred_mid = (ps + pe) / 2.0
        for i, (gs, ge) in enumerate(gt_intervals):
            if gt_detected[i]:
                continue
            gt_mid = (gs + ge) / 2.0
            # Condition 1: pred midpoint within GT window + collar
            cond1 = (gs - collar) <= pred_mid <= (ge + collar)
            # Condition 2: GT midpoint within pred window + collar
            cond2 = (ps - collar) <= gt_mid <= (pe + collar)
            if cond1 or cond2:
                gt_detected[i] = True
                pred_used[j] = True
                break

    tp = sum(gt_detected)
    fn = n_gt - tp
    fp = sum(1 for u in pred_used if not u)
    return tp, fp, fn


# ---------------------------------------------------------------------------
# Frame-level evaluation
# ---------------------------------------------------------------------------

def rasterize_intervals(
    intervals: List[Tuple[float, float]], n_frames: int, frame_res: float
) -> np.ndarray:
    """Create a boolean frame-level vector from a list of intervals."""
    vec = np.zeros(n_frames, dtype=bool)
    for start, end in intervals:
        f_start = int(math.floor(start / frame_res))
        f_end = int(math.ceil(end / frame_res))
        f_start = max(0, f_start)
        f_end = min(n_frames, f_end)
        vec[f_start:f_end] = True
    return vec


def frame_metrics_file(
    gt_intervals: List[Tuple[float, float]],
    pred_intervals: List[Tuple[float, float]],
    file_duration: float,
) -> Tuple[int, int, int, int]:
    """Compute frame-level TP, FP, FN, TN for a single file.

    Returns (tp, fp, fn, tn).
    """
    n_frames = int(math.ceil(file_duration / FRAME_RES))
    if n_frames == 0:
        return 0, 0, 0, 0

    gt_vec = rasterize_intervals(gt_intervals, n_frames, FRAME_RES)
    pred_vec = rasterize_intervals(pred_intervals, n_frames, FRAME_RES)

    tp = int(np.sum(gt_vec & pred_vec))
    fp = int(np.sum(~gt_vec & pred_vec))
    fn = int(np.sum(gt_vec & ~pred_vec))
    tn = int(np.sum(~gt_vec & ~pred_vec))

    return tp, fp, fn, tn


# ---------------------------------------------------------------------------
# Diagnostic: merge rate, fragmentation rate
# ---------------------------------------------------------------------------

def overlap_count(
    source: List[Tuple[float, float]], target: List[Tuple[float, float]]
) -> List[int]:
    """For each interval in source, count how many intervals in target overlap it."""
    counts = []
    for ss, se in source:
        n = 0
        for ts, te in target:
            if ss < te and ts < se:
                n += 1
        counts.append(n)
    return counts


# ---------------------------------------------------------------------------
# Compute all metrics
# ---------------------------------------------------------------------------

def compute_all_metrics(
    pred_df: pd.DataFrame,
    gt_df: pd.DataFrame,
    audio_dir: Path,
) -> Dict:
    """Compute comprehensive evaluation metrics.

    Returns a dict containing event-level, collar-based, frame-level, and
    diagnostic metrics.
    """
    wav_files = sorted(audio_dir.glob("*.wav"))
    all_filenames = sorted(
        set(gt_df["Sound_file"].unique())
        | set(pred_df["Sound_file"].unique())
        | {f.name for f in wav_files}
    )

    # Filter GT: rumble-containing calls only for IoU metrics
    rumble_pattern = "rumble"
    gt_rumble = gt_df[
        gt_df["Call_type"].str.lower().str.contains(rumble_pattern, na=False)
    ].copy()

    # ---- accumulators ----
    # IoU-based (per threshold)
    iou_results = {
        t: {"tp": 0, "fp": 0, "fn": 0, "ious": [], "boundary_errors": []}
        for t in IOU_THRESHOLDS
    }

    # Collar-based (all GT, not filtered)
    collar_tp_total = 0
    collar_fp_total = 0
    collar_fn_total = 0

    # Frame-level
    frame_tp_total = 0
    frame_fp_total = 0
    frame_fn_total = 0
    frame_tn_total = 0

    # Diagnostics
    total_audio_duration_s = 0.0
    merge_preds = 0
    total_preds_for_merge = 0
    frag_gts = 0
    total_gts_for_frag = 0

    # Per-file breakdown (uses IoU 0.5)
    per_file_rows: List[Dict] = []

    for fname in all_filenames:
        # Gather GT and preds for this file
        gt_file = gt_rumble[gt_rumble["Sound_file"] == fname]
        gt_all_file = gt_df[gt_df["Sound_file"] == fname]
        pred_file = pred_df[pred_df["Sound_file"] == fname]

        gt_intervals = list(zip(gt_file["Start_time"], gt_file["End_time"]))
        gt_all_intervals = list(
            zip(gt_all_file["Start_time"], gt_all_file["End_time"])
        )
        pred_intervals = list(zip(pred_file["Start_time"], pred_file["End_time"]))

        # --- IoU matching (rumble-only GT) ---
        for t in IOU_THRESHOLDS:
            matches, unmatched_gt, unmatched_pred = iou_match_file(
                gt_intervals, pred_intervals, t
            )
            iou_results[t]["tp"] += len(matches)
            iou_results[t]["fp"] += len(unmatched_pred)
            iou_results[t]["fn"] += len(unmatched_gt)
            for gi, pi, iou_val in matches:
                iou_results[t]["ious"].append(iou_val)
                gs, ge = gt_intervals[gi]
                ps, pe = pred_intervals[pi]
                start_err = abs(ps - gs)
                end_err = abs(pe - ge)
                iou_results[t]["boundary_errors"].append((start_err, end_err))

        # Per-file row at IoU=0.5
        matches_05, ugt_05, upred_05 = iou_match_file(
            gt_intervals, pred_intervals, 0.5
        )
        per_file_rows.append(
            {
                "file": fname,
                "gt_calls": len(gt_intervals),
                "pred_calls": len(pred_intervals),
                "tp": len(matches_05),
                "fp": len(upred_05),
                "fn": len(ugt_05),
            }
        )

        # --- Collar-based matching (all GT call types) ---
        c_tp, c_fp, c_fn = collar_match_file(
            gt_all_intervals, pred_intervals, COLLAR_S
        )
        collar_tp_total += c_tp
        collar_fp_total += c_fp
        collar_fn_total += c_fn

        # --- Frame-level (rumble GT) ---
        wav_path = audio_dir / fname
        if wav_path.exists():
            file_dur = librosa.get_duration(path=str(wav_path))
            total_audio_duration_s += file_dur
            ftp, ffp, ffn, ftn = frame_metrics_file(
                gt_intervals, pred_intervals, file_dur
            )
            frame_tp_total += ftp
            frame_fp_total += ffp
            frame_fn_total += ffn
            frame_tn_total += ftn

        # --- Diagnostics ---
        if pred_intervals:
            counts = overlap_count(pred_intervals, gt_intervals)
            merge_preds += sum(1 for c in counts if c >= 2)
            total_preds_for_merge += len(pred_intervals)

        if gt_intervals:
            counts = overlap_count(gt_intervals, pred_intervals)
            frag_gts += sum(1 for c in counts if c >= 2)
            total_gts_for_frag += len(gt_intervals)

    # ---- aggregate metrics ----
    event_metrics = {}
    for t in IOU_THRESHOLDS:
        r = iou_results[t]
        tp, fp, fn = r["tp"], r["fp"], r["fn"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        avg_iou = float(np.mean(r["ious"])) if r["ious"] else 0.0

        if r["boundary_errors"]:
            be = np.array(r["boundary_errors"])
            mean_boundary_err = float(np.mean(be))
        else:
            mean_boundary_err = float("nan")

        event_metrics[t] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "avg_iou": avg_iou,
            "mean_boundary_error": mean_boundary_err,
        }

    # Collar-based
    collar_prec = (
        collar_tp_total / (collar_tp_total + collar_fp_total)
        if (collar_tp_total + collar_fp_total) > 0
        else 0.0
    )
    collar_rec = (
        collar_tp_total / (collar_tp_total + collar_fn_total)
        if (collar_tp_total + collar_fn_total) > 0
        else 0.0
    )
    collar_f1 = (
        2 * collar_prec * collar_rec / (collar_prec + collar_rec)
        if (collar_prec + collar_rec) > 0
        else 0.0
    )

    # Frame-level
    frame_prec = (
        frame_tp_total / (frame_tp_total + frame_fp_total)
        if (frame_tp_total + frame_fp_total) > 0
        else 0.0
    )
    frame_rec = (
        frame_tp_total / (frame_tp_total + frame_fn_total)
        if (frame_tp_total + frame_fn_total) > 0
        else 0.0
    )
    frame_f1 = (
        2 * frame_prec * frame_rec / (frame_prec + frame_rec)
        if (frame_prec + frame_rec) > 0
        else 0.0
    )

    # Diagnostics
    merge_rate = (
        merge_preds / total_preds_for_merge * 100
        if total_preds_for_merge > 0
        else 0.0
    )
    frag_rate = (
        frag_gts / total_gts_for_frag * 100
        if total_gts_for_frag > 0
        else 0.0
    )
    hours = total_audio_duration_s / 3600.0
    total_fp_all = event_metrics[0.3]["fp"]  # FPs at lowest threshold
    fa_per_hour = total_fp_all / hours if hours > 0 else 0.0

    # Combined boundary error from IoU=0.3 (most inclusive)
    mean_boundary_err = event_metrics[0.3]["mean_boundary_error"]

    return {
        "event_metrics": event_metrics,
        "collar": {
            "tp": collar_tp_total,
            "fp": collar_fp_total,
            "fn": collar_fn_total,
            "precision": collar_prec,
            "recall": collar_rec,
            "f1": collar_f1,
        },
        "frame": {
            "precision": frame_prec,
            "recall": frame_rec,
            "f1": frame_f1,
        },
        "diagnostics": {
            "mean_boundary_error": mean_boundary_err,
            "merge_rate": merge_rate,
            "fragmentation_rate": frag_rate,
            "fa_per_hour": fa_per_hour,
        },
        "per_file": per_file_rows,
        "total_audio_hours": hours,
    }


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def print_report(
    metrics: Dict,
    n_audio_files: int,
    n_gt_calls: int,
    n_pred_calls: int,
    pred_df: pd.DataFrame,
) -> None:
    """Print a comprehensive, well-formatted evaluation report."""

    w = 70  # report width

    print()
    print("=" * w)
    print("  ElephantVoices Harmonic Detector -- Evaluation Report")
    print("=" * w)
    print()
    print("  Detector: Two-stage harmonic salience + pYIN + F0 stability")
    print(
        f"  Audio files: {n_audio_files} | Ground truth calls: {n_gt_calls}"
        f" | Predicted calls: {n_pred_calls}"
    )

    # ── Event-Level Metrics ──
    print()
    print(f"  -- Event-Level Metrics {'-' * (w - 27)}")
    print()
    print(
        f"  {'IoU Threshold':<16} {'Precision':>10} {'Recall':>10}"
        f" {'F1':>10} {'Avg IoU':>10}"
    )
    print(
        f"  {'-' * 14:<16} {'-' * 9:>10} {'-' * 6:>10}"
        f" {'-' * 6:>10} {'-' * 7:>10}"
    )
    em = metrics["event_metrics"]
    for t in IOU_THRESHOLDS:
        m = em[t]
        print(
            f"  {t:<16.1f} {m['precision'] * 100:>9.1f}%"
            f" {m['recall'] * 100:>9.1f}%"
            f" {m['f1'] * 100:>9.1f}%"
            f" {m['avg_iou']:>10.3f}"
        )

    # ── Detection Metrics (collar-based) ──
    print()
    print(f"  -- Detection Metrics (+/-{COLLAR_S:.1f}s collar) {'-' * (w - 42)}")
    print()
    c = metrics["collar"]
    print(f"  Detection Precision:  {c['precision'] * 100:.1f}%")
    print(f"  Detection Recall:     {c['recall'] * 100:.1f}%")
    print(f"  Detection F1:         {c['f1'] * 100:.1f}%")

    # ── Frame-Level Metrics ──
    print()
    print(f"  -- Frame-Level Metrics {'-' * (w - 27)}")
    print()
    f = metrics["frame"]
    print(f"  Frame Precision:  {f['precision'] * 100:.1f}%")
    print(f"  Frame Recall:     {f['recall'] * 100:.1f}%")
    print(f"  Frame F1:         {f['f1'] * 100:.1f}%")

    # ── Diagnostic ──
    print()
    print(f"  -- Diagnostic {'-' * (w - 18)}")
    print()
    d = metrics["diagnostics"]
    be = d["mean_boundary_error"]
    be_str = f"{be:.2f} s" if not math.isnan(be) else "N/A"
    print(f"  Mean boundary error:     {be_str}")
    print(f"  Merge rate:              {d['merge_rate']:.1f}%")
    print(f"  Fragmentation rate:      {d['fragmentation_rate']:.1f}%")
    print(f"  False alarms per hour:   {d['fa_per_hour']:.1f}")

    # ── Per-File Breakdown ──
    print()
    print(f"  -- Per-File Breakdown {'-' * (w - 26)}")
    print()
    hdr = f"  {'File':<45} {'GT':>4} {'Pred':>5} {'TP':>4} {'FP':>4} {'FN':>4}"
    print(hdr)
    print(f"  {'-' * 45} {'-' * 4} {'-' * 5} {'-' * 4} {'-' * 4} {'-' * 4}")

    for r in metrics["per_file"]:
        if r["gt_calls"] > 0 or r["pred_calls"] > 0:
            name = r["file"]
            if len(name) > 45:
                name = name[:42] + "..."
            print(
                f"  {name:<45} {r['gt_calls']:>4} {r['pred_calls']:>5}"
                f" {r['tp']:>4} {r['fp']:>4} {r['fn']:>4}"
            )

    # ── Predictions (Selection Table) ──
    print()
    print(f"  -- Predictions (Selection Table) {'-' * (w - 37)}")
    print()
    print(
        f"  {'Sel':>4}  {'Sound_file':<45}  {'Start':>10}  {'End':>10}  {'Call_type'}"
    )
    print(
        f"  {'-' * 4}  {'-' * 45}  {'-' * 10}  {'-' * 10}  {'-' * 10}"
    )
    for _, row in pred_df.iterrows():
        fname = row["Sound_file"]
        if len(fname) > 45:
            fname = fname[:42] + "..."
        print(
            f"  {row['Selection']:>4}  {fname:<45}"
            f"  {row['Start_time']:>10.4f}  {row['End_time']:>10.4f}"
            f"  {row['Call_type']}"
        )

    print()
    print("=" * w)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # 1. Load ground truth
    gt_df = pd.read_csv(GT_CSV)
    n_gt = len(gt_df)
    print(f"[eval] Ground truth: {n_gt} annotations from {GT_CSV.name}")
    print(f"       Files with annotations: {gt_df['Sound_file'].nunique()}")

    # 2. Run detector on all audio files
    print("\n[eval] Running harmonic detector on all audio files...")
    pred_df = detect_all_files(AUDIO_DIR)
    n_pred = len(pred_df)
    print(f"[eval] Total predictions: {n_pred}")

    # 3. Save predictions
    pred_csv_path = PROJECT_ROOT / "predictions.csv"
    pred_df.to_csv(pred_csv_path, index=False)
    print(f"[eval] Predictions saved to {pred_csv_path.name}")

    # 4. Compute metrics
    n_audio_files = len(list(AUDIO_DIR.glob("*.wav")))
    print("[eval] Computing metrics...")
    metrics = compute_all_metrics(pred_df, gt_df, AUDIO_DIR)

    # 5. Print report
    print_report(metrics, n_audio_files, n_gt, n_pred, pred_df)
    print("[eval] Done.")


if __name__ == "__main__":
    main()
