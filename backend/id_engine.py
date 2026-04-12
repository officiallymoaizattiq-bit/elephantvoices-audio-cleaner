"""
Interspecies Identification & Labeling Factory.

Handles systematic storage of cleaned elephant audio: splicing per-caller
waveforms into individual murmur clips, assigning stable elephant IDs, and
persisting everything into a local SQLite database so the frontend can
render a searchable labeling UI.

Database schema (backend/elephant_registry.db):

    Elephants
        elephant_id   TEXT PRIMARY KEY   e.g. 'Elephant_A'
        nickname      TEXT               e.g. 'Kijana' (user-editable)

    Clips
        clip_id       INTEGER PRIMARY KEY AUTOINCREMENT
        file_path     TEXT NOT NULL       relative to project root
        parent_wav    TEXT NOT NULL       original noisy wav filename
        elephant_id   TEXT NOT NULL       FK -> Elephants.elephant_id
        start_time    REAL NOT NULL       seconds into the parent wav
        end_time      REAL NOT NULL
        clip_type     TEXT DEFAULT 'rumble'
        human_label   TEXT DEFAULT ''     free-text annotation
        created_at    TEXT DEFAULT (datetime('now'))
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "backend" / "elephant_registry.db"
CLIPS_DIR = PROJECT_ROOT / "backend" / "clips"

# Minimum clip length in seconds. Splices shorter than this are noise blips
# and should be dropped rather than cluttering the database.
MIN_CLIP_SECONDS = 0.3


def _get_connection() -> sqlite3.Connection:
    """Open (and lazily create) the SQLite database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create the two tables if they don't exist yet."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS Elephants (
            elephant_id   TEXT PRIMARY KEY,
            nickname      TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS Clips (
            clip_id       INTEGER PRIMARY KEY AUTOINCREMENT,
            file_path     TEXT NOT NULL,
            parent_wav    TEXT NOT NULL,
            elephant_id   TEXT NOT NULL,
            start_time    REAL NOT NULL,
            end_time      REAL NOT NULL,
            clip_type     TEXT NOT NULL DEFAULT 'rumble',
            human_label   TEXT NOT NULL DEFAULT '',
            created_at    TEXT NOT NULL DEFAULT (datetime('now')),
            FOREIGN KEY (elephant_id) REFERENCES Elephants(elephant_id)
        );
    """)
    conn.commit()


# ---------------------------------------------------------------------------
# Murmur detection: find contiguous high-energy segments within one caller
# ---------------------------------------------------------------------------

def _detect_murmurs(
    waveform: np.ndarray,
    sr: int,
    energy_window_s: float = 0.25,
    threshold_fraction: float = 0.15,
    min_duration_s: float = MIN_CLIP_SECONDS,
    merge_gap_s: float = 0.3,
) -> List[Tuple[float, float]]:
    """Detect individual murmur boundaries inside a single-caller waveform.

    A "murmur" is a contiguous region where the short-time RMS energy stays
    above `threshold_fraction * peak_rms`. Returns a list of (start_s, end_s)
    tuples.
    """
    if waveform.size == 0:
        return []

    # Short-time RMS energy in non-overlapping windows
    window_samples = max(1, int(energy_window_s * sr))
    n_windows = len(waveform) // window_samples
    if n_windows < 1:
        return [(0.0, float(len(waveform)) / sr)]

    trimmed = waveform[: n_windows * window_samples]
    frames = trimmed.reshape(n_windows, window_samples)
    rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1) + 1e-12)

    peak_rms = float(rms.max())
    if peak_rms < 1e-8:
        return []

    threshold = threshold_fraction * peak_rms
    active = rms > threshold

    # Connected components -> raw segments
    padded = np.concatenate([[False], active.astype(bool), [False]])
    diffs = padded[1:].astype(np.int8) - padded[:-1].astype(np.int8)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    raw: List[Tuple[float, float]] = []
    for s_idx, e_idx in zip(starts, ends):
        s_sec = float(s_idx * window_samples) / sr
        e_sec = float(e_idx * window_samples) / sr
        raw.append((s_sec, e_sec))

    # Merge segments whose gap is below merge_gap_s
    merged: List[Tuple[float, float]] = []
    for s, e in raw:
        if merged and (s - merged[-1][1]) <= merge_gap_s:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))

    # Drop too-short clips
    return [(s, e) for s, e in merged if (e - s) >= min_duration_s]


# ---------------------------------------------------------------------------
# Splicing: slice waveform into per-murmur wav files
# ---------------------------------------------------------------------------

def _splice_murmurs(
    waveform: np.ndarray,
    sr: int,
    elephant_id: str,
    parent_wav: str,
    clip_type: str = "rumble",
) -> List[Dict]:
    """Detect murmurs and write each one to its own wav file.

    Returns a list of dicts ready for database insertion, each containing:
        file_path, parent_wav, elephant_id, start_time, end_time, clip_type
    """
    CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    murmurs = _detect_murmurs(waveform, sr)

    if not murmurs:
        # No sub-murmur structure detected — treat the whole track as one clip
        murmurs = [(0.0, float(len(waveform)) / sr)]

    records: List[Dict] = []
    for start_s, end_s in murmurs:
        s0 = int(round(start_s * sr))
        s1 = int(round(end_s * sr))
        clip_samples = waveform[s0:s1]
        if clip_samples.size == 0:
            continue

        safe_id = elephant_id.replace(" ", "_")
        safe_parent = parent_wav.replace(".wav", "").replace(" ", "_")[:30]
        fname = f"{safe_parent}_{safe_id}_{clip_type}_T{start_s:.1f}.wav"
        out_path = CLIPS_DIR / fname
        sf.write(out_path, clip_samples, sr)

        rel_path = str(out_path.relative_to(PROJECT_ROOT))
        records.append({
            "file_path": rel_path,
            "parent_wav": parent_wav,
            "elephant_id": elephant_id,
            "start_time": start_s,
            "end_time": end_s,
            "clip_type": clip_type,
            "clip_id": None,
        })

    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_cleaning_result(
    parent_wav_name: str,
    caller_waveforms: List[np.ndarray],
    sr: int,
) -> List[Dict]:
    """After process_audio(), splice each caller into murmurs and register in DB.

    Args:
        parent_wav_name: original noisy wav filename (e.g. '04-040920-02_vehicle_1.wav')
        caller_waveforms: list of cleaned np arrays, one per detected caller
        sr: sample rate of the waveforms

    Returns:
        List of all clip records inserted (dicts with clip metadata).
    """
    conn = _get_connection()
    all_records: List[Dict] = []

    for idx, waveform in enumerate(caller_waveforms):
        # Assign a stable elephant ID for this caller in this file
        if len(caller_waveforms) == 1:
            elephant_id = "Elephant_A"
        else:
            letter = chr(ord("A") + idx)
            elephant_id = f"Elephant_{letter}"

        # Ensure the elephant exists in the Elephants table
        existing = conn.execute(
            "SELECT elephant_id FROM Elephants WHERE elephant_id = ?",
            (elephant_id,),
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO Elephants (elephant_id, nickname) VALUES (?, ?)",
                (elephant_id, ""),
            )

        # Splice into murmur clips and write wav files
        records = _splice_murmurs(
            waveform=waveform,
            sr=sr,
            elephant_id=elephant_id,
            parent_wav=parent_wav_name,
        )

        # Insert clip records and capture the auto-generated clip_id
        for rec in records:
            cursor = conn.execute(
                """INSERT INTO Clips
                   (file_path, parent_wav, elephant_id, start_time, end_time, clip_type, human_label)
                   VALUES (:file_path, :parent_wav, :elephant_id, :start_time, :end_time, :clip_type, '')""",
                rec,
            )
            rec["clip_id"] = cursor.lastrowid

        all_records.extend(records)

    conn.commit()
    conn.close()
    return all_records


def get_registry() -> Dict:
    """Fetch the full registry: all elephants and their clips."""
    conn = _get_connection()
    try:
        elephants = [
            dict(row)
            for row in conn.execute("SELECT * FROM Elephants ORDER BY elephant_id").fetchall()
        ]
        clips = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM Clips ORDER BY elephant_id, start_time"
            ).fetchall()
        ]
        return {"elephants": elephants, "clips": clips}
    finally:
        conn.close()


def rename_elephant(elephant_id: str, nickname: str) -> bool:
    """Update an elephant's nickname. Returns True if the row existed."""
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "UPDATE Elephants SET nickname = ? WHERE elephant_id = ?",
            (nickname, elephant_id),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def get_clips_for_parent(parent_wav: str) -> List[Dict]:
    """Fetch all clips from a specific parent wav file."""
    conn = _get_connection()
    try:
        return [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM Clips WHERE parent_wav = ? ORDER BY elephant_id, start_time",
                (parent_wav,),
            ).fetchall()
        ]
    finally:
        conn.close()


def get_clip_audio_path(clip_id: int) -> Optional[Path]:
    """Resolve a clip's wav file from its database ID."""
    conn = _get_connection()
    try:
        row = conn.execute("SELECT file_path FROM Clips WHERE clip_id = ?", (clip_id,)).fetchone()
        if row is None:
            return None
        full = PROJECT_ROOT / row["file_path"]
        return full if full.exists() else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # Fake a 10-second cleaned waveform with two energy bursts (murmurs)
    sr = 4000
    wav = np.zeros(sr * 10, dtype=np.float32)
    # Murmur 1 at 2-4s
    wav[sr * 2 : sr * 4] = rng.standard_normal(sr * 2).astype(np.float32) * 0.5
    # Murmur 2 at 6-8s
    wav[sr * 6 : sr * 8] = rng.standard_normal(sr * 2).astype(np.float32) * 0.3

    records = register_cleaning_result(
        parent_wav_name="test_smoke.wav",
        caller_waveforms=[wav],
        sr=sr,
    )
    print(f"[id_engine] registered {len(records)} clips:")
    for r in records:
        print(f"  {r['file_path']} | {r['start_time']:.1f}-{r['end_time']:.1f}s | {r['elephant_id']}")

    registry = get_registry()
    print(f"[id_engine] elephants: {len(registry['elephants'])}")
    print(f"[id_engine] total clips: {len(registry['clips'])}")

    ok = rename_elephant("Elephant_A", "Kijana")
    print(f"[id_engine] rename Elephant_A -> Kijana: {ok}")

    registry2 = get_registry()
    for e in registry2["elephants"]:
        print(f"  {e['elephant_id']}: nickname='{e['nickname']}'")

    print("[id_engine] OK")
