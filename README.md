# ElephantVoices Audio Cleaner

AI-powered pipeline that isolates elephant rumble vocalizations (8–50 Hz F0) from noisy field recordings containing airplanes, vehicles, and generators, then provides LLM-powered bioacoustic interpretation.

Built for the **Infrastructure Masons Northern Texas Hackathon @ SMU Lyle School of Engineering**.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![React](https://img.shields.io/badge/React-18-61dafb) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688) ![License](https://img.shields.io/badge/License-MIT-green)

## How It Works

The system has two independent modes:

### Mode 1: Audio Cleaning Pipeline (Backend API)

Upload a noisy `.wav` → get cleaned per-caller audio with murmur clips.

```
Upload .wav → Resample to 4 kHz
  → U-Net soft mask (noise removal, 10–1000 Hz rumble band)
  → Harmonic caller separation (pYIN F0 + Wiener masking, if annotations exist)
  → Energy-based murmur splicing (0.25s windows, 15% RMS threshold)
  → SQLite registry (Elephant_A, Elephant_B, ...)
  → LLM acoustic analysis (Gemini 2.0 Flash via OpenRouter)
```

**Key details:**
- U-Net is a depth-4 model (~31M params) trained on spectrograms at `N_FFT=8192`, `HOP_LENGTH=1024`, `TARGET_SR=4000`
- Produces a soft sigmoid mask (no hard thresholding) — preserves gradient information
- Harmonic separator runs at finer `HOP_LENGTH=256` — the mask is upsampled 4x via bilinear interpolation
- Harmonic separation uses time-varying harmonic combs (Gaussian peaks at integer multiples of F0) with Wiener masking on the complex STFT (preserves phase — no Griffin-Lim needed)
- For engulfed calls (caller fully overlapped, no solo portion), uses Klapuri iterative multipitch estimation
- Harmonic separation **only activates if a CSV annotation file exists** in the project root. Without it, the pipeline applies noise removal only

### Mode 2: Standalone Detector (detect.py)

Runs on all audio files and outputs a selection table with start/end times — no neural network, pure classical DSP.

```
.wav → STFT → Three binary masks → Fused mask → Morphological smoothing → Call segments
```

**Three independent masks (all per-frame at 256-sample hop):**

| Mask | Method | What it catches | What it rejects |
|------|--------|----------------|-----------------|
| **A: pYIN Voicing** | `librosa.pyin` voiced probability > 0.2 | Periodic signals in 8–50 Hz | Broadband noise (airplanes) |
| **B: Harmonic Contrast** | Klapuri salience in elephant band (8–50 Hz) vs mechanical band (50–100 Hz) | Elephant harmonic series | Generator/engine harmonics at higher F0 |
| **C: F0 Stability** | Rolling std of F0 over 1s window, accept 0.2–6.0 Hz | Biological vibrato/drift | Generators (std ≈ 0) and random noise (std > 6) |

**Fusion rule:** `A AND (B OR C)` — voicing is mandatory, then either harmonic contrast or F0 stability must also pass.

**Post-processing:** morphological closing (1.0s) → opening (0.4s) → merge gaps ≤ 2.0s → reject segments < 1.0s or > 12.0s → confidence = mean voicing probability.

### Frontend

React + WaveSurfer.js timeline-based UI:
- Waveform hero with play/pause, zoom, scrub
- Timeline with per-caller colored segments synced to waveform (click segment → seek audio)
- Progressive disclosure: click a timeline segment to see details, audio player, and analysis
- LLM-powered vocalization analysis via "Analyze" button (calls `/api/analyze_clip/{id}`)
- Elephant nickname editing with SQLite persistence

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### Setup

```bash
git clone https://github.com/officiallymoaizattiq-bit/elephantvoices-audio-cleaner.git
cd elephantvoices-audio-cleaner

# Environment — add your OpenRouter API key
cp .env.example .env
# Edit .env: OPENROUTER_API_KEY=sk-or-v1-...

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

### Train the Model (required before using the API)

```bash
# Place .wav files in "Audio Files (04-10-2026)/"
# Place annotations CSV in project root
python model/train.py

# Or quick smoke test (1 epoch each stage):
STAGE1_EPOCHS=1 STAGE2_EPOCHS=1 python model/train.py
```

Saves weights to `backend/weights/unet_baseline.pth` (~118 MB, gitignored).

Stage 2 (instance segmentation with Deep Clustering) trains but is **not used in production inference** — only the mask head from Stage 1 is loaded.

### Run

```bash
# Terminal 1: Backend
python -m uvicorn backend.main:app --port 8000 --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173) and upload a `.wav` file.

### Run the Standalone Detector

```bash
# Detect calls in all audio files (no neural network, pure DSP)
python detect.py

# Evaluate against ground truth CSV
python evaluate.py
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness probe |
| `POST` | `/api/clean` | Upload .wav → cleaned audio + caller profiles + clip metadata |
| `GET` | `/api/registry` | All elephants + clips from SQLite |
| `PUT` | `/api/rename_elephant` | Update elephant nickname (`{elephant_id, nickname}`) |
| `GET` | `/api/clip/{id}` | Stream a murmur clip as .wav |
| `GET` | `/api/analyze_clip/{id}` | Acoustic feature extraction + LLM interpretation |
| `GET` | `/api/download/{filename}` | Serve cleaned output file or zip |

## Evaluation Metrics

`evaluate.py` computes comprehensive metrics against the ground truth annotation CSV:

- **Event-level**: IoU-based matching (Hungarian algorithm) at thresholds 0.3, 0.5, 0.7
- **Detection-level**: ±1.0s collar-based matching (did you find the call at all?)
- **Frame-level**: Per-frame precision/recall/F1 at 0.256s resolution
- **Diagnostics**: boundary error, merge rate, fragmentation rate, false alarms per hour

## Project Structure

```
├── backend/
│   ├── main.py              # FastAPI server (7 endpoints)
│   ├── inference.py          # U-Net noise removal + harmonic separation
│   ├── id_engine.py          # SQLite registry + energy-based murmur splicing
│   ├── llm_engine.py         # Acoustic features + Gemini 2.0 Flash via OpenRouter
│   ├── cpp_ext/              # Optional C++ mask acceleration (pybind11)
│   ├── weights/              # Trained .pth files (gitignored, ~236 MB)
│   ├── clips/                # Spliced murmur .wav files (gitignored)
│   └── outputs_persist/      # Cleaned outputs for download (gitignored)
├── model/
│   ├── train.py              # Two-stage training: mask U-Net → Deep Clustering
│   ├── unet.py               # Depth-4 U-Net (~31M params), optional embedding head
│   ├── harmonic_separator.py # pYIN + Klapuri F0 + Wiener masking (HOP=256)
│   ├── dataset.py            # ElephantSpectrogramDataset + InstanceMixDataset
│   ├── data_engine.py        # Raven-Pro CSV parsing + NoiseBank sampler
│   ├── losses.py             # DeepClusteringLoss + InstanceSegLoss
│   └── instance_inference.py # HDBSCAN embedding clustering (experimental, not used)
├── frontend/
│   ├── src/App.jsx           # React UI: WaveSurfer + timeline + analysis
│   ├── src/index.css         # Tailwind + animations
│   └── ...                   # Vite + Tailwind + PostCSS config
├── detect.py                 # Standalone DSP detector (voicing + harmonic contrast + F0 stability)
├── evaluate.py               # Multi-metric evaluation framework
├── .env.example              # Environment variable template
└── requirements.txt          # Python dependencies
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENROUTER_API_KEY` | For LLM analysis | API key from [openrouter.ai](https://openrouter.ai). Without it, analysis falls back to rule-based template. |

## Key Technical Details

| Parameter | Value | Why |
|-----------|-------|-----|
| Sample rate | 4000 Hz | Nyquist at 2 kHz covers all elephant harmonics up to ~1000 Hz |
| FFT size | 8192 | 0.488 Hz/bin resolution — resolves individual harmonics at 8 Hz F0 spacing |
| Training hop | 1024 | 0.256s frames, produces 2048×32 spectrograms per 8s chunk |
| Separator hop | 256 | 0.064s frames for finer temporal resolution in harmonic tracking |
| Elephant F0 | 8–50 Hz | Literature range for African elephant rumbles |
| Max harmonics | 1000 Hz | Harmonics extend up to ~50× F0 |

## Research Grounding

- Poole, J.H. (2011) — Behavioral contexts of elephant acoustic communication
- Soltis, J. (2010) — Vocal communication in African elephants
- Stoeger, A.S. & Baotic, A. (2016) — Information content of male African elephant social rumbles
- Hershey, J.R. et al. (2016) — Deep Clustering for segmentation and separation

## License

MIT
