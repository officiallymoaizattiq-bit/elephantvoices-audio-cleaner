# ElephantVoices Audio Cleaner

AI-powered pipeline that isolates elephant rumble vocalizations from noisy field recordings (airplanes, vehicles, generators) and provides bioacoustic analysis via LLM.

Built for the **Infrastructure Masons Northern Texas Hackathon @ SMU**.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![React](https://img.shields.io/badge/React-18-61dafb) ![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688) ![License](https://img.shields.io/badge/License-MIT-green)

## Architecture

```
Noisy .wav → U-Net Noise Removal → Harmonic Caller Separation → Murmur Splicing → LLM Analysis
```

### Pipeline Stages

1. **Noise Removal** — U-Net (31M params) trained on elephant rumble spectrograms masks the 10-1000 Hz band
2. **Caller Separation** — Classical DSP harmonic tracking (pYIN F0 estimation + Wiener masking) separates overlapping callers
3. **Murmur Splicing** — Energy-based boundary detection splits cleaned audio into individual call clips
4. **Acoustic Analysis** — Feature extraction (F0, harmonics, spectral centroid, RMS) + LLM interpretation via OpenRouter
5. **Detection & Evaluation** — 4-stage Acoustic Sieve (harmonic salience + pYIN voicing + F0 stability) for precision call detection

### Frontend

Timeline-based UI with WaveSurfer.js waveform visualization, synced call segments, progressive disclosure, and AI-powered vocalization analysis.

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+

### Setup

```bash
# Clone
git clone https://github.com/moaizattiq/elephantvoices-audio-cleaner.git
cd elephantvoices-audio-cleaner

# Environment
cp .env.example .env
# Edit .env and add your OpenRouter API key

# Python dependencies
pip install -r requirements.txt

# Frontend dependencies
cd frontend && npm install && cd ..
```

### Train the Model

```bash
# Place your .wav files in "Audio Files (04-10-2026)/"
# Place annotations CSV in project root

python model/train.py
# Saves weights to backend/weights/unet_baseline.pth
```

### Run

```bash
# Terminal 1: Backend
python -m uvicorn backend.main:app --port 8000 --reload

# Terminal 2: Frontend
cd frontend && npm run dev
```

Open [http://localhost:5173](http://localhost:5173) and upload a `.wav` file.

### Evaluate Detection Accuracy

```bash
python evaluate.py
```

## Project Structure

```
.
├── backend/
│   ├── main.py              # FastAPI server + endpoints
│   ├── inference.py          # U-Net + harmonic separator pipeline
│   ├── id_engine.py          # SQLite registry + murmur splicing
│   ├── llm_engine.py         # Acoustic features + OpenRouter LLM
│   └── cpp_ext/              # Optional C++ acceleration
├── model/
│   ├── train.py              # Two-stage training pipeline
│   ├── unet.py               # U-Net architecture (31M params)
│   ├── harmonic_separator.py # Classical DSP caller separation
│   ├── dataset.py            # Spectrogram datasets + augmentation
│   ├── data_engine.py        # Annotation parsing + noise bank
│   └── losses.py             # Deep Clustering loss
├── frontend/
│   ├── src/App.jsx           # React UI with WaveSurfer
│   └── ...                   # Vite + Tailwind config
├── detect.py                 # 4-stage Acoustic Sieve detector
├── evaluate.py               # Comprehensive evaluation metrics
├── .env.example              # Environment template
└── requirements.txt          # Python dependencies
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/health` | Liveness probe |
| `POST` | `/api/clean` | Upload .wav, get cleaned audio + caller profiles |
| `GET` | `/api/registry` | All elephants + clips |
| `PUT` | `/api/rename_elephant` | Update elephant nickname |
| `GET` | `/api/clip/{id}` | Stream a murmur clip |
| `GET` | `/api/analyze_clip/{id}` | Acoustic analysis + LLM interpretation |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENROUTER_API_KEY` | API key for LLM analysis ([openrouter.ai](https://openrouter.ai)) |

## Tech Stack

- **ML/DSP**: PyTorch, librosa, NumPy, SciPy, scikit-learn
- **Backend**: FastAPI, uvicorn, SQLite
- **Frontend**: React 18, Vite, Tailwind CSS, WaveSurfer.js, Lucide Icons
- **LLM**: OpenRouter (Google Gemini Flash)

## Research Grounding

- Poole, J.H. (2011) — Behavioral contexts of elephant acoustic communication
- Soltis, J. (2010) — Vocal communication in African elephants
- Stoeger, A.S. & Baotic, A. (2016) — Information content of male African elephant social rumbles

## License

MIT
