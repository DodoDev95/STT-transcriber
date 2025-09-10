# Real-Time Voice Transcription

This project is split into three main parts:

- **client/** – Next.js frontend UI
- **gateway/** – Go-based WebSocket gateway between browser and STT
- **STT/** – FastAPI + Faster-Whisper speech-to-text server

---

## IMPORTANT
The project is set to use the "cuda" model in the transcriber, which relies on GPU processing so you would have to install the NVIDIA CUDA toolkit and CUDNN.
To use the CPU powered version of the model set the variables in /STT/app.py to:
`
model = WhisperModel("base", device="cpu", compute_type="int8")
`
This model is weaker and makes the transcriber less accurate when listening, but the audio parameters can still be toyed with to improve the "base" model

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/DodoDev95/STT-transcriber
cd STT-transcriber

```

### 2. Install STT (Python + FastAPI + Faster-Whisper)

```bash
cd STT
python -m venv .venv
source .venv/bin/activate # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
In the app.py file you can change the language of the model from "en" to whichever language you need

### Run STT server

```bash
uvicorn app:app --host 0.0.0.0 --port 9000 --reload
```

### 3. Gateway (Go)

```bash
cd gateway
go run main.go
```

### 4. Client Next.js

```bash
cd client
npm install
npm run dev
```




