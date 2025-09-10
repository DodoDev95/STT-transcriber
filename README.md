# Real-Time Voice Transcription (EN only)

This project is split into three main parts:

- **client/** – Next.js frontend UI
- **gateway/** – Go-based WebSocket gateway between browser and STT
- **STT/** – FastAPI + Faster-Whisper speech-to-text server

---

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


