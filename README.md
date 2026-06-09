# JARVIS Vision

Real-time object detection with a voice-controlled JARVIS interface. Built for **National University Dasmariñas (NUD) STEM Career Fair A.Y. 2025–2026** demo. Runs fully offline after first setup — no cloud at runtime.

---

## Features

- **YOLO11n** object detection on live webcam feed
- **Wake-word voice control** — say *"JARVIS, ..."* to issue commands
- **Holographic standby screen** — animated gold and blue wireframe eye on pure black
- **Detection flash effect** — white pulse when a new object enters the frame
- **Snapshot saving** — saves annotated frames to `snapshots/`
- **Class filtering** — lock detection to a single object type
- **Scene description** — JARVIS narrates everything it can see
- **Edge TTS voice** (en-GB-RyanNeural) for pre-generated clips; pyttsx3 for dynamic responses
- Graceful mic-absent mode (display-only, no crash)

---

## Requirements

- Windows 10 / 11
- Python 3.10+
- Webcam
- Microphone (optional — app runs without one)
- Internet connection for first run only (YOLO model auto-download + voice clip generation)

---

## Installation

```bash
git clone https://github.com/Matlih/JARVIS-Vision.git
cd JARVIS-Vision
pip install -r requirements.txt
```

### Generate voice clips (one-time)

Requires internet access. Creates 5 `.mp3` files in `voice/`.

```bash
python utils/generate_voice_clips.py
```

---

## Running

```bash
python main.py
```

The app opens in windowed mode showing the standby screen. Speak the wake word to begin.

---

## Voice Commands

All commands begin with the wake word **"JARVIS"**.

| Command | Effect |
|---|---|
| `JARVIS, activate` | Opens camera and starts detection (goes fullscreen) |
| `JARVIS, stop` | Releases camera and returns to standby |
| `JARVIS, what do you see` | JARVIS describes everything currently detected |
| `JARVIS, detect only [class]` | Filter to one object type, e.g. *"detect only person"* |
| `JARVIS, detect everything` | Remove filter, detect all objects |
| `JARVIS, how many [class] do you see` | Count a specific class in the current frame |
| `JARVIS, take snapshot` | Save annotated frame to `snapshots/` |

**Aliases:** "start" works for activate; "deactivate" works for stop; "detect all" works for detect everything.

---

## Keyboard Shortcuts

| Key | Action |
|---|---|
| `F` | Toggle fullscreen / windowed |
| `S` | Save snapshot (same as voice command) |
| `Q` / `ESC` | Quit |

---

## Project Structure

```
JARVIS-Vision/
├── main.py                  — App entry point, main loop, state machine
├── detector.py              — YOLO11 wrapper (inference, filtering, scene description)
├── voice_handler.py         — Speech recognition + TTS threads
├── config.py                — All constants (no magic numbers elsewhere)
├── requirements.txt
├── models/                  — YOLO weights (auto-downloaded, gitignored)
├── voice/                   — Pre-generated .mp3 clips (gitignored)
├── snapshots/               — Saved frames (gitignored)
└── utils/
    ├── generate_voice_clips.py   — One-time TTS clip generator
    └── list_voices.py            — Browse available edge-tts voices
```

---

## Configuration

All tunable values are in `config.py`. Key ones:

| Constant | Default | Description |
|---|---|---|
| `CAMERA_INDEX` | `0` | Webcam index |
| `CONFIDENCE_THRESHOLD` | `0.45` | Minimum detection confidence |
| `WAKE_WORD` | `"jarvis"` | Voice activation keyword |
| `SR_ENERGY_THRESHOLD` | `300` | Mic sensitivity — lower picks up quieter speech |
| `SR_PHRASE_TIME_LIMIT` | `6.0` | Max seconds to capture one command |
| `EDGE_TTS_VOICE` | `en-GB-RyanNeural` | Voice for generated clips |
| `FLASH_DURATION` | `0.5` | Seconds a new-detection flash lasts |

To browse available voices and change the TTS voice:

```bash
python utils/list_voices.py
```

Then update `EDGE_TTS_VOICE` in `config.py` and re-run `generate_voice_clips.py`.

---

## Notes

- YOLO11n weights are downloaded automatically on first run and cached in `models/`.
- The app runs without a microphone — detection and keyboard shortcuts still work.
- Snapshots are saved to `snapshots/jarvis_YYYYMMDD_HHMMSS.jpg` with detections drawn.
- `config.py` does not import OpenCV — safe to use from any utility script.
