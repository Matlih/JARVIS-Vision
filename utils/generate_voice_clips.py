"""
Generate static JARVIS voice clips using edge-tts.
Run: python utils/generate_voice_clips.py
"""

import asyncio
import sys
from pathlib import Path

# Allow running from project root or from utils/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import edge_tts
import config

CLIPS = {
    "startup":      "System initialized. Awaiting commands, sir.",
    "activated":    "Vision system activated, sir.",
    "deactivated":  "Vision system deactivated, sir.",
    "target_lost":  "No objects detected.",
    "camera_error": "Camera not found. Please check your configuration.",
}


async def generate_clip(name: str, text: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.mp3"
    communicate = edge_tts.Communicate(
        text,
        voice=config.EDGE_TTS_VOICE,
        rate=config.EDGE_TTS_RATE,
        pitch=config.EDGE_TTS_PITCH,
    )
    await communicate.save(str(path))
    print(f"  [OK] {path}")


async def main() -> None:
    out_dir = Path(config.VOICE_DIR)
    print(f"Generating {len(CLIPS)} voice clips into '{out_dir}/' ...")
    await asyncio.gather(*(generate_clip(n, t, out_dir) for n, t in CLIPS.items()))
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
