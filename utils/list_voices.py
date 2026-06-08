"""
List available edge-tts voices filtered to English.
Run: python utils/list_voices.py
"""

import asyncio
import edge_tts


async def main() -> None:
    voices = await edge_tts.list_voices()
    en_voices = [v for v in voices if v["Locale"].startswith("en")]
    print(f"{'ShortName':<45} {'Gender':<8} Locale")
    print("-" * 75)
    for v in sorted(en_voices, key=lambda x: x["ShortName"]):
        print(f"{v['ShortName']:<45} {v['Gender']:<8} {v['Locale']}")
    print(f"\nTotal English voices: {len(en_voices)}")


if __name__ == "__main__":
    asyncio.run(main())
