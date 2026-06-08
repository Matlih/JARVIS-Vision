import queue
import threading
from pathlib import Path

import speech_recognition as sr

import config


class VoiceHandler:
    def __init__(self, voice_dir: str, tts_rate: int):
        self._voice_dir = Path(voice_dir)
        self._tts_rate = tts_rate

        self._cmd_queue: queue.Queue[str] = queue.Queue()
        self._tts_queue: queue.Queue[dict] = queue.Queue()
        self._stop_event = threading.Event()

        self._sr_thread: threading.Thread | None = None
        self._tts_thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    def start(self) -> None:
        self._sr_thread = threading.Thread(target=self._sr_worker, daemon=True, name="SR")
        self._tts_thread = threading.Thread(target=self._tts_worker, daemon=True, name="TTS")
        self._sr_thread.start()
        self._tts_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._sr_thread:
            self._sr_thread.join(timeout=2.0)
        if self._tts_thread:
            self._tts_thread.join(timeout=2.0)

    # ------------------------------------------------------------------
    def get_command(self) -> str | None:
        try:
            return self._cmd_queue.get_nowait()
        except queue.Empty:
            return None

    def speak(self, text: str) -> None:
        self._tts_queue.put({"type": "speak", "text": text})

    def play_clip(self, clip_name: str) -> None:
        path = self._voice_dir / f"{clip_name}.mp3"
        self._tts_queue.put({"type": "clip", "filename": str(path)})

    # ------------------------------------------------------------------
    def _sr_worker(self) -> None:
        recognizer = sr.Recognizer()
        try:
            mic = sr.Microphone()
        except OSError as e:
            print(f"[SR] No microphone found — running in display-only mode. ({e})")
            return

        with mic as source:
            recognizer.adjust_for_ambient_noise(source, duration=1)

        recognizer.energy_threshold         = config.SR_ENERGY_THRESHOLD
        recognizer.dynamic_energy_threshold = config.SR_DYNAMIC_ENERGY

        while not self._stop_event.is_set():
            try:
                with mic as source:
                    audio = recognizer.listen(
                        source,
                        timeout=config.SR_TIMEOUT,
                        phrase_time_limit=config.SR_PHRASE_TIME_LIMIT,
                    )
                text = recognizer.recognize_google(audio).lower()
                if config.WAKE_WORD in text:
                    # Strip everything up to and including the wake word
                    idx = text.index(config.WAKE_WORD)
                    command = text[idx + len(config.WAKE_WORD):].strip().lstrip(",").strip()
                    if command:
                        self._cmd_queue.put(command)
            except sr.WaitTimeoutError:
                pass
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"[SR] Recognition service error: {e}")
            except Exception as e:
                print(f"[SR] Unexpected error: {e}")

    # ------------------------------------------------------------------
    def _tts_worker(self) -> None:
        import pygame

        pygame.mixer.init()

        while not self._stop_event.is_set():
            try:
                item = self._tts_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if item["type"] == "clip":
                path = Path(item["filename"])
                if path.exists():
                    try:
                        sound = pygame.mixer.Sound(str(path))
                        channel = sound.play()
                        while channel.get_busy():
                            pygame.time.wait(50)
                    except Exception as e:
                        print(f"[TTS] pygame playback error: {e}")
                        self._fallback_speak(path.stem, self._tts_rate)
                else:
                    self._fallback_speak(path.stem.replace("_", " "), self._tts_rate)

            elif item["type"] == "speak":
                self._fallback_speak(item["text"], self._tts_rate)

        pygame.mixer.quit()

    @staticmethod
    def _fallback_speak(text: str, rate: int) -> None:
        # Fresh engine each call — pyttsx3 on Windows breaks after first runAndWait()
        import pyttsx3
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[TTS] pyttsx3 speak error: {e}")
