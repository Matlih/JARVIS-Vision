import math
import re
import time
from enum import Enum, auto
from pathlib import Path

import cv2
import numpy as np
import pygame

import config
from detector import Detector
from voice_handler import VoiceHandler


class AppState(Enum):
    INACTIVE = auto()
    ACTIVE = auto()


# ---------------------------------------------------------------------------
def draw_holographic_eye(frame: np.ndarray, cx: int, cy: int, t: float) -> np.ndarray:
    EW  = 375   # eye half-width
    EH  = 147   # eye half-height
    IR  = 118   # iris radius
    PR  = 38    # pupil radius

    GOLD     = (0,   195, 255)   # BGR: school gold
    BLUE     = (200,  70,  20)   # BGR: school royal blue
    BLUE_DIM = ( 55,  18,   5)   # dim blue for subtle sphere lines

    eh = EH
    ys = 1.0

    # --- Almond eye outline (gold, pointed corners) ---
    N = 200
    pts = []
    for i in range(N + 1):
        norm = i / N
        x = int(cx - EW + 2 * EW * norm)
        y = int(cy - eh * math.sin(math.pi * norm) ** 1.0)
        pts.append([x, y])
    for i in range(N, -1, -1):
        norm = i / N
        x = int(cx - EW + 2 * EW * norm)
        y = int(cy + eh * math.sin(math.pi * norm) ** 1.0)
        pts.append([x, y])
    cv2.polylines(frame, [np.array(pts, np.int32).reshape((-1, 1, 2))],
                  True, GOLD, 2, cv2.LINE_AA)

    # --- Iris outer ring (gold) ---
    cv2.ellipse(frame, (cx, cy), (IR, max(1, int(IR * ys))),
                0, 0, 360, GOLD, 1, cv2.LINE_AA)

    # --- Latitude lines: blue sphere depth effect ---
    n_lat = 8
    for i in range(-n_lat + 1, n_lat):
        phi   = (math.pi / 2) * i / n_lat
        r_lat = int(IR * math.cos(phi))
        y_lat = int(cy + IR * math.sin(phi) * ys)
        y_r   = max(1, int(r_lat * 0.28 * ys))
        if r_lat > 3:
            shade = BLUE if i == 0 else BLUE_DIM
            cv2.ellipse(frame, (cx, y_lat), (r_lat, y_r),
                        0, 0, 360, shade, 1, cv2.LINE_AA)

    # --- Longitude curves: slowly rotating blue wireframe ---
    rot = t * 0.25
    for i in range(6):
        lam   = rot + math.pi * i / 6
        pts_l = []
        for j in range(60):
            theta = math.pi * j / 59 - math.pi / 2
            x = int(cx + IR * math.cos(theta) * math.sin(lam))
            y = int(cy + IR * math.sin(theta) * ys)
            pts_l.append([[x, y]])
        cv2.polylines(frame, [np.array(pts_l, np.int32)],
                      False, BLUE_DIM, 1, cv2.LINE_AA)

    # --- Pupil: filled black + gold ring ---
    pr_h = max(1, int(PR * ys))
    cv2.ellipse(frame, (cx, cy), (PR, pr_h), 0, 0, 360, (5, 5, 5), -1)
    cv2.ellipse(frame, (cx, cy), (PR, pr_h), 0, 0, 360, GOLD, 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
def draw_standby(base: np.ndarray, t: float) -> np.ndarray:
    h, w = base.shape[:2]
    frame = np.zeros((h, w, 3), dtype=np.uint8)  # pure black

    draw_holographic_eye(frame, w // 2, h // 2, t)

    # "J.A.R.V.I.S." title above the eye
    GOLD = (0, 195, 255)
    title = "J.A.R.V.I.S."
    scale = 1.6
    thick = 2
    (tw, th), _ = cv2.getTextSize(title, config.FONT, scale, thick)
    tx = (w - tw) // 2
    ty = h // 2 - 132 - 30          # 30 px above the eye top
    cv2.putText(frame, title, (tx + 1, ty + 1), config.FONT, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
    cv2.putText(frame, title, (tx, ty), config.FONT, scale, GOLD, thick, cv2.LINE_AA)

    # Blinking "AWAITING ACTIVATION" at bottom centre
    if int(t * 2) % 2 == 0:
        label = "[ AWAITING ACTIVATION ]"
        (tw, _), _ = cv2.getTextSize(label, config.FONT, 0.65, 1)
        lx = (w - tw) // 2
        cv2.putText(frame, label, (lx + 1, h - 29), config.FONT, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, label, (lx, h - 30), config.FONT, 0.65, GOLD, 1, cv2.LINE_AA)

    # Status dot (top-right)
    dot_col = GOLD if int(t * 3) % 2 == 0 else (0, 60, 80)
    cv2.circle(frame, (w - 28, 28), 5, dot_col, -1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
def draw_detections(frame: np.ndarray, results: list[dict], active_filter: str | None,
                    flash_map: dict | None = None) -> np.ndarray:
    now = time.perf_counter()
    for det in results:
        label = det["label"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["box"]

        color = tuple(int(v) for v in np.array([
            (hash(label) * 2654435761) & 0xFF,
            (hash(label) * 1234567891) & 0xFF,
            (hash(label) * 987654321)  & 0xFF,
        ], dtype=np.uint8))

        # Flash pulse for newly appeared objects
        if flash_map:
            remaining = flash_map.get(label, 0) - now
            if remaining > 0:
                progress = remaining / config.FLASH_DURATION
                pad = int(8 * progress)
                thick = max(1, int(4 * progress))
                cv2.rectangle(frame, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad),
                              (255, 255, 255), thick, cv2.LINE_AA)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

        text = f"{label} {int(conf * 100)}%"
        (tw, th), baseline = cv2.getTextSize(text, config.FONT, config.FONT_SCALE, config.FONT_THICKNESS)
        cv2.rectangle(frame, (x1, y1 - th - baseline - 4), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, text, (x1 + 2, y1 - baseline - 2),
                    config.FONT, config.FONT_SCALE, (0, 0, 0), config.FONT_THICKNESS, cv2.LINE_AA)
    return frame


# ---------------------------------------------------------------------------
def draw_hud(frame: np.ndarray, active_filter: str | None, fps: float, inference_ms: float) -> np.ndarray:
    mode_text = f"MODE: {('FILTER:' + active_filter.upper()) if active_filter else 'ALL OBJECTS'}"
    fps_text  = f"FPS:  {fps:.1f}"
    inf_text  = f"INF:  {inference_ms:.1f} ms"

    y = 24
    for line in (mode_text, fps_text, inf_text):
        # 1px black shadow
        cv2.putText(frame, line, (11, y + 1), config.FONT, config.FONT_SCALE,
                    (0, 0, 0), config.FONT_THICKNESS, cv2.LINE_AA)
        cv2.putText(frame, line, (10, y), config.FONT, config.FONT_SCALE,
                    config.HUD_TEXT_COLOR, config.FONT_THICKNESS, cv2.LINE_AA)
        y += 26
    return frame


# ---------------------------------------------------------------------------
def handle_command(
    cmd: str,
    state: AppState,
    cap: cv2.VideoCapture | None,
    detector: Detector,
    voice: VoiceHandler,
    last_results: list[dict],
    last_frame: np.ndarray | None = None,
) -> tuple[AppState, cv2.VideoCapture | None]:

    print(f"[CMD] {cmd!r}")

    # --- activate ---
    if ("start" in cmd or "activ" in cmd) and state == AppState.INACTIVE:
        new_cap = cv2.VideoCapture(config.CAMERA_INDEX)
        if not new_cap.isOpened():
            voice.play_clip("camera_error")
            return state, cap
        cv2.setWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        voice.play_clip("activated")
        return AppState.ACTIVE, new_cap

    # --- deactivate ---
    if ("stop" in cmd or "deact" in cmd) and state == AppState.ACTIVE:
        if cap:
            cap.release()
        cv2.setWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        voice.play_clip("deactivated")
        return AppState.INACTIVE, None

    # --- filter: detect only <class> ---
    if "detect only" in cmd:
        after = cmd.split("detect only", 1)[1].strip()
        matched = Detector.match_class(after, list(detector._model.names.values()))
        if matched:
            detector.set_filter(matched)
            voice.speak(f"Now detecting only {matched}, sir.")
        else:
            voice.speak(f"I don't recognise that object class, sir.")
        return state, cap

    # --- remove filter ---
    if "detect everything" in cmd or "detect all" in cmd:
        detector.set_filter(None)
        voice.speak("Detecting all objects, sir.")
        return state, cap

    # --- how many <class> ---
    if "how many" in cmd:
        m = re.search(r"how many (.+?)(?:\s+(?:do|can) you see)?$", cmd)
        spoken_class = m.group(1).strip() if m else ""
        if not spoken_class:
            return state, cap
        matched = Detector.match_class(spoken_class, list(detector._model.names.values()))
        if matched:
            count = sum(1 for r in last_results if r["label"] == matched)
            noun = matched + ("s" if count != 1 and not matched.endswith("s") else "")
            voice.speak(f"I detect {count} {noun}, sir.")
        else:
            voice.speak("I'm not sure what class you mean, sir.")
        return state, cap

    # --- what do you see ---
    if "what" in cmd or "what do you" in cmd or "what can you" in cmd or "what do you see" in cmd or "what can you see" in cmd:
        voice.speak(detector.describe_scene(last_results))
        return state, cap

    # --- snapshot ---
    if "snapshot" in cmd and state == AppState.ACTIVE:
        if last_frame is not None:
            Path(config.SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            path = Path(config.SNAPSHOTS_DIR) / f"jarvis_{ts}.jpg"
            cv2.imwrite(str(path), last_frame)
            voice.speak("Snapshot saved, sir.")
        return state, cap

    return state, cap


# ---------------------------------------------------------------------------
def main() -> None:
    pygame.init()

    detector = Detector(config.MODEL_NAME, config.MODELS_DIR, config.CONFIDENCE_THRESHOLD)
    voice = VoiceHandler(config.VOICE_DIR, config.TTS_RATE)
    voice.start()
    voice.play_clip("startup")

    state = AppState.INACTIVE
    cap: cv2.VideoCapture | None = None
    last_results: list[dict] = []
    inference_ms = 0.0
    prev_labels: set[str] = set()
    flash_map: dict[str, float] = {}
    last_annotated_frame: np.ndarray | None = None

    standby = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.namedWindow(config.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(config.WINDOW_TITLE, 1280, 720)

    fps = 0.0
    frame_count = 0
    t_fps = time.perf_counter()

    try:
        while True:
            # Window close detection
            if cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                break

            # Voice commands
            cmd = voice.get_command()
            if cmd:
                state, cap = handle_command(cmd, state, cap, detector, voice, last_results, last_annotated_frame)

            if state == AppState.ACTIVE and cap is not None:
                ret, frame = cap.read()
                if not ret:
                    voice.play_clip("camera_error")
                    cap.release()
                    cap = None
                    state = AppState.INACTIVE
                    continue

                last_results, inference_ms = detector.get_results(frame)

                # Flash map: register newly appeared labels
                now = time.perf_counter()
                current_labels = {r["label"] for r in last_results}
                for label in current_labels - prev_labels:
                    flash_map[label] = now + config.FLASH_DURATION
                prev_labels = current_labels

                frame = draw_detections(frame, last_results, detector.get_filter(), flash_map)
                last_annotated_frame = frame.copy()

                # FPS calculation
                frame_count += 1
                if frame_count % config.LOOP_FPS_DISPLAY_UPDATE == 0:
                    elapsed = time.perf_counter() - t_fps
                    fps = config.LOOP_FPS_DISPLAY_UPDATE / elapsed if elapsed > 0 else 0.0
                    t_fps = time.perf_counter()

                frame = draw_hud(frame, detector.get_filter(), fps, inference_ms)
                cv2.imshow(config.WINDOW_TITLE, frame)
                key = cv2.waitKey(1) & 0xFF
            else:
                cv2.imshow(config.WINDOW_TITLE, draw_standby(standby, time.perf_counter()))
                key = cv2.waitKey(30) & 0xFF

            if key in (ord("q"), ord("Q"), 27):  # Q or ESC
                break
            if key in (ord("f"), ord("F")):
                fs = cv2.getWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty(config.WINDOW_TITLE, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL if fs == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)
            if key in (ord("s"), ord("S")) and state == AppState.ACTIVE and last_annotated_frame is not None:
                Path(config.SNAPSHOTS_DIR).mkdir(parents=True, exist_ok=True)
                ts = time.strftime("%Y%m%d_%H%M%S")
                path = Path(config.SNAPSHOTS_DIR) / f"jarvis_{ts}.jpg"
                cv2.imwrite(str(path), last_annotated_frame)
                voice.speak("Snapshot saved, sir.")

    finally:
        if cap:
            cap.release()
        cv2.destroyAllWindows()
        voice.stop()
        pygame.quit()


if __name__ == "__main__":
    main()
