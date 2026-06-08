import shutil
import time
from pathlib import Path

from ultralytics import YOLO


class Detector:
    def __init__(self, model_name: str, models_dir: str, confidence_threshold: float):
        self._conf = confidence_threshold
        self._filter: str | None = None

        self._model = YOLO(model_name)

        dest = Path(models_dir) / model_name
        try:
            src = Path(self._model.ckpt_path)
            if src.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(src, dest)
        except Exception as e:
            print(f"[Detector] Warning: could not copy weights to {dest}: {e}")

    # ------------------------------------------------------------------
    def get_results(self, frame) -> tuple[list[dict], float]:
        t0 = time.perf_counter()
        raw = self._model(frame, verbose=False, conf=self._conf)
        inference_ms = (time.perf_counter() - t0) * 1000

        results: list[dict] = []
        for box in raw[0].boxes:
            cls_id = int(box.cls[0])
            label = self._model.names[cls_id]
            if self._filter is not None and label != self._filter:
                continue
            x1, y1, x2, y2 = (int(v) for v in box.xyxy[0])
            results.append({
                "label": label,
                "confidence": float(box.conf[0]),
                "box": (x1, y1, x2, y2),
            })
        return results, inference_ms

    # ------------------------------------------------------------------
    def set_filter(self, class_name: str | None) -> None:
        self._filter = class_name

    def get_filter(self) -> str | None:
        return self._filter

    # ------------------------------------------------------------------
    def describe_scene(self, results: list[dict]) -> str:
        if not results:
            return "I detect nothing in the scene, sir."
        counts: dict[str, int] = {}
        for r in results:
            counts[r["label"]] = counts.get(r["label"], 0) + 1

        parts = []
        for label, n in counts.items():
            display = f"{n} {label}" + ("s" if n > 1 and not label.endswith("s") else "")
            parts.append(display)

        if len(parts) == 1:
            return f"I can see {parts[0]}, sir."
        return "I can see " + ", ".join(parts[:-1]) + ", and " + parts[-1] + ", sir."

    # ------------------------------------------------------------------
    @staticmethod
    def match_class(spoken: str, coco_classes: list[str]) -> str | None:
        spoken = spoken.lower().strip()

        # Multi-word alias map (common spoken forms → COCO canonical names)
        aliases = {
            "phone": "cell phone",
            "phones": "cell phone",
            "mobile": "cell phone",
            "mobiles": "cell phone",
            "cellphone": "cell phone",
            "cellphones": "cell phone",
            "tv": "tv",
            "television": "tv",
            "televisions": "tv",
            "couch": "couch",
            "sofa": "couch",
            "sofas": "couch",
            "motorbike": "motorcycle",
            "motorbikes": "motorcycle",
            "motorcycles": "motorcycle",
        }
        if spoken in aliases:
            return aliases[spoken]

        # Exact match
        if spoken in coco_classes:
            return spoken

        # Strip trailing plural suffix and retry
        for suffix in ("ies", "es", "s"):
            if spoken.endswith(suffix):
                singular = spoken[: -len(suffix)]
                if singular in coco_classes:
                    return singular
                # re-add 'y' for -ies → -y
                if suffix == "ies":
                    singular_y = spoken[:-3] + "y"
                    if singular_y in coco_classes:
                        return singular_y

        # Substring: spoken contains class name or class name contains spoken
        for cls in coco_classes:
            if spoken in cls or cls in spoken:
                return cls

        return None
