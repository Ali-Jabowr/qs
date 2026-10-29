"""Снимок по Enter с подписью — для реального брака и пробных кадров (kit-moscow).

Открывает вебкамеру через тот же `WebcamSource`, что и `qs shoot` (настройки ручные и
проверяются на каждом кадре), и на каждый Enter сохраняет один PNG без потерь:
`NNN_<подпись>.png`, плюс строку в `snimki.csv`. Подпись — то, что человек написал про моти
(«brak01-vmyatina-vozvrat»). После снимка печатает белый лист и долю пересвета, чтобы на месте
было видно, не съехал ли свет. Пустая строка — выход.

Запуск — через kit-moscow/foto.sh.
"""

from __future__ import annotations

import argparse
import csv
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2

from qs.config import load_config
from qs.contracts import TriggerEvent
from qs.errors import CameraSettingsDrift, CaptureError, ConfigError, LightingFault
from qs.sources.webcam import WebcamSource

_CLIP = 250  # канал >= 250 из 255 — пересвет


def _safe(label: str) -> str:
    """Подпись → кусок имени файла: буквы (в т.ч. кириллица), цифры, '.', '-'; прочее → '_'."""
    return re.sub(r"[^\w.-]+", "_", label).strip("_")[:60] or "bez-podpisi"


def main() -> int:
    parser = argparse.ArgumentParser(description="снимок по Enter с подписью")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    try:
        config = load_config(Path(args.config)).config
    except ConfigError as exc:
        print(f"конфиг: {exc}")
        return 2
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seq = len(list(out.glob("*.png")))
    expect = config.lighting.reference_patch.expect

    source = WebcamSource(config)
    try:
        source.open()
    except (CaptureError, CameraSettingsDrift) as exc:
        print(f"камера: {exc}")
        source.close()
        return 2
    time.sleep(1.0)  # первые кадры после включения
    print(f"Снимки идут в {out}/")
    try:
        with (out / "snimki.csv").open("a", newline="", encoding="utf-8") as log:
            writer = csv.writer(log)
            while True:
                try:
                    label = input("\nПодпись (например brak01-vmyatina), пусто = выход: ").strip()
                except EOFError:
                    break
                if not label:
                    break
                seq += 1
                event = TriggerEvent(seq, time.perf_counter(), datetime.now(UTC), "kit-snap")
                try:
                    frame = source.capture(event)
                except (CaptureError, CameraSettingsDrift, LightingFault) as exc:
                    print(f"  НЕ СНЯТО: {exc} — попробуй ещё раз")
                    seq -= 1
                    continue
                name = f"{seq:03d}_{_safe(label)}.png"
                ok, encoded = cv2.imencode(".png", frame.pixels)
                if not ok:
                    print("  НЕ СОХРАНЕНО: ошибка кодирования PNG")
                    seq -= 1
                    continue
                with (out / name).open("xb") as png:
                    png.write(encoded.tobytes())
                clipped = float((frame.pixels.max(axis=2) >= _CLIP).mean() * 100)
                patch = frame.reference_patch
                writer.writerow([name, label, event.wall.isoformat(), f"{patch:.4f}",
                                 f"{clipped:.3f}"])
                log.flush()
                warn = ""
                if abs(patch - expect) > 0.10:
                    warn += "  ← белый лист сильно не как на стенде: лист в рамке? свет тот же?"
                if clipped > 1.0:
                    warn += "  ← ПЕРЕСВЕТ: уменьшить gain (05, шаг 4)"
                print(f"  сохранён {name} · белый лист {patch:.2f} (стенд {expect:.2f}) · "
                      f"пересвет {clipped:.2f}%{warn}")
    except KeyboardInterrupt:
        print("\nостановлено")
    finally:
        source.close()
    print(f"Готово: {seq} снимков в {out}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
