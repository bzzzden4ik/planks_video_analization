"""
Шаг 2. Поиск моментов, когда на конвейере что-то есть.

Берём первый кадр (конвейер пустой) как образец. Для каждого 5-го кадра
сравниваем полосу конвейера с образцом: считаем долю пикселей, которые
заметно изменились. Результат пишем в activity.csv и рисуем график activity.png.

Запуск:
    .venv\\Scripts\\python.exe scan_activity.py
"""

import argparse
import csv

import cv2
import numpy as np

# Полоса конвейера в координатах исходного кадра 2688x1520 (подобрана по кадрам).
ROI_Y1, ROI_Y2 = 700, 940
DIFF_THRESHOLD = 40      # насколько должна измениться яркость пикселя (0..255)
ACTIVE_LEVEL = 0.05      # доля изменившихся пикселей, выше которой считаем "что-то есть"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="sample.mp4")
    parser.add_argument("--step", type=int, default=5, help="анализировать каждый N-й кадр")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ОШИБКА: не удалось открыть {args.video}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0

    ok, first = cap.read()
    if not ok:
        print("ОШИБКА: не удалось прочитать первый кадр")
        return

    def prepare(frame):
        band = frame[ROI_Y1:ROI_Y2]
        gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
        return cv2.GaussianBlur(gray, (7, 7), 0)  # сглаживаем шум сжатия

    reference = prepare(first)
    rows = [(0, 0.0, 0.0)]
    frame_idx = 1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx % args.step == 0:
            diff = cv2.absdiff(prepare(frame), reference)
            changed = float(np.mean(diff > DIFF_THRESHOLD))
            rows.append((frame_idx, frame_idx / fps, changed))
        frame_idx += 1
    cap.release()

    with open("activity.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "seconds", "changed_fraction"])
        writer.writerows((fr, f"{s:.2f}", f"{c:.4f}") for fr, s, c in rows)

    # Интервалы активности: подряд идущие замеры выше ACTIVE_LEVEL.
    print("Интервалы, когда на конвейере что-то есть:")
    start = None
    for fr, sec, ch in rows + [(frame_idx, frame_idx / fps, 0.0)]:
        if ch > ACTIVE_LEVEL and start is None:
            start = sec
        elif ch <= ACTIVE_LEVEL and start is not None:
            if sec - start >= 2:  # короче 2 секунд - скорее помеха
                print(f"  {int(start // 60):02d}:{start % 60:04.1f} - {int(sec // 60):02d}:{sec % 60:04.1f}"
                      f"  ({sec - start:.0f} с)")
            start = None

    # График: по горизонтали время, по вертикали доля изменившихся пикселей.
    w, h = 1700, 300
    plot = np.full((h, w, 3), 255, np.uint8)
    total = frame_idx / fps
    peak = max(c for _, _, c in rows) or 1.0
    pts = np.array([(int(s / total * (w - 1)), int(h - 1 - c / peak * (h - 20))) for _, s, c in rows], np.int32)
    cv2.polylines(plot, [pts], False, (200, 80, 0), 1)
    y_level = int(h - 1 - ACTIVE_LEVEL / peak * (h - 20))
    cv2.line(plot, (0, y_level), (w, y_level), (0, 0, 255), 1)
    for minute in range(int(total // 60) + 1):
        x = int(minute * 60 / total * (w - 1))
        cv2.line(plot, (x, h - 10), (x, h), (0, 0, 0), 1)
        cv2.putText(plot, str(minute), (x + 2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    cv2.imwrite("activity.png", plot)
    print("\nСохранено: activity.csv, activity.png")


if __name__ == "__main__":
    main()
