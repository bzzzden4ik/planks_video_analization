"""
Шаг 1. Анализ видео.

Скрипт открывает видео, печатает его параметры (разрешение, FPS, длительность),
читает файл кадр за кадром до конца, чтобы проверить, что он не битый,
и сохраняет кадры через равные промежутки времени в папку frames/.

Запуск:
    .venv\\Scripts\\python.exe analyze_video.py
    .venv\\Scripts\\python.exe analyze_video.py --every 10   (кадр каждые 10 секунд)
"""

import argparse
import time
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description="Анализ видео конвейера")
    parser.add_argument("--video", default="sample.mp4", help="путь к видеофайлу")
    parser.add_argument("--every", type=float, default=30.0,
                        help="сохранять кадр каждые N секунд видео")
    parser.add_argument("--out", default="frames", help="папка для сохранённых кадров")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ОШИБКА: файл {video_path} не найден")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ОШИБКА: OpenCV не смог открыть {video_path}")
        return

    # Параметры, которые сообщает сам файл. Для потоков без оглавления (MPEG-PS)
    # число кадров здесь бывает неточным, поэтому ниже считаем кадры сами.
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    reported_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Файл:             {video_path} ({video_path.stat().st_size / 1024**2:.0f} МБ)")
    print(f"Разрешение:       {width} x {height}")
    print(f"FPS (из файла):   {fps:.2f}")
    print(f"Кадров (из файла): {reported_frames}")
    if fps <= 0:
        print("ВНИМАНИЕ: FPS не определён, считаю 25 кадров/с")
        fps = 25.0

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    step = max(1, round(args.every * fps))  # через сколько кадров сохранять
    frame_idx = 0
    saved = 0
    last_pos_ms = 0.0
    started = time.time()

    print("\nЧитаю видео до конца...")
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        last_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

        if frame_idx % step == 0:
            seconds = frame_idx / fps
            name = out_dir / f"frame_{frame_idx:06d}_{int(seconds // 60):02d}m{int(seconds % 60):02d}s.jpg"
            cv2.imwrite(str(name), frame)
            saved += 1

        frame_idx += 1
        if frame_idx % 5000 == 0:
            print(f"  прочитано {frame_idx} кадров...")

    cap.release()

    duration = frame_idx / fps
    print(f"\nПрочитано кадров:  {frame_idx}")
    print(f"Длительность:      {int(duration // 60)} мин {duration % 60:.1f} с (по числу кадров и FPS)")
    print(f"Время последнего кадра по меткам файла: {last_pos_ms / 1000:.1f} с")
    print(f"Сохранено кадров:  {saved} в папку {out_dir}/")
    print(f"Время работы:      {time.time() - started:.0f} с")


if __name__ == "__main__":
    main()
