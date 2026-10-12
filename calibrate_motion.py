"""
Калибровка "пиксели -> сантиметры" вдоль конвейера.

Модель объектива "рыбий глаз" (радиально-симметричная):
    u = (x - cx) / (1 + k1*r^2 + k2*r^4),  r — расстояние от центра кадра
и s = u * (см на единицу u), масштаб даёт маркер.

Параметры k1, k2, cx подбираются одновременно по двум видам данных:
  1. Ролики (центр кадра, x=490..2070): стоят с одинаковым шагом.
  2. Движущиеся доски (правый край, x=1850..2660): пока доску толкают,
     её правый торец "в системе доски" не меняется:
         s(x_торца(t)) - D(t) = const,
     где D(t) — сдвиг доски, измеренный по полосе в центре кадра.
Левый край кадра модель предсказывает за счёт симметрии объектива.

Результат: calibration.json. Кэш данных: cache/motion_data.npz
(удалите его, чтобы пересобрать данные из видео).

Запуск:
    .venv\\Scripts\\python.exe calibrate_motion.py
"""

import json
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares

from config import load_config
from lens import Lens, board_band_profile, shift_cm

VIDEO = "sample.mp4"
START_S, END_S = 75, 345          # участок с досками, пока фон под конвейером не меняли
CACHE = Path("cache/motion_data.npz")

# Ролики, найденные на пустом кадре (x) и их номер; пропущенные учтены.
ROLLERS_X = [492, 547, 604, 666, 732, 798, 866, 936, 1007, 1081, 1154, 1229, 1305, 1380,
             1451, 1597, 1668, 1737, 1803, 1864, 2002, 2069]
ROLLERS_N = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 22, 23]
RIGHT_ZONE = (1850, 2660)         # где используем правый торец для калибровки


def right_end(gray_band, ref_band):
    """Правый торец самой правой доски в полосе конвейера (или -1)."""
    cov = ((np.abs(gray_band.astype(np.int16) - ref_band) > 25) & (gray_band > 110)).mean(axis=0) > 0.35
    cov = np.convolve(cov.astype(float), np.ones(9) / 9, "same") > 0.5
    xs = np.nonzero(cov)[0]
    return int(xs.max()) if len(xs) else -1


def collect(lens0):
    """Проход по видео: правый торец и сдвиг доски D для каждого кадра."""
    cap = cv2.VideoCapture(VIDEO)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, ref = cap.read()
    ref_band = cv2.cvtColor(ref[760:870], cv2.COLOR_BGR2GRAY).astype(np.int16)
    rows, prev, k = [], None, 1
    while k <= int(END_S * fps):
        ok, f = cap.read()
        if not ok:
            break
        if k >= int(START_S * fps):
            gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            xr = right_end(gray[760:870], ref_band)
            prof, covered = board_band_profile(gray, ref, lens0)
            d, score = (np.nan, 0.0)
            if prev is not None and covered and prev[1]:
                d, score = shift_cm(prev[0], prof)
            rows.append((k, xr, d, score))
            prev = (prof, covered)
        k += 1
    cap.release()
    return np.array(rows, float)


def split_boards(data):
    """Делим кадры на доски: новая доска — когда правый торец прыгает вправо."""
    k, xr = data[:, 0], data[:, 1]
    board = np.zeros(len(k), int)
    b, last = 0, xr[0]
    for i in range(1, len(k)):
        if xr[i] > 0 and last > 0 and xr[i] - last > 400:
            b += 1
        if xr[i] > 0:
            last = xr[i]
        board[i] = b
    return board


def main():
    cfg = load_config()
    lens0 = Lens(-0.2106, -0.1811, 1344.0, cfg.marker_length_cm)   # стартовая модель
    if CACHE.exists():
        data = np.load(CACHE)["data"]
    else:
        print("Собираю данные из видео (несколько минут)...")
        data = collect(lens0)
        CACHE.parent.mkdir(exist_ok=True)
        np.savez(CACHE, data=data)
    board = split_boards(data)

    # Накопленный сдвиг D внутри каждой доски; там, где сдвиг не измерен, цепочка рвётся.
    groups = []         # (номер группы, x торца, D)
    gid = 0
    for b in np.unique(board):
        idx = np.nonzero(board == b)[0]
        D = 0.0
        for i in idx:
            d, score, xr = data[i, 2], data[i, 3], data[i, 1]
            if np.isnan(d) or score < 0.6:
                gid += 1
                D = 0.0
                continue
            D += d
            if RIGHT_ZONE[0] <= xr <= RIGHT_ZONE[1]:
                groups.append((gid, xr, D))
    groups = np.array(groups)
    # Оставляем группы, где торец прошёл хотя бы 150 пикселей.
    keep = []
    for g in np.unique(groups[:, 0]):
        m = groups[:, 0] == g
        if np.ptp(groups[m, 1]) > 150:
            keep.append(g)
    groups = groups[np.isin(groups[:, 0], keep)]
    gids = {g: i for i, g in enumerate(np.unique(groups[:, 0]))}
    print(f"Доски: {board.max() + 1}, участков движения у правого края: {len(gids)}, точек: {len(groups)}")

    rx, rn = np.array(ROLLERS_X, float), np.array(ROLLERS_N, float)

    def residuals(p):
        k1, k2, cx, pitch, off = p[:5]
        consts = p[5:]
        lens = Lens(k1, k2, cx, cfg.marker_length_cm)
        r1 = (lens.s(rx) - (off + rn * pitch)) / 2.0        # ролики, ошибка ~2 см
        gi = np.array([gids[g] for g in groups[:, 0]])
        r2 = (lens.s(groups[:, 1]) - groups[:, 2] - consts[gi]) / 3.0   # торец, ошибка ~3 см
        return np.concatenate([r1, r2])

    p0 = [-0.21, -0.18, 1344.0, 19.0, 0.0] + [0.0] * len(gids)
    # Стартовые константы: среднее положение торца в системе доски.
    for g, i in gids.items():
        m = groups[:, 0] == g
        p0[5 + i] = float(np.mean(lens0.s(groups[m, 1]) - groups[m, 2]))
    # Разумные границы: объектив "бочка" (k1<0), центр недалеко от середины кадра.
    lo = [-1.0, -1.5, 1200.0, 10.0, -1e4] + [-1e4] * len(gids)
    hi = [0.2, 1.0, 1500.0, 30.0, 1e4] + [1e4] * len(gids)
    res = least_squares(residuals, p0, bounds=(lo, hi), loss="soft_l1", f_scale=1.0)
    k1, k2, cx, pitch = res.x[:4]
    lens = Lens(k1, k2, cx, cfg.marker_length_cm)

    r = residuals(res.x)
    n_rollers = len(rx)
    print(f"Параметры: k1={k1:.4f} k2={k2:.4f} cx={cx:.1f}, шаг роликов {pitch:.2f} см")
    print(f"Ошибка по роликам: СКО {np.std(r[:n_rollers]) * 2:.1f} см")
    print(f"Ошибка по торцам у правого края: СКО {np.std(r[n_rollers:]) * 3:.1f} см, "
          f"медиана |ошибки| {np.median(np.abs(r[n_rollers:])) * 3:.1f} см")

    # Сравнение со старыми вариантами на типичной доске.
    print("\nПроверка: положение (см) точки x относительно x=73:")
    for x in (500, 1344, 2200, 2500, 2620):
        print(f"  x={x}: {float(lens.s(x) - lens.s(73)):.1f}")

    lens.save("calibration.json")
    print("\nСохранено: calibration.json")


if __name__ == "__main__":
    main()
