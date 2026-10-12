"""
Измерение длины первой доски (она принимается за N).

Проблема: в sample.mp4 оба торца первой доски не видны в одном кадре.
Левый торец виден около 79.2-79.6 с (потом его закрывает рабочий),
правый — с 80.8 с (раньше он за краем кадра или ещё не опущен). Доска движется.

Решение:
  длина = s(правый торец, t_R) - s(левый торец, t_L) + D(t_L -> t_R)
  где s — положение вдоль доски в сантиметрах, D — на сколько сантиметров
  доска уехала влево между кадрами t_L и t_R.

D измеряем сравнением центральной части доски (x=500..1900, с сучками)
между соседними кадрами. Полоска заранее пересчитана в сантиметры,
поэтому сдвиг сразу получается в сантиметрах.

Положения торцов проверены глазами (автоматика ошибалась из-за рабочего
и тёмной рамы) и записаны в LEFT_END / RIGHT_ENDS.

Перевод в сантиметры: модель "рыбьего глаза" по шагу роликов (3 варианта,
одинаково хорошо описывающих центр кадра) + маркер. Проверка модели у края:
правый торец и центр доски должны сдвигаться на одно и то же расстояние.

Запуск:
    .venv\\Scripts\\python.exe measure_first_board.py
"""

import cv2
import numpy as np

from config import load_config

VIDEO = "sample.mp4"
FIRST, LAST = 1975, 2050
W, H = 2688, 1520
MARKER_PX = ((1089, 515), (1471, 515))       # концы маркера на кадре

# Торцы, проверенные глазами по увеличенным кадрам (x в пикселях).
LEFT_END = {1980: 73, 1985: 73, 1990: 72}
RIGHT_ENDS = {2020: 2617, 2025: 2592, 2035: 2544, 2050: 2512}

LENS_VARIANTS = {
    "k1": (-0.2739, 0.0, W / 2),
    "k1+k2": (-0.2106, -0.1811, W / 2),
    "k1+k2+центр": (-0.223, -0.2153, 1320.67),
}
STEP_CM = 0.5                                 # шаг полоски при пересчёте в см


def undistort(pts, k1, k2, cx, cy=H / 2):
    pts = np.asarray(pts, float)
    d = pts - (cx, cy)
    r2 = (d ** 2).sum(-1, keepdims=True) / (np.hypot(W, H) / 2) ** 2
    return d / (1 + k1 * r2 + k2 * r2 ** 2)


def board_axis(frame, ref):
    """Прямая y = a*x + b по середине доски на участке x=300..1800."""
    diff = cv2.absdiff(frame, ref).max(axis=2)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = (diff > 30) & (gray > 140)
    pts = []
    for x in range(300, 1800, 25):
        ys = np.nonzero(mask[650:1000, x])[0]
        if len(ys) > 40:
            pts.append((x, 650 + (ys.min() + ys.max()) / 2))
    pts = np.array(pts)
    return np.polyfit(pts[:, 0], pts[:, 1], 1)


class Ruler:
    """Положение вдоль оси доски в сантиметрах для одного варианта модели."""

    def __init__(self, lens, cm_per_unit):
        self.lens, self.cpu = lens, cm_per_unit

    def s(self, x, axis):
        """x (пиксели на кадре) -> сантиметры вдоль доски."""
        x = np.atleast_1d(np.asarray(x, float))
        p = undistort(np.stack([x, np.polyval(axis, x)], -1), *self.lens)
        a = undistort([[0, np.polyval(axis, 0)], [W, np.polyval(axis, W)]], *self.lens)
        direction = (a[1] - a[0]) / np.linalg.norm(a[1] - a[0])
        return p @ direction * self.cpu

    def strip(self, frame, axis, s_from, s_to):
        """Яркость вдоль доски с шагом STEP_CM (усреднение поперёк доски ±10 пикс)."""
        xs = np.arange(300, 2100, 0.25)
        ss = self.s(xs, axis)
        grid = np.arange(s_from, s_to, STEP_CM)
        gx = np.interp(grid, ss, xs).astype(np.float32)
        gy = np.polyval(axis, gx).astype(np.float32)
        offs = np.arange(-10, 11, dtype=np.float32)[:, None]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        band = cv2.remap(gray, np.tile(gx, (21, 1)), gy[None, :] + offs, cv2.INTER_LINEAR)
        prof = band.astype(np.float32).mean(axis=0)
        prof -= np.convolve(prof, np.ones(41) / 41, "same")     # убираем плавные перепады
        return prof


def shift_cm(a, b, max_shift_cm=30):
    """На сколько см профиль b сдвинут относительно a (b(s) = a(s + shift))."""
    m = int(max_shift_cm / STEP_CM)
    core = a[m:-m]
    scores = [np.corrcoef(core, b[m - d:len(b) - m - d])[0, 1] for d in range(-m, m + 1)]
    i = int(np.argmax(scores))
    if 0 < i < len(scores) - 1:
        l, c, r = scores[i - 1], scores[i], scores[i + 1]
        i = i + 0.5 * (l - r) / (l - 2 * c + r)
    return (i - m) * STEP_CM, max(scores)


def main():
    cfg = load_config()
    cap = cv2.VideoCapture(VIDEO)
    ok, ref = cap.read()
    frames, i = {}, 1
    while i <= LAST:
        ok, f = cap.read()
        if not ok:
            break
        if i >= FIRST:
            frames[i] = f
        i += 1
    cap.release()
    axes = {k: board_axis(f, ref) for k, f in frames.items()}

    results = {}
    for name, lens in LENS_VARIANTS.items():
        m = undistort(MARKER_PX, *lens)
        ruler = Ruler(lens, cfg.marker_length_cm / np.linalg.norm(m[1] - m[0]))

        # Смещение доски D(k): сумма сдвигов между соседними кадрами.
        lo = float(ruler.s(500, axes[FIRST])[0])
        hi = float(ruler.s(1900, axes[FIRST])[0])
        k0 = min(LEFT_END)
        D, worst = {k0: 0.0}, 1.0
        prev = ruler.strip(frames[k0], axes[k0], lo, hi)
        for k in range(k0 + 1, LAST + 1):
            cur = ruler.strip(frames[k], axes[k], lo, hi)
            d, score = shift_cm(prev, cur)
            D[k] = D[k - 1] + d
            worst = min(worst, score)
            prev = cur

        left = {k: float(ruler.s(x, axes[k])[0]) + D[k] for k, x in LEFT_END.items()}
        right = {k: float(ruler.s(x, axes[k])[0]) + D[k] for k, x in RIGHT_ENDS.items()}
        # left/right — положения торцов "в системе доски" (как будто доска не двигалась).
        lengths = [r - l for r in right.values() for l in left.values()]
        results[name] = lengths

        print(f"=== вариант модели: {name} ===")
        print(f"  доска сдвинулась с {k0 / 25:.1f} по {LAST / 25:.1f} с на {D[LAST]:.1f} см "
              f"(худшая похожесть соседних кадров {worst:.2f})")
        print("  правый торец в системе доски (должен быть постоянным), см:",
              [round(float(v), 1) for v in right.values()])
        print("  левый торец в системе доски, см:", [round(float(v), 1) for v in left.values()])
        print(f"  длина: медиана {np.median(lengths):.0f} см (от {min(lengths):.0f} до {max(lengths):.0f})\n")

    all_l = np.concatenate(list(results.values()))
    print(f"ИТОГ: длина первой доски ≈ {np.median(all_l):.0f} см, "
          f"разброс по вариантам модели {min(np.median(v) for v in results.values()):.0f}"
          f"–{max(np.median(v) for v in results.values()):.0f} см")


if __name__ == "__main__":
    main()
