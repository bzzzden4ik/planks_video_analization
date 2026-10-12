"""
Общие функции: модель объектива (пиксели -> сантиметры вдоль конвейера)
и измерение сдвига доски между кадрами.
"""

import json

import cv2
import numpy as np

W, H = 2688, 1520
CY = H / 2
NORM = np.hypot(W, H) / 2
LINE_Y = 815                         # линия конвейера (середина доски по высоте)
MARKER_PX = ((1089, 515), (1471, 515))   # концы маркера на кадре
STRIP_X = (800, 1800)                # где меряем сдвиг доски (центр кадра)
STEP_CM = 0.5


class Lens:
    """Модель "рыбий глаз" + масштаб по маркеру. s(x) — см вдоль конвейера."""

    def __init__(self, k1, k2, cx, marker_length_cm):
        self.k1, self.k2, self.cx = float(k1), float(k2), float(cx)
        self.marker_length_cm = float(marker_length_cm)
        m = self.undistort(np.array(MARKER_PX, float))
        self.cm_per_unit = marker_length_cm / np.linalg.norm(m[1] - m[0])

    def undistort(self, pts):
        d = np.asarray(pts, float) - (self.cx, CY)
        r2 = (d ** 2).sum(-1, keepdims=True) / NORM ** 2
        return d / (1 + self.k1 * r2 + self.k2 * r2 ** 2)

    def s(self, x, y=LINE_Y):
        """Пиксель x (на линии конвейера) -> сантиметры от центра объектива."""
        x = np.asarray(x, float)
        pts = np.stack([x, np.full_like(x, y)], -1)
        return self.undistort(pts)[..., 0] * self.cm_per_unit

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"k1": self.k1, "k2": self.k2, "cx": self.cx,
                       "marker_length_cm": self.marker_length_cm}, f, indent=2)

    @classmethod
    def load(cls, path="calibration.json", marker_length_cm=None):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        if marker_length_cm is not None:
            d["marker_length_cm"] = marker_length_cm
        return cls(**d)


def board_mask(gray, refg, y1=740, y2=890):
    """Пиксели, где лежит доска (светлое и отличается от пустого конвейера), в строках y1..y2."""
    band = gray[y1:y2]
    return (cv2.absdiff(band, refg[y1:y2]) > 25) & (band > 110)


def board_center_y(mask, x0, x1, y1=740):
    """Середина доски по высоте на участке x0..x1 (или None)."""
    rows = mask[:, x0:x1].mean(axis=1)
    ys = np.nonzero(rows > 0.6)[0]
    if len(ys) < 40:
        return None
    return y1 + int((ys.min() + ys.max()) / 2)


def board_band_profile(gray, ref, lens, half=15, min_cover=0.85):
    """
    Яркость доски вдоль конвейера в центре кадра, пересчитанная в сантиметры
    (шаг STEP_CM), без плавных перепадов. Второе значение — лежит ли там доска.
    Полоса берётся по середине доски (доска может смещаться к бортам).
    """
    refg = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY) if ref.ndim == 3 else ref
    x0, x1 = STRIP_X
    yc = board_center_y(board_mask(gray, refg), x0, x1)
    if yc is None:
        return None, False
    y1, y2 = yc - half, yc + half
    band = gray[y1:y2]
    covered = ((cv2.absdiff(band[:, x0:x1], refg[y1:y2, x0:x1]) > 25) & (band[:, x0:x1] > 110)).mean() > min_cover
    xs = np.arange(x0, x1, 0.25)
    ss = lens.s(xs)
    grid = np.arange(ss[0], ss[-1], STEP_CM)
    gx = np.interp(grid, ss, xs).astype(np.float32)
    prof = cv2.remap(band.astype(np.float32), np.tile(gx, (y2 - y1, 1)),
                     np.tile(np.arange(y2 - y1, dtype=np.float32)[:, None], (1, len(gx))),
                     cv2.INTER_LINEAR).mean(axis=0)
    prof = prof - np.convolve(prof, np.ones(41) / 41, "same")
    return prof.astype(np.float32), covered


def shift_cm(a, b, max_shift_cm=25):
    """На сколько см сдвинулась доска от профиля a к профилю b (минус = влево)."""
    m = int(max_shift_cm / STEP_CM)
    n = min(len(a), len(b))
    tpl = a[m:n - m].reshape(1, -1)
    res = cv2.matchTemplate(b[:n].reshape(1, -1), tpl, cv2.TM_CCOEFF_NORMED)[0]
    i = int(np.argmax(res))
    score = float(res[i])
    if 0 < i < len(res) - 1:
        l, c, r = res[i - 1], res[i], res[i + 1]
        den = l - 2 * c + r
        if den != 0:
            i = i + 0.5 * (l - r) / den
    return (i - m) * STEP_CM, score
