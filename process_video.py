"""
Подсчёт и измерение досок на видео конвейера.

Как работает:
  1. Доска = длинная полоса в зоне конвейера, которая отличается от пустого
     конвейера И закрывает ролики (так отсеиваются люди, пол, доски рядом).
  2. Новая доска = правый торец резко "прыгнул" вправо (новую доску кладут
     справа, пока прошлая уходит в станок слева).
  3. Длину меряем ТОЛЬКО пока доска лежит целиком (до начала проталкивания):
         длина = s(правый торец) - s(левый торец)
     s — сантиметры по калибровке (calibration.json, учитывает "рыбий глаз").
     Если левый торец всё время закрыт рабочим или уже в станке,
     берём край кадра — это оценка снизу (помечается в таблице).
  4. N — длина нормальной доски из config.toml.
     Доска — выброс, если длина <= N * (1 - погрешность/100).
  5. Пишет results/boards.csv и results/annotated.mp4 с разметкой.

Запуск (из папки проекта):
    .venv\\Scripts\\python.exe process_video.py
    .venv\\Scripts\\python.exe process_video.py --start 75 --end 350   (часть видео, секунды)
    .venv\\Scripts\\python.exe process_video.py --no-video             (только таблица)
"""

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.signal import find_peaks

from config import load_config
from lens import Lens, board_band_profile, shift_cm

# --- зона конвейера и пороги ---
BAND_Y1, BAND_Y2 = 740, 900      # строки, где лежит доска на роликах
H_RUN = 81                       # минимальная длина горизонтального участка доски, пикс
VERT_RUN = 25                    # минимальная "толщина" доски в столбце (пикселей доски), пикс
MIN_GAP_S = 8.0                  # доски кладут раз в ~14 с: ближе 8 с — не новая доска
DIFF = 25                        # насколько пиксель должен отличаться от пустого конвейера
MIN_BRIGHT = 90                  # доска светлая (тёмное — рабочий/тени)
COLUMN_ON = 0.3                  # доля строк в столбце, чтобы считать "там доска"
MAX_GAP = 40                     # разрывы внутри доски, пикс
MIN_BOARD_PX = 300               # короче — не доска
ROLLERS_HIDDEN = 0.7             # доля роликов под объектом, которые тоже изменились
RIGHT_CUT = 2630                 # правее — торец может быть за краем кадра или у стойки рамы
LEFT_CUT = 8                     # левее — торец за краем кадра (в станке)
NEW_BOARD_JUMP = 400             # прыжок правого торца вправо = новая доска
CONFIRM_FRAMES = 8               # событие должно держаться столько кадров
PUSH_START = 250                 # правый торец ушёл влево на столько пикс от максимума = толкают
GONE_FRAMES = 50                 # столько кадров без доски — доска ушла
MIN_TRAVEL = 500                 # доска должна уехать влево хотя бы на столько пикс
MIN_TRACK_OK = 50                # за рисунком доски должно удаться проследить хотя бы столько кадров
X_LEFT = 73                      # вход в станок: здесь левый торец при укладке (проверено на 1-й доске)
R_MEAS = (1700, 2600)            # где правый торец виден надёжно (не за краем кадра и не у рабочего)
FONT_PATH = "C:/Windows/Fonts/arial.ttf"


@dataclass
class Board:
    number: int
    start_frame: int
    end_frame: int = 0
    max_xr: int = 0
    last_xr: int = 0                             # последнее положение правого торца
    min_xr: int = 10 ** 9                        # самое левое положение правого торца
    pushed: bool = False                         # доску начали толкать в станок
    D: float = 0.0                               # сдвиг доски с момента укладки, см (минус = влево)
    prev_prof: object = None
    track_breaks: int = 0                        # сколько раз слежение за сдвигом прерывалось
    track_ok: int = 0                            # сколько раз сдвиг удалось измерить
    samples: list = field(default_factory=list)  # оценки длины по кадрам
    length_cm: float | None = None
    method: str = ""

    def is_confirmed(self):
        """Подтверждённая доска: уехала в станок и за её рисунком удалось проследить."""
        return self.max_xr - self.min_xr >= MIN_TRAVEL and self.track_ok >= MIN_TRACK_OK

    def finish(self):
        """Итоговая длина: по первым кадрам, где появился правый торец (меньше накопленная ошибка)."""
        if self.samples:
            self.length_cm = float(np.median(self.samples[:10]))
            self.method = "левый торец у входа в станок + сдвиг + правый торец"
        else:
            self.length_cm = None
            self.method = "правый торец не появился в кадре"

def find_rollers(gray):
    """Положения роликов на пустом конвейере (тёмные вертикальные полосы)."""
    prof = gray[770:860].astype(np.float32).mean(axis=0)
    prof = np.convolve(prof, np.ones(15) / 15, "same")
    peaks, _ = find_peaks(-prof, distance=25, prominence=12)
    return peaks[(peaks > 30) & (peaks < gray.shape[1] - 30)]


def roller_visibility(gray, rollers):
    """Для каждого ролика: виден ли он (тёмная полоса на месте)."""
    prof = gray[790:850].astype(np.float32).mean(axis=0)
    centre = np.array([prof[x - 3:x + 4].mean() for x in rollers])
    side = np.array([(prof[x - 25:x - 14].mean() + prof[x + 15:x + 26].mean()) / 2 for x in rollers])
    return side - centre > 15


def find_boards(gray, refg, rollers):
    """
    Доски в кадре: список (лево, право, верх, низ), слева направо.
    Столбец — "доска", если в нём есть сплошной вертикальный участок изменений
    высотой >= VERT_RUN пикс (доска в кадре ~75 пикс). Объект — доска, только если
    изменились и сами ролики под ним (а не только пол между роликами).
    """
    band = gray[BAND_Y1:BAND_Y2]
    ch = ((cv2.absdiff(band, refg[BAND_Y1:BAND_Y2]) > DIFF) & (band > 55)).astype(np.uint8)
    ch = cv2.morphologyEx(ch, cv2.MORPH_CLOSE, np.ones((5, 9), np.uint8))
    # Доска — длинная горизонтальная полоса: оставляем только горизонтальные участки
    # длиннее H_RUN (обрезки поперёк конвейера и пол между роликами короче).
    tall = cv2.morphologyEx(ch, cv2.MORPH_OPEN, np.ones((1, H_RUN), np.uint8))
    col = tall.sum(axis=0) >= VERT_RUN
    xs = np.nonzero(col)[0]
    if len(xs) == 0:
        return []
    segs, start, prev = [], xs[0], xs[0]
    for x in list(xs[1:]) + [10 ** 9]:
        if x - prev > MAX_GAP:
            segs.append((start, prev))
            start = x
        prev = x
    boards = []
    for l, r in segs:
        if r - l < MIN_BOARD_PX:
            continue
        under = rollers[(rollers >= l + 10) & (rollers <= r - 10)]
        if len(under) >= 3:
            covered = np.mean([col[x - 3:x + 4].any() for x in under])
            if covered < ROLLERS_HIDDEN:
                continue                              # изменился только пол между роликами
        rows = np.nonzero(tall[:, l:r + 1].mean(axis=1) > 0.3)[0]
        top, bot = (BAND_Y1 + rows.min(), BAND_Y1 + rows.max()) if len(rows) else (BAND_Y1, BAND_Y2)
        boards.append((int(l), int(r), int(top), int(min(bot, BAND_Y2))))
    return boards


def worker_over_end(gray, refg, x, top, bot):
    """Рабочий над левым торцом: тёмные изменившиеся пиксели в полосе доски рядом с торцом."""
    a, b = max(0, x - 60), x + 20
    g, r = gray[top:bot, a:b].astype(np.int16), refg[top:bot, a:b].astype(np.int16)
    return ((g < 75) & (np.abs(g - r) > 35)).mean() > 0.08


class Overlay:
    def __init__(self, scale):
        self.scale = scale
        self.font = ImageFont.truetype(FONT_PATH, 22)
        self.small = ImageFont.truetype(FONT_PATH, 17)

    def draw(self, frame, boards_in_frame, current, all_boards, n_cm, tol, t, ends_ok):
        s = self.scale
        img = cv2.resize(frame, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        limit = n_cm * (1 - tol / 100) if n_cm else None
        confirmed = [b for b in all_boards if b.number]
        for i, (l, r, top, bot) in enumerate(boards_in_frame):
            is_cur = current is not None and current.number and i == len(boards_in_frame) - 1
            color = (160, 160, 160)
            if is_cur:
                color = (0, 220, 0)
                if current.length_cm is not None and limit and current.length_cm <= limit:
                    color = (0, 0, 255)
            cv2.rectangle(img, (int(l * s), int(top * s)), (int(r * s), int(bot * s)), color, 2)
        if current is not None and boards_in_frame:
            l, r, top, bot = boards_in_frame[-1]
            yc = int((top + bot) / 2 * s)
            for x, ok in zip((l, r), ends_ok):
                cv2.circle(img, (int(x * s), yc), 7, (0, 255, 255) if ok else (0, 120, 255), -1 if ok else 2)

        pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        d = ImageDraw.Draw(pil, "RGBA")
        measured = [b for b in confirmed if b.length_cm is not None]
        n_out = sum(b.length_cm <= limit for b in measured) if limit else 0
        lines = [
            f"Время видео: {int(t // 60)}:{t % 60:04.1f}",
            f"Досок: {len(confirmed)}",
            f"Нормальных: {len(measured) - n_out}   Выбросов: {n_out}   Не измерено: {len(confirmed) - len(measured)}",
            f"N = {n_cm:.0f} см,  выброс если ≤ {limit:.0f} см",
        ]
        d.rectangle((8, 8, 660, 8 + 30 * len(lines) + 8), fill=(0, 0, 0, 160))
        for i, line in enumerate(lines):
            d.text((16, 14 + 30 * i), line, font=self.font, fill=(255, 255, 255))
        if current is not None and boards_in_frame:
            l, r, top, bot = boards_in_frame[-1]
            if not current.number:
                text = "Объект: проверяется..."
            elif current.length_cm is None:
                text = f"Доска №{current.number}: измеряется..."
            else:
                status = "ВЫБРОС" if limit and current.length_cm <= limit else "норма"
                text = f"Доска №{current.number}: {current.length_cm:.0f} см ({status})"
            x, y = int((l + r) / 2 * s) - 160, int(top * s) - 34
            d.rectangle((x - 4, y - 2, x + 340, y + 26), fill=(0, 0, 0, 160))
            d.text((x, y), text, font=self.font, fill=(255, 255, 0))
        d.text((16, pil.height - 28),
               "Зелёная рамка — доска, красная — выброс, серая — проверяется или уходит в станок.  "
               "Жёлтая точка — торец виден",
               font=self.small, fill=(255, 255, 255))
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def main():
    ap = argparse.ArgumentParser(description="Подсчёт и измерение досок")
    ap.add_argument("--start", type=float, default=0, help="начало, секунды")
    ap.add_argument("--end", type=float, default=None, help="конец, секунды")
    ap.add_argument("--scale", type=float, default=0.5, help="масштаб выходного видео")
    ap.add_argument("--no-video", action="store_true", help="не записывать видео (быстрее)")
    args = ap.parse_args()

    cfg = load_config()
    lens = Lens.load("calibration.json", marker_length_cm=cfg.marker_length_cm)
    global S_LEFT
    S_LEFT = float(lens.s(X_LEFT))
    if cfg.normal_length_cm <= 0:
        print("ОШИБКА: впишите длину нормальной доски N (boards.normal_length_cm) в config.toml")
        return
    n_cm = cfg.normal_length_cm
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(cfg.video_path)
    if not cap.isOpened():
        print(f"ОШИБКА: не открылось видео {cfg.video_path}")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    ok, frame = cap.read()
    refg = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)     # первый кадр: конвейер пуст
    rollers = find_rollers(refg)
    h, w = frame.shape[:2]

    writer = None
    if not args.no_video:
        size = (int(w * args.scale), int(h * args.scale))
        writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    overlay = Overlay(args.scale)

    boards, current = [], None
    pending, missing = 0, 0

    def close(board, k):
        board.end_frame = k
        board.finish()
        # Настоящая доска уезжает в станок. Если правый торец почти не сдвинулся —
        # это не доска (изменился пол, лежит предмет) — не считаем её.
        if not board.is_confirmed():
            boards.remove(board)
            print(f"  отброшен объект с {board.start_frame / fps:.1f} с: не уехал в станок или это не доска")
            return
        if not board.number:
            board.number = sum(1 for b in boards if b.number) + 1

    k = 0
    end_frame = int(args.end * fps) if args.end else None
    while ok:
        t = k / fps
        if k >= args.start * fps:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Образец пустого конвейера — первый кадр. Обновление образца отключено:
            # в него попадал стоящий рабочий, и длины занижались на ~6%.

            found = find_boards(gray, refg, rollers)
            ends_ok = (False, False)

            # --- отслеживание ---
            if found:
                missing = 0
                l, r, top, bot = found[-1]                 # самая правая доска — самая новая
                long_enough = r - l > 800
                is_new = long_enough and (current is None or (
                    current.pushed and r > current.last_xr + NEW_BOARD_JUMP
                    and k - current.start_frame > MIN_GAP_S * fps))
                pending = pending + 1 if is_new else 0
                if pending >= CONFIRM_FRAMES:
                    if current is not None:
                        close(current, k)
                    current = Board(number=0, start_frame=k, last_xr=r)   # номер — после подтверждения
                    boards.append(current)
                    pending = 0

                # Для текущей доски берём объект, ближайший к её прошлому положению
                # (справа может появиться кусок новой доски, которую кладут наискосок).
                if current is not None and current.last_xr:
                    near = min(found, key=lambda b: abs(b[1] - current.last_xr))
                    if abs(near[1] - current.last_xr) < 300:
                        l, r, top, bot = near
                    else:
                        l = r = None                  # текущую доску в этом кадре не нашли
                if r is None:
                    pass
                # --- измерение ---
                elif current is not None and pending == 0:
                    current.last_xr = r
                    current.min_xr = min(current.min_xr, r)
                    current.max_xr = max(current.max_xr, r if r < RIGHT_CUT else 0)
                    if current.max_xr and r < current.max_xr - PUSH_START:
                        current.pushed = True
                    # Сдвиг доски D: сравниваем рисунок центральной части доски с прошлым кадром.
                    prof, covered = board_band_profile(gray, refg, lens, min_cover=0.6)
                    if covered and current.prev_prof is not None:
                        d, score = shift_cm(current.prev_prof, prof)
                        if score > 0.5:
                            current.D += d
                            current.track_ok += 1
                        else:
                            current.track_breaks += 1
                    elif not covered:
                        current.track_breaks += 1
                    current.prev_prof = prof if covered else None
                    # Правый торец в надёжной зоне -> оценка длины:
                    # длина = (правый торец в системе доски) - (вход в станок).
                    right_ok = R_MEAS[0] < r < R_MEAS[1]
                    if right_ok:
                        current.samples.append(float(lens.s(r)) - current.D - S_LEFT)
                    ends_ok = (False, right_ok)
                    current.finish()                      # промежуточная длина для показа
                    if not current.number and current.is_confirmed():
                        current.number = sum(1 for b in boards if b.number) + 1
            else:
                missing += 1
                if current is not None and missing >= GONE_FRAMES:
                    close(current, k)
                    current = None

            if writer is not None:
                writer.write(overlay.draw(frame, found, current, boards, n_cm, cfg.tolerance_percent, t, ends_ok))
            if k % 1500 == 0:
                print(f"  {int(t // 60)}:{int(t % 60):02d}  досок: {len(boards)}", flush=True)
        k += 1
        if end_frame and k > end_frame:
            break
        ok, frame = cap.read()

    if current is not None:
        close(current, k)
    cap.release()
    if writer is not None:
        writer.release()

    limit = cfg.outlier_limit_cm
    with open(out_dir / "boards.csv", "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.writer(f, delimiter=";")
        wr.writerow(["номер", "появилась_с", "ушла_с", "длина_см", "статус", "как_измерена",
                     "кадров_с_правым_торцом", "разрывов_слежения"])
        for b in boards:
            if b.length_cm is None:
                status = "не измерена"
            else:
                status = "выброс" if b.length_cm <= limit else "норма"
            wr.writerow([b.number, f"{b.start_frame / fps:.1f}", f"{b.end_frame / fps:.1f}",
                         "" if b.length_cm is None else f"{b.length_cm:.0f}", status, b.method,
                         len(b.samples), b.track_breaks])

    print(f"\nN = {n_cm:.0f} см, выброс если длина <= {limit:.0f} см")
    print(f"Досок: {len(boards)}")
    for b in boards:
        L = "не измерена" if b.length_cm is None else f"{b.length_cm:.0f} см"
        status = "" if b.length_cm is None else ("ВЫБРОС" if b.length_cm <= limit else "норма")
        print(f"  №{b.number:2d}  {int(b.start_frame / fps // 60)}:{b.start_frame / fps % 60:04.1f}  "
              f"{L:>12s}  {status:7s} {b.method}  (кадров: {len(b.samples)}, разрывов слежения: {b.track_breaks})")
    print(f"\nСохранено: {out_dir / 'boards.csv'}" + ("" if args.no_video else f", {out_dir / 'annotated.mp4'}"))


if __name__ == "__main__":
    main()
