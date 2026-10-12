"""
Чтение настроек из config.toml и правило "доска-выброс".

Проверить настройки:
    .venv\\Scripts\\python.exe config.py
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("config.toml")


@dataclass
class Config:
    video_path: str
    marker_length_cm: float
    marker_width_cm: float
    normal_length_cm: float   # N
    tolerance_percent: float

    @property
    def outlier_limit_cm(self) -> float:
        """Доска длиной меньше или равной этому значению — выброс."""
        return self.normal_length_cm * (1 - self.tolerance_percent / 100)

    def is_outlier(self, length_cm: float) -> bool:
        return length_cm <= self.outlier_limit_cm

    def problems(self) -> list[str]:
        """Список незаполненных или странных значений."""
        result = []
        if self.marker_length_cm <= 0:
            result.append("marker.length_cm не заполнена")
        if self.marker_width_cm <= 0:
            result.append("marker.width_cm не заполнена")
        if self.normal_length_cm <= 0:
            result.append("boards.normal_length_cm (N) не заполнена")
        if not 0 < self.tolerance_percent < 100:
            result.append("boards.tolerance_percent должна быть от 0 до 100")
        return result


def load_config(path: Path = CONFIG_PATH) -> Config:
    with open(path, "rb") as f:
        data = tomllib.load(f)
    return Config(
        video_path=data["video"]["path"],
        marker_length_cm=float(data["marker"]["length_cm"]),
        marker_width_cm=float(data["marker"]["width_cm"]),
        normal_length_cm=float(data["boards"]["normal_length_cm"]),
        tolerance_percent=float(data["boards"]["tolerance_percent"]),
    )


if __name__ == "__main__":
    cfg = load_config()
    print(f"Видео:            {cfg.video_path}")
    print(f"Маркер:           {cfg.marker_length_cm} x {cfg.marker_width_cm} см")
    print(f"N:                {cfg.normal_length_cm} см")
    print(f"Погрешность:      {cfg.tolerance_percent} %")
    print(f"Выброс, если длина <= {cfg.outlier_limit_cm:.1f} см")
    for p in cfg.problems():
        print(f"ВНИМАНИЕ: {p}")
