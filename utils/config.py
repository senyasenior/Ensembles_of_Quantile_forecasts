import os
import sys
from pathlib import Path

class ProjectConfig:
    def __init__(self, dataset_name="air-passengers"):
        """
        Конфигурация проекта.
        Управляет путями и параметрами эксперимента через переменные окружения.
        """
        # --- 1. Определение среды ---
        self.IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
        self.IS_COLAB = 'google.colab' in sys.modules
        self.IS_LOCAL = not (self.IS_KAGGLE or self.IS_COLAB)
        
        # --- 2. Настройка путей ---
        self.dataset_name = dataset_name
        self.INPUT_DIR = self._get_input_path()
        self.OUTPUT_DIR = self._get_output_path()
        
        # Создаем папку для output, если её нет (важно для Colab/Local)
        if not self.IS_KAGGLE: # На Kaggle папка working уже есть
             self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # --- 3. Параметры эксперимента (из Env Vars) ---
        # Тип данных: "PASSENGERS" или "M5"
        self.SERIES_TYPE = os.environ.get("SERIES_TYPE", "PASSENGERS")
        
        # ID ряда (для M5 это id, для Passengers это имя колонки)
        # Дефолтное значение зависит от типа, но можно задать универсальный дефолт
        self.SERIES_ID = os.environ.get("SERIES_ID", "Passengers")
        
        # Горизонт прогнозирования
        self.FORECAST_HORIZON = int(os.environ.get("FORECAST_HORIZON", 28))
        
        # Квантили (передаем строкой через запятую, например "0.1,0.5,0.9")
        q_str = os.environ.get("QUANTILES", "0.1,0.25,0.5,0.75,0.9")
        self.QUANTILES = [float(x) for x in q_str.split(",")]

    def _get_input_path(self):
        if self.IS_KAGGLE:
            # На Kaggle датасеты всегда в /kaggle/input/
            candidate = Path(f"/kaggle/input/{self.dataset_name}")
            if candidate.exists():
                return candidate
            try:
                return next(Path("/kaggle/input").iterdir())
            except StopIteration:
                return Path("/kaggle/input")

        elif self.IS_COLAB:
            if (Path("/content") / "data").exists():
                 return Path("/content") / "data"
            return Path("/content")

        else: # LOCAL
            current_path = Path.cwd()
            # Ищем папку data, поднимаясь вверх, но не выше корня диска
            while current_path != current_path.parent:
                if (current_path / "data").exists():
                    return current_path / "data"
                # Дополнительная проверка: если мы в корне проекта (где pyproject.toml)
                if (current_path / "pyproject.toml").exists():
                     # Если папки data нет, но есть pyproject, значит data может быть не создана
                     # Возвращаем предполагаемый путь
                     return current_path / "data"
                current_path = current_path.parent
            
            # Fallback: текущая папка
            return Path.cwd()

    def _get_output_path(self):
        if self.IS_KAGGLE:
            return Path("/kaggle/working")
        elif self.IS_COLAB:
            return Path("/content/output")
        else:
            current_path = Path.cwd()
            while current_path != current_path.parent:
                if (current_path / "pyproject.toml").exists():
                    return current_path / "output"
                current_path = current_path.parent
            return Path.cwd() / "output"

    def __repr__(self):
        """Красивый вывод текущей конфигурации"""
        return (
            f"ProjectConfig(\n"
            f"  Environment: {'Kaggle' if self.IS_KAGGLE else 'Colab' if self.IS_COLAB else 'Local'},\n"
            f"  Input Dir: {self.INPUT_DIR},\n"
            f"  Output Dir: {self.OUTPUT_DIR},\n"
            f"  Series Type: {self.SERIES_TYPE},\n"
            f"  Series ID: {self.SERIES_ID},\n"
            f"  Horizon: {self.FORECAST_HORIZON},\n"
            f"  Quantiles: {self.QUANTILES}\n"
            f")"
        )

# Создаем глобальный экземпляр, который можно импортировать
# При импорте он сразу прочитает переменные окружения
cfg = ProjectConfig()