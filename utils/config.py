import os
import sys
from pathlib import Path

class ProjectConfig:
    def __init__(self, dataset_name="air-passengers"):
        """
        dataset_name: имя папки датасета на Kaggle (обычно совпадает с url-slug)
        или имя файла, если он лежит в корне.
        """
        self.IS_KAGGLE = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ
        self.IS_COLAB = 'google.colab' in sys.modules
        self.IS_LOCAL = not (self.IS_KAGGLE or self.IS_COLAB)
        
        self.dataset_name = dataset_name
        self.INPUT_DIR = self._get_input_path()
        self.OUTPUT_DIR = self._get_output_path()
        
        # Создаем папку для output, если её нет (важно для Colab/Local)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def _get_input_path(self):
        if self.IS_KAGGLE:
            # На Kaggle датасеты всегда в /kaggle/input/
            # Часто имя папки может отличаться от имени репозитория, 
            # поэтому тут может потребоваться корректировка
            candidate = Path(f"/kaggle/input/{self.dataset_name}")
            if candidate.exists():
                return candidate
            # Если не нашли по имени, берем первую папку в input (частый кейс)
            try:
                return next(Path("/kaggle/input").iterdir())
            except StopIteration:
                return Path("/kaggle/input")

        elif self.IS_COLAB:
            # В Colab обычно данные загружают в корень или маунтят диск
            # Проверяем, есть ли данные в корне
            if (Path("/content") / "data").exists():
                 return Path("/content") / "data"
            return Path("/content")

        else: # LOCAL
            # Локально ищем папку data относительно корня проекта
            # Предполагаем, что запускаем из notebooks/ или корня
            # Идем вверх, пока не найдем папку data или pyproject.toml
            current_path = Path.cwd()
            while current_path != current_path.parent:
                if (current_path / "data").exists():
                    return current_path / "data"
                current_path = current_path.parent
            
            # Если не нашли, возвращаем текущую (fallback)
            return Path.cwd()

    def _get_output_path(self):
        if self.IS_KAGGLE:
            # На Kaggle писать можно ТОЛЬКО сюда
            return Path("/kaggle/working")
        elif self.IS_COLAB:
            return Path("/content/output")
        else:
            # Локально ищем корень проекта
            current_path = Path.cwd()
            while current_path != current_path.parent:
                if (current_path / "pyproject.toml").exists():
                    return current_path / "output"
                current_path = current_path.parent
            return Path.cwd() / "output"