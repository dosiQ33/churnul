"""
Централизованная конфигурация для проекта Churn Prediction
"""

from pathlib import Path
import numpy as np


class Config:
    """Централизованная конфигурация"""

    # ВОСПРОИЗВОДИМОСТЬ
    RANDOM_SEED = 42

    # ПУТИ
    DATA_DIR = Path("data")
    OUTPUT_DIR = Path("output")
    FIGURES_DIR = Path("figures")
    MODELS_DIR = Path("models")

    # ФАЙЛЫ
    TRAIN_FILE = "churn_train_ul.parquet"

    # КОЛОНКИ
    ID_COLUMNS = ['cli_code', 'client_id', 'observation_point']
    TARGET_COLUMN = 'target_churn_3m'
    SEGMENT_COLUMN = 'segment_group'
    DATE_COLUMN = 'observation_point'
    CATEGORICAL_FEATURES = ['segment_group', 'obs_month', 'obs_quarter']

    # СЕГМЕНТЫ (ДВЕ МОДЕЛИ)
    SEGMENT_1_NAME = "Small Business"
    SEGMENT_1_VALUES = ['SMALL_BUSINESS']

    SEGMENT_2_NAME = "Middle + Large Business"
    SEGMENT_2_VALUES = ['MIDDLE_BUSINESS', 'LARGE_BUSINESS']

    # ВРЕМЕННОЕ РАЗБИЕНИЕ
    TRAIN_SIZE = 0.70
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15

    # PREPROCESSING (как в исходном файле)
    REMOVE_GAPS = True  # Gap detection
    HANDLE_OUTLIERS = True  # IQR clipping
    REMOVE_HIGH_CORRELATIONS = True  # Удаление коррелирующих признаков
    CORRELATION_THRESHOLD = 0.85  # Порог корреляции между признаками
    OUTLIER_IQR_MULTIPLIER = 1.5  # Множитель для IQR

    # CORRELATION ANALYSIS С TARGET
    CORRELATION_P_VALUE_THRESHOLD = 0.05
    DATA_LEAKAGE_THRESHOLD = 0.9
    TOP_N_CORRELATIONS = 20
    TOP_N_VISUALIZATION = 30

    @classmethod
    def create_directories(cls):
        """Создание необходимых директорий"""
        for dir_path in [cls.OUTPUT_DIR, cls.FIGURES_DIR, cls.MODELS_DIR]:
            dir_path.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_train_path(cls):
        """Путь к файлу обучающих данных"""
        return cls.DATA_DIR / cls.TRAIN_FILE

    @classmethod
    def initialize(cls):
        """Инициализация конфигурации"""
        cls.create_directories()
        np.random.seed(cls.RANDOM_SEED)
        return cls()


# Автоматическая инициализация при импорте
def get_config():
    """Получить инициализированную конфигурацию"""
    config = Config()
    config.create_directories()
    np.random.seed(config.RANDOM_SEED)
    return config
