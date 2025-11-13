"""
Скрипт для конвертации CSV в Parquet с оптимизацией памяти
Учитывает все нюансы текущей загрузки данных
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time


def optimize_dtypes(df, categorical_features, id_columns, target_column):
    """
    Оптимизирует типы данных DataFrame (как в оригинальном DataLoader)
    """
    print("\n  Оптимизация типов данных...")
    memory_before = df.memory_usage(deep=True).sum() / (1024**2)
    print(f"    Память до оптимизации: {memory_before:.2f} MB")

    # Сначала обрабатываем категориальные признаки
    for col in categorical_features:
        if col in df.columns:
            df[col] = df[col].astype('category')
            print(f"    ✓ {col}: category ({df[col].nunique()} уникальных)")

    # Оптимизация числовых колонок
    optimized_count = 0
    for col in df.columns:
        # Пропускаем ID колонки, target и уже обработанные категориальные
        if col in id_columns + [target_column] + categorical_features:
            continue

        col_type = df[col].dtype

        # Только числовые колонки
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()

            # Целочисленные типы
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                    optimized_count += 1
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                    optimized_count += 1
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                    optimized_count += 1

            # Вещественные типы
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                    optimized_count += 1

    memory_after = df.memory_usage(deep=True).sum() / (1024**2)
    savings = (1 - memory_after/memory_before) * 100

    print(f"    Память после оптимизации: {memory_after:.2f} MB")
    print(f"    Экономия: {savings:.1f}%")
    print(f"    Оптимизировано колонок: {optimized_count}")

    return df


def load_csv_with_original_settings(csv_path, delimiter='|', encoding='windows-1251'):
    """
    Загружает CSV с теми же настройками, что и в оригинальном DataLoader
    """
    print(f"\n{'='*60}")
    print(f"ЗАГРУЗКА CSV: {csv_path.name}")
    print(f"{'='*60}")

    if not csv_path.exists():
        raise FileNotFoundError(f"Файл не найден: {csv_path}")

    file_size = csv_path.stat().st_size / (1024**2)
    print(f"  Размер файла: {file_size:.2f} MB")

    print(f"\n  Загрузка CSV...")
    start_time = time.time()

    # Загружаем CSV с ОРИГИНАЛЬНЫМИ настройками
    df = pd.read_csv(
        csv_path,
        delimiter=delimiter,
        encoding=encoding,
        thousands=',',      # Важно! Для разделителя тысяч
        low_memory=False
    )

    load_time = time.time() - start_time

    # Стандартизация имен колонок (как в оригинале)
    df.columns = df.columns.str.lower().str.strip()

    print(f"  ✓ CSV загружен за {load_time:.2f} сек")
    print(f"  Размер данных: {df.shape}")
    print(f"  Колонки: {len(df.columns)}")

    return df, load_time


def save_to_parquet(df, parquet_path, compression='snappy'):
    """
    Сохраняет DataFrame в Parquet с сжатием

    Опции сжатия:
    - 'snappy': быстрое сжатие (по умолчанию)
    - 'gzip': лучшее сжатие, но медленнее
    - 'brotli': отличное сжатие, медленнее
    - 'zstd': хороший баланс скорости и сжатия
    """
    print(f"\n{'='*60}")
    print(f"СОХРАНЕНИЕ В PARQUET")
    print(f"{'='*60}")
    print(f"  Файл: {parquet_path.name}")
    print(f"  Сжатие: {compression}")

    start_time = time.time()

    df.to_parquet(
        parquet_path,
        engine='pyarrow',
        compression=compression,
        index=False
    )

    save_time = time.time() - start_time
    file_size = parquet_path.stat().st_size / (1024**2)

    print(f"  ✓ Parquet сохранен за {save_time:.2f} сек")
    print(f"  Размер файла: {file_size:.2f} MB")

    return file_size, save_time


def load_parquet(parquet_path):
    """
    Загружает Parquet файл
    """
    print(f"\n{'='*60}")
    print(f"ЗАГРУЗКА PARQUET: {parquet_path.name}")
    print(f"{'='*60}")

    if not parquet_path.exists():
        raise FileNotFoundError(f"Файл не найден: {parquet_path}")

    file_size = parquet_path.stat().st_size / (1024**2)
    print(f"  Размер файла: {file_size:.2f} MB")

    print(f"\n  Загрузка Parquet...")
    start_time = time.time()

    df = pd.read_parquet(parquet_path)

    load_time = time.time() - start_time

    print(f"  ✓ Parquet загружен за {load_time:.2f} сек")
    print(f"  Размер данных: {df.shape}")
    print(f"  Память: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    return df, load_time


def convert_and_compare(csv_path, parquet_path,
                       categorical_features, id_columns, target_column,
                       delimiter='|', encoding='windows-1251',
                       compression='snappy'):
    """
    Полный цикл: CSV → оптимизация → Parquet → загрузка обратно
    """
    print("\n" + "="*60)
    print("КОНВЕРТАЦИЯ CSV → PARQUET С ОПТИМИЗАЦИЕЙ")
    print("="*60)

    # ШАГ 1: Загрузка CSV
    df_csv, csv_load_time = load_csv_with_original_settings(
        csv_path, delimiter, encoding
    )
    csv_size = csv_path.stat().st_size / (1024**2)

    # ШАГ 2: Оптимизация типов данных
    df_optimized = optimize_dtypes(
        df_csv, categorical_features, id_columns, target_column
    )

    # ШАГ 3: Сохранение в Parquet
    parquet_size, save_time = save_to_parquet(
        df_optimized, parquet_path, compression
    )

    # ШАГ 4: Загрузка Parquet для проверки
    df_parquet, parquet_load_time = load_parquet(parquet_path)

    # ШАГ 5: Итоговое сравнение
    print(f"\n{'='*60}")
    print("📊 ИТОГОВОЕ СРАВНЕНИЕ")
    print(f"{'='*60}")

    print(f"\n{'Метрика':<30} {'CSV':<15} {'Parquet':<15} {'Улучшение':<15}")
    print("-" * 75)

    print(f"{'Размер файла (MB):':<30} {csv_size:<15.2f} {parquet_size:<15.2f} {csv_size/parquet_size:.1f}x меньше")
    print(f"{'Время загрузки (сек):':<30} {csv_load_time:<15.2f} {parquet_load_time:<15.2f} {csv_load_time/parquet_load_time:.1f}x быстрее")

    csv_memory = df_csv.memory_usage(deep=True).sum() / (1024**2)
    parquet_memory = df_parquet.memory_usage(deep=True).sum() / (1024**2)
    print(f"{'Память (MB):':<30} {csv_memory:<15.2f} {parquet_memory:<15.2f} {csv_memory/parquet_memory:.1f}x экономия")

    # Проверка целостности данных
    print(f"\n{'Проверка целостности:':<30}")
    print(f"  Размер совпадает: {df_csv.shape == df_parquet.shape}")
    print(f"  Колонки совпадают: {list(df_csv.columns) == list(df_parquet.columns)}")

    # Проверка типов категориальных признаков
    print(f"\n{'Категориальные признаки:':<30}")
    for col in categorical_features:
        if col in df_parquet.columns:
            print(f"  {col}: {df_parquet[col].dtype} (unique={df_parquet[col].nunique()})")

    # Проверка target
    if target_column in df_parquet.columns:
        print(f"\n{'Target распределение:':<30}")
        print(f"  {df_parquet[target_column].value_counts().to_dict()}")

    return df_parquet


def main():
    """
    Основная функция - конвертация всех датасетов
    """
    # Настройки из вашего проекта
    CATEGORICAL_FEATURES = ['segment_group', 'obs_month', 'obs_quarter']
    ID_COLUMNS = ['cli_code', 'client_id', 'observation_point']
    TARGET_COLUMN = 'target_churn_3m'
    DELIMITER = '|'
    ENCODING = 'windows-1251'

    # Файлы для конвертации
    data_dir = Path("data")

    datasets = [
        # (CSV путь, Parquet путь, имя)
        (data_dir / "churn_train_ul.csv", data_dir / "churn_train_ul.parquet", "Training"),
        (data_dir / "churn_prod_ul.csv", data_dir / "churn_prod_ul.parquet", "Production"),
    ]

    print("\n" + "="*60)
    print("🚀 МАССОВАЯ КОНВЕРТАЦИЯ CSV → PARQUET")
    print("="*60)
    print(f"\nНастройки:")
    print(f"  Delimiter: '{DELIMITER}'")
    print(f"  Encoding: {ENCODING}")
    print(f"  Categorical features: {CATEGORICAL_FEATURES}")
    print(f"  Compression: snappy")

    results = {}

    for csv_path, parquet_path, name in datasets:
        if not csv_path.exists():
            print(f"\n⚠ Пропущен {name}: файл {csv_path} не найден")
            continue

        print(f"\n\n{'#'*60}")
        print(f"# {name.upper()} DATASET")
        print(f"{'#'*60}")

        df = convert_and_compare(
            csv_path=csv_path,
            parquet_path=parquet_path,
            categorical_features=CATEGORICAL_FEATURES,
            id_columns=ID_COLUMNS,
            target_column=TARGET_COLUMN,
            delimiter=DELIMITER,
            encoding=ENCODING,
            compression='snappy'  # Можно изменить на 'gzip', 'zstd', 'brotli'
        )

        results[name] = {
            'csv': csv_path,
            'parquet': parquet_path,
            'shape': df.shape,
            'memory': df.memory_usage(deep=True).sum() / (1024**2)
        }

    # Финальный отчет
    print("\n\n" + "="*60)
    print("✅ КОНВЕРТАЦИЯ ЗАВЕРШЕНА")
    print("="*60)

    print(f"\n{'Dataset':<20} {'Shape':<20} {'Memory (MB)':<15} {'Parquet путь':<30}")
    print("-" * 90)

    for name, info in results.items():
        print(f"{name:<20} {str(info['shape']):<20} {info['memory']:<15.2f} {info['parquet'].name:<30}")

    print("\n💡 Теперь используйте pd.read_parquet() для загрузки данных!")
    print("   Пример: df = pd.read_parquet('data/churn_train_ul.parquet')")


if __name__ == "__main__":
    main()
