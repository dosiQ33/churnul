"""
Скрипт для конвертации CSV в Parquet с оптимизацией
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time

def convert_csv_to_parquet(csv_path, parquet_path, delimiter='|', encoding='windows-1251'):
    """
    Конвертирует CSV в Parquet с оптимизацией типов данных
    """
    print(f"Загрузка {csv_path}...")
    start = time.time()

    # Читаем CSV
    df = pd.read_csv(
        csv_path,
        delimiter=delimiter,
        encoding=encoding,
        thousands=',',
        low_memory=False
    )

    csv_time = time.time() - start
    csv_size = Path(csv_path).stat().st_size / (1024**2)  # MB

    print(f"✓ CSV загружен за {csv_time:.2f} сек")
    print(f"  Размер файла: {csv_size:.2f} MB")
    print(f"  Форма данных: {df.shape}")
    print(f"  Память: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # Оптимизация типов данных
    print("\nОптимизация типов данных...")

    categorical_features = ['segment_group', 'obs_month', 'obs_quarter']

    for col in df.columns:
        # Категориальные признаки
        if col in categorical_features:
            df[col] = df[col].astype('category')
            print(f"  {col}: category (unique={df[col].nunique()})")
            continue

        col_type = df[col].dtype

        # Оптимизация числовых типов
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()

            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)

    print(f"✓ Оптимизация завершена")
    print(f"  Память после оптимизации: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # Сохранение в Parquet
    print(f"\nСохранение в {parquet_path}...")
    start = time.time()

    df.to_parquet(
        parquet_path,
        engine='pyarrow',
        compression='snappy',  # Быстрое сжатие (можно 'gzip' для меньшего размера)
        index=False
    )

    parquet_time = time.time() - start
    parquet_size = Path(parquet_path).stat().st_size / (1024**2)  # MB

    print(f"✓ Parquet сохранен за {parquet_time:.2f} сек")
    print(f"  Размер файла: {parquet_size:.2f} MB")
    print(f"  Компрессия: {(1 - parquet_size/csv_size)*100:.1f}%")

    # Тест загрузки Parquet
    print(f"\nТест загрузки Parquet...")
    start = time.time()
    df_test = pd.read_parquet(parquet_path)
    load_time = time.time() - start

    print(f"✓ Parquet загружен за {load_time:.2f} сек")
    print(f"  Ускорение чтения: {csv_time/load_time:.1f}x")
    print(f"  Типы данных сохранены: {(df.dtypes == df_test.dtypes).all()}")

    return df


def convert_all_datasets():
    """Конвертирует все датасеты проекта"""
    data_dir = Path("data")
    output_dir = Path("output")

    files_to_convert = [
        # Исходные данные
        (data_dir / "churn_train_ul.csv", data_dir / "churn_train_ul.parquet"),
        (data_dir / "churn_prod_ul.csv", data_dir / "churn_prod_ul.parquet"),

        # Обработанные данные
        (output_dir / "train_processed.csv", output_dir / "train_processed.parquet"),
        (output_dir / "val_processed.csv", output_dir / "val_processed.parquet"),
        (output_dir / "test_processed.csv", output_dir / "test_processed.parquet"),
        (output_dir / "prod_processed.csv", output_dir / "prod_processed.parquet"),
    ]

    print("="*60)
    print("КОНВЕРТАЦИЯ CSV → PARQUET")
    print("="*60)

    for csv_path, parquet_path in files_to_convert:
        if not csv_path.exists():
            print(f"\n⚠ Пропущен: {csv_path} (файл не найден)")
            continue

        print(f"\n{'='*60}")
        print(f"Файл: {csv_path.name}")
        print(f"{'='*60}")

        convert_csv_to_parquet(csv_path, parquet_path)

    print(f"\n{'='*60}")
    print("✓ ВСЕ ФАЙЛЫ КОНВЕРТИРОВАНЫ")
    print(f"{'='*60}")


def compare_read_speeds():
    """Сравнение скорости чтения CSV vs Parquet"""
    print("\n" + "="*60)
    print("СРАВНЕНИЕ СКОРОСТИ ЧТЕНИЯ")
    print("="*60)

    csv_path = Path("output/train_processed.csv")
    parquet_path = Path("output/train_processed.parquet")

    if not csv_path.exists() or not parquet_path.exists():
        print("⚠ Файлы не найдены для сравнения")
        return

    # CSV
    print("\n📄 CSV:")
    start = time.time()
    df_csv = pd.read_csv(csv_path, delimiter='|')
    csv_time = time.time() - start
    csv_size = csv_path.stat().st_size / (1024**2)
    print(f"  Время: {csv_time:.2f} сек")
    print(f"  Размер: {csv_size:.2f} MB")
    print(f"  Память: {df_csv.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    # Parquet
    print("\n🚀 Parquet:")
    start = time.time()
    df_parquet = pd.read_parquet(parquet_path)
    parquet_time = time.time() - start
    parquet_size = parquet_path.stat().st_size / (1024**2)
    print(f"  Время: {parquet_time:.2f} сек")
    print(f"  Размер: {parquet_size:.2f} MB")
    print(f"  Память: {df_parquet.memory_usage(deep=True).sum() / (1024**2):.2f} MB")

    print(f"\n📊 РЕЗУЛЬТАТ:")
    print(f"  Parquet быстрее в {csv_time/parquet_time:.1f}x раз")
    print(f"  Parquet меньше в {csv_size/parquet_size:.1f}x раз")

    # Частичное чтение (только некоторые колонки)
    print(f"\n📖 ЧАСТИЧНОЕ ЧТЕНИЕ (5 колонок):")

    columns_to_read = ['cli_code', 'observation_point', 'target_churn_3m',
                      'segment_group', 'avg_activity_6m']

    start = time.time()
    df_partial = pd.read_parquet(parquet_path, columns=columns_to_read)
    partial_time = time.time() - start

    print(f"  Parquet (частично): {partial_time:.2f} сек")
    print(f"  Ускорение: {parquet_time/partial_time:.1f}x (по сравнению с полным чтением)")
    print(f"  ⚠ CSV не поддерживает частичное чтение колонок!")


if __name__ == "__main__":
    # Конвертировать все файлы
    convert_all_datasets()

    # Сравнить скорость
    compare_read_speeds()
