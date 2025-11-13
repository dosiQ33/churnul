"""
Пример использования конвертера CSV → Parquet
"""
import pandas as pd
from pathlib import Path
import time

# ============================================================
# ВАРИАНТ 1: Конвертация одного файла
# ============================================================

def example_single_file():
    """Конвертация одного файла"""
    from csv_to_parquet_converter import convert_and_compare

    print("\n" + "="*60)
    print("ПРИМЕР 1: Конвертация одного файла")
    print("="*60)

    csv_path = Path("data/churn_train_ul.csv")
    parquet_path = Path("data/churn_train_ul.parquet")

    # Настройки
    CATEGORICAL_FEATURES = ['segment_group', 'obs_month', 'obs_quarter']
    ID_COLUMNS = ['cli_code', 'client_id', 'observation_point']
    TARGET_COLUMN = 'target_churn_3m'

    # Конвертация
    df = convert_and_compare(
        csv_path=csv_path,
        parquet_path=parquet_path,
        categorical_features=CATEGORICAL_FEATURES,
        id_columns=ID_COLUMNS,
        target_column=TARGET_COLUMN,
        delimiter='|',
        encoding='windows-1251',
        compression='snappy'
    )

    print(f"\n✅ Готово! DataFrame загружен из Parquet")
    print(f"   Размер: {df.shape}")


# ============================================================
# ВАРИАНТ 2: Конвертация всех файлов
# ============================================================

def example_all_files():
    """Конвертация всех файлов проекта"""
    from csv_to_parquet_converter import main

    print("\n" + "="*60)
    print("ПРИМЕР 2: Конвертация всех файлов")
    print("="*60)

    main()


# ============================================================
# ВАРИАНТ 3: Прямая загрузка Parquet (после конвертации)
# ============================================================

def example_load_parquet():
    """Загрузка Parquet файла (самый быстрый способ)"""
    print("\n" + "="*60)
    print("ПРИМЕР 3: Загрузка Parquet напрямую")
    print("="*60)

    parquet_path = Path("data/churn_train_ul.parquet")

    if not parquet_path.exists():
        print(f"⚠ Файл {parquet_path} не найден!")
        print("   Сначала запустите конвертацию: python csv_to_parquet_converter.py")
        return

    print(f"\nЗагрузка {parquet_path}...")
    start = time.time()

    df = pd.read_parquet(parquet_path)

    load_time = time.time() - start

    print(f"✅ Загружено за {load_time:.2f} сек")
    print(f"   Размер: {df.shape}")
    print(f"   Память: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"   Колонки: {list(df.columns[:5])}...")

    # Проверка категориальных признаков
    categorical_cols = df.select_dtypes(include='category').columns
    if len(categorical_cols) > 0:
        print(f"\n   Категориальные признаки сохранены:")
        for col in categorical_cols:
            print(f"     - {col}: {df[col].nunique()} уникальных")

    return df


# ============================================================
# ВАРИАНТ 4: Частичная загрузка Parquet (только нужные колонки)
# ============================================================

def example_partial_load():
    """Загрузка только нужных колонок (супер быстро!)"""
    print("\n" + "="*60)
    print("ПРИМЕР 4: Частичная загрузка (только нужные колонки)")
    print("="*60)

    parquet_path = Path("data/churn_train_ul.parquet")

    if not parquet_path.exists():
        print(f"⚠ Файл {parquet_path} не найден!")
        return

    # Загружаем только нужные колонки
    columns = ['cli_code', 'observation_point', 'target_churn_3m',
               'segment_group', 'avg_activity_6m', 'active_months_6m']

    print(f"\nЗагрузка только {len(columns)} колонок из файла...")
    start = time.time()

    df = pd.read_parquet(parquet_path, columns=columns)

    load_time = time.time() - start

    print(f"✅ Загружено за {load_time:.2f} сек")
    print(f"   Размер: {df.shape}")
    print(f"   Память: {df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
    print(f"\n   💡 Это НАМНОГО быстрее чем загружать весь CSV!")

    return df


# ============================================================
# ВАРИАНТ 5: Сравнение CSV vs Parquet
# ============================================================

def example_comparison():
    """Прямое сравнение скорости CSV vs Parquet"""
    print("\n" + "="*60)
    print("ПРИМЕР 5: Сравнение CSV vs Parquet")
    print("="*60)

    csv_path = Path("data/churn_train_ul.csv")
    parquet_path = Path("data/churn_train_ul.parquet")

    if not csv_path.exists() or not parquet_path.exists():
        print("⚠ Файлы не найдены!")
        return

    # CSV
    print("\n📄 Загрузка CSV...")
    start = time.time()
    df_csv = pd.read_csv(csv_path, delimiter='|', encoding='windows-1251',
                         thousands=',', low_memory=False)
    csv_time = time.time() - start
    csv_size = csv_path.stat().st_size / (1024**2)
    csv_memory = df_csv.memory_usage(deep=True).sum() / (1024**2)

    # Parquet
    print("\n🚀 Загрузка Parquet...")
    start = time.time()
    df_parquet = pd.read_parquet(parquet_path)
    parquet_time = time.time() - start
    parquet_size = parquet_path.stat().st_size / (1024**2)
    parquet_memory = df_parquet.memory_usage(deep=True).sum() / (1024**2)

    # Сравнение
    print("\n" + "="*60)
    print("📊 РЕЗУЛЬТАТЫ")
    print("="*60)

    print(f"\n{'Метрика':<25} {'CSV':<15} {'Parquet':<15} {'Разница':<15}")
    print("-" * 70)
    print(f"{'Размер файла (MB)':<25} {csv_size:>10.2f}     {parquet_size:>10.2f}     {csv_size/parquet_size:.1f}x меньше")
    print(f"{'Время загрузки (сек)':<25} {csv_time:>10.2f}     {parquet_time:>10.2f}     {csv_time/parquet_time:.1f}x быстрее")
    print(f"{'Память (MB)':<25} {csv_memory:>10.2f}     {parquet_memory:>10.2f}     {csv_memory/parquet_memory:.1f}x экономия")

    print(f"\n✅ Parquet побеждает по всем параметрам!")


# ============================================================
# ГЛАВНОЕ МЕНЮ
# ============================================================

def main():
    """Главное меню примеров"""
    print("\n" + "="*60)
    print("🎯 ПРИМЕРЫ ИСПОЛЬЗОВАНИЯ CSV → PARQUET")
    print("="*60)

    print("\nВыберите пример:")
    print("  1 - Конвертация одного файла")
    print("  2 - Конвертация всех файлов (РЕКОМЕНДУЕТСЯ)")
    print("  3 - Загрузка Parquet напрямую")
    print("  4 - Частичная загрузка (только нужные колонки)")
    print("  5 - Сравнение скорости CSV vs Parquet")
    print("  0 - Выход")

    choice = input("\nВведите номер (0-5): ").strip()

    if choice == '1':
        example_single_file()
    elif choice == '2':
        example_all_files()
    elif choice == '3':
        example_load_parquet()
    elif choice == '4':
        example_partial_load()
    elif choice == '5':
        example_comparison()
    elif choice == '0':
        print("\n👋 До свидания!")
    else:
        print("\n❌ Неверный выбор!")


if __name__ == "__main__":
    # Можно запустить напрямую любой пример:

    # Для первого запуска - конвертация всех файлов
    example_all_files()

    # # Или запустить интерактивное меню
    # main()
