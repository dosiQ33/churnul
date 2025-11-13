"""
Быстрый тест: сравнение CSV vs Parquet
Просто запустите: python quick_test.py
"""
import pandas as pd
from pathlib import Path
import time


def quick_test():
    """Быстрое сравнение CSV vs Parquet"""
    print("\n" + "="*70)
    print("🚀 БЫСТРЫЙ ТЕСТ: CSV vs PARQUET")
    print("="*70)

    csv_path = Path("data/churn_train_ul.csv")
    parquet_path = Path("data/churn_train_ul.parquet")

    # Проверка наличия файлов
    if not csv_path.exists():
        print(f"\n❌ CSV файл не найден: {csv_path}")
        print("   Убедитесь, что файл находится в папке data/")
        return

    print(f"\n✅ CSV файл найден: {csv_path}")
    print(f"   Размер: {csv_path.stat().st_size / (1024**2):.2f} MB")

    # Если Parquet не существует - конвертируем
    if not parquet_path.exists():
        print(f"\n⚠ Parquet файл не найден - начинаем конвертацию...")
        print("   Это займет 2-3 минуты (только один раз!)")

        from csv_to_parquet_converter import convert_and_compare

        CATEGORICAL_FEATURES = ['segment_group', 'obs_month', 'obs_quarter']
        ID_COLUMNS = ['cli_code', 'client_id', 'observation_point']
        TARGET_COLUMN = 'target_churn_3m'

        convert_and_compare(
            csv_path=csv_path,
            parquet_path=parquet_path,
            categorical_features=CATEGORICAL_FEATURES,
            id_columns=ID_COLUMNS,
            target_column=TARGET_COLUMN,
            delimiter='|',
            encoding='windows-1251',
            compression='snappy'
        )
    else:
        print(f"✅ Parquet файл найден: {parquet_path}")
        print(f"   Размер: {parquet_path.stat().st_size / (1024**2):.2f} MB")

        # Быстрое сравнение
        print("\n" + "="*70)
        print("📊 СРАВНЕНИЕ СКОРОСТИ ЗАГРУЗКИ")
        print("="*70)

        # CSV
        print("\n1️⃣  Загрузка CSV (с вашими настройками)...")
        print("   ⏳ Подождите ~60 секунд...")
        start = time.time()
        df_csv = pd.read_csv(
            csv_path,
            delimiter='|',
            encoding='windows-1251',
            thousands=',',
            low_memory=False
        )
        csv_time = time.time() - start
        csv_memory = df_csv.memory_usage(deep=True).sum() / (1024**2)

        print(f"   ✅ Загружено за {csv_time:.2f} секунд")
        print(f"   📦 Размер: {df_csv.shape}")
        print(f"   💾 Память: {csv_memory:.2f} MB")

        # Parquet
        print("\n2️⃣  Загрузка Parquet...")
        print("   ⚡ Должно быть быстро...")
        start = time.time()
        df_parquet = pd.read_parquet(parquet_path)
        parquet_time = time.time() - start
        parquet_memory = df_parquet.memory_usage(deep=True).sum() / (1024**2)

        print(f"   ✅ Загружено за {parquet_time:.2f} секунд")
        print(f"   📦 Размер: {df_parquet.shape}")
        print(f"   💾 Память: {parquet_memory:.2f} MB")

        # Итоги
        print("\n" + "="*70)
        print("🎯 РЕЗУЛЬТАТЫ")
        print("="*70)

        speedup = csv_time / parquet_time
        csv_size = csv_path.stat().st_size / (1024**2)
        parquet_size = parquet_path.stat().st_size / (1024**2)
        compression = (1 - parquet_size/csv_size) * 100

        print(f"\n⚡ СКОРОСТЬ:")
        print(f"   CSV:     {csv_time:>8.2f} сек")
        print(f"   Parquet: {parquet_time:>8.2f} сек")
        print(f"   ➡ Parquet быстрее в {speedup:.1f}x раз!")

        print(f"\n💾 РАЗМЕР ФАЙЛА:")
        print(f"   CSV:     {csv_size:>8.2f} MB")
        print(f"   Parquet: {parquet_size:>8.2f} MB")
        print(f"   ➡ Экономия {compression:.1f}%!")

        print(f"\n🧠 ПАМЯТЬ:")
        print(f"   CSV:     {csv_memory:>8.2f} MB")
        print(f"   Parquet: {parquet_memory:>8.2f} MB")
        print(f"   ➡ Разница: {(1 - parquet_memory/csv_memory)*100:.1f}% экономии!")

        # Проверка категориальных типов
        print(f"\n🏷  ТИПЫ ДАННЫХ:")
        categorical_cols = df_parquet.select_dtypes(include='category').columns
        print(f"   CSV: все типы нужно конвертировать вручную ❌")
        print(f"   Parquet: {len(categorical_cols)} категориальных признаков сохранены ✅")
        for col in categorical_cols:
            print(f"     - {col}: {df_parquet[col].nunique()} уникальных")

        # Финальный вывод
        print("\n" + "="*70)
        if speedup > 5:
            print("✅ ОТЛИЧНО! Parquet работает как ожидалось!")
        elif speedup > 2:
            print("✅ ХОРОШО! Есть заметное ускорение!")
        else:
            print("⚠ Небольшое ускорение. Возможно, файл маленький или медленный диск.")

        print("="*70)

        print("\n💡 РЕКОМЕНДАЦИИ:")
        print("   1. Используйте Parquet для всех датасетов")
        print("   2. Обновите код: замените pd.read_csv() на pd.read_parquet()")
        print("   3. Старые CSV можно удалить (если Parquet работает)")
        print("\n   📖 Подробнее: см. PARQUET_GUIDE.md")


if __name__ == "__main__":
    try:
        quick_test()
    except KeyboardInterrupt:
        print("\n\n⚠ Прервано пользователем")
    except Exception as e:
        print(f"\n\n❌ Ошибка: {e}")
        print("\nПопробуйте:")
        print("  1. Проверить, что файл data/churn_train_ul.csv существует")
        print("  2. Установить pyarrow: pip install pyarrow")
        print("  3. Запустить снова: python quick_test.py")
