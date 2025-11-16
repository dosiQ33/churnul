"""
Визуализация целевой переменной в двухмерном пространстве PCA/LDA/NCA
=============================================================================

Этот скрипт загружает данные и строит визуализацию целевой переменной
используя три метода снижения размерности:
- PCA (Principal Component Analysis) - максимизирует дисперсию
- LDA (Linear Discriminant Analysis) - максимизирует разделимость классов
- NCA (Neighborhood Components Analysis) - оптимизирует kNN классификацию
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import NeighborhoodComponentsAnalysis as NCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Настройки
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
RANDOM_STATE = 42
SAMPLE_SIZE = 10000  # Для ускорения NCA (очень медленный)

print("="*80)
print("ВИЗУАЛИЗАЦИЯ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ: PCA / LDA / NCA")
print("="*80)

# ============================================================================
# 1. ЗАГРУЗКА ДАННЫХ
# ============================================================================

print("\n1. Загрузка данных...")

# Пути к данным
DATA_DIR = Path("data")
TRAIN_FILE = DATA_DIR / "churn_train_ul.parquet"

# Проверка существования файла
if not TRAIN_FILE.exists():
    print(f"\n⚠ Файл не найден: {TRAIN_FILE}")
    print("\nСоздание синтетических данных для демонстрации...")

    # Создаем синтетические данные для демонстрации
    np.random.seed(RANDOM_STATE)
    n_samples = 5000
    n_features = 50

    # Генерируем данные с двумя классами
    # Класс 0 (не churn): центрирован в начале координат
    X_class0 = np.random.randn(int(n_samples * 0.985), n_features) * 0.8
    y_class0 = np.zeros(int(n_samples * 0.985))

    # Класс 1 (churn): смещен и имеет другое распределение
    X_class1 = np.random.randn(int(n_samples * 0.015), n_features) * 1.2 + 2
    y_class1 = np.ones(int(n_samples * 0.015))

    # Объединяем
    X = np.vstack([X_class0, X_class1])
    y = np.hstack([y_class0, y_class1])

    # Создаем DataFrame для единообразия
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
    df['target_churn_3m'] = y.astype(int)

    print(f"  ✓ Создано {len(df):,} синтетических записей")
    print(f"  ✓ Признаков: {n_features}")

else:
    # Загружаем реальные данные
    print(f"  Загрузка: {TRAIN_FILE}")
    df = pd.read_parquet(TRAIN_FILE)
    print(f"  ✓ Загружено: {df.shape}")

# ============================================================================
# 2. ПОДГОТОВКА ДАННЫХ
# ============================================================================

print("\n2. Подготовка данных...")

# Определяем целевую переменную
TARGET_COL = 'target_churn_3m'
ID_COLS = ['cli_code', 'client_id', 'observation_point']

# Проверяем наличие целевой переменной
if TARGET_COL not in df.columns:
    raise ValueError(f"Целевая переменная '{TARGET_COL}' не найдена!")

# Удаляем ID колонки и выбираем только числовые признаки
feature_cols = [col for col in df.columns
                if col not in ID_COLS + [TARGET_COL]]

# Выбираем только числовые колонки
numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()

print(f"  Всего признаков: {len(feature_cols)}")
print(f"  Числовых признаков: {len(numeric_cols)}")

# Создаем X и y
X = df[numeric_cols].copy()
y = df[TARGET_COL].copy()

# Обработка пропусков (если есть)
if X.isnull().any().any():
    print("  Заполнение пропусков медианой...")
    X = X.fillna(X.median())

# Удаляем константные признаки
constant_cols = [col for col in X.columns if X[col].nunique() <= 1]
if constant_cols:
    print(f"  Удаление {len(constant_cols)} константных признаков...")
    X = X.drop(columns=constant_cols)

print(f"\n  Финальная размерность: {X.shape}")
print(f"  Целевая переменная:")
print(f"    Класс 0 (No Churn): {(y==0).sum():,} ({(y==0).mean()*100:.2f}%)")
print(f"    Класс 1 (Churn):    {(y==1).sum():,} ({(y==1).mean()*100:.2f}%)")

# Сэмплирование для ускорения (особенно для NCA)
if len(X) > SAMPLE_SIZE:
    print(f"\n  Сэмплирование {SAMPLE_SIZE:,} записей для визуализации...")
    # Стратифицированная выборка для сохранения пропорций классов
    X_sample, _, y_sample, _ = train_test_split(
        X, y,
        train_size=SAMPLE_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    X = X_sample
    y = y_sample
    print(f"  ✓ Выборка: {X.shape}")

# Стандартизация (обязательно для PCA/LDA/NCA!)
print("\n  Стандартизация признаков...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("  ✓ Данные стандартизированы")

# ============================================================================
# 3. ПРИМЕНЕНИЕ МЕТОДОВ СНИЖЕНИЯ РАЗМЕРНОСТИ
# ============================================================================

print("\n3. Применение методов снижения размерности...")

# --- PCA (Principal Component Analysis) ---
print("\n  [PCA] Principal Component Analysis...")
pca = PCA(n_components=2, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)
explained_var_pca = pca.explained_variance_ratio_
print(f"    ✓ Explained variance: PC1={explained_var_pca[0]:.4f}, PC2={explained_var_pca[1]:.4f}")
print(f"    ✓ Total: {explained_var_pca.sum():.4f}")

# --- LDA (Linear Discriminant Analysis) ---
print("\n  [LDA] Linear Discriminant Analysis...")
# LDA требует n_components <= n_classes - 1, для бинарной классификации это 1
# Но мы можем использовать 1 компоненту и добавить вторую через PCA
lda = LDA(n_components=1)
X_lda_1d = lda.fit_transform(X_scaled, y)

# Для визуализации добавим вторую компоненту через PCA на остатках
# Сначала получаем 2 компоненты PCA, затем применяем LDA к первой
lda_full = LDA(n_components=1)
X_lda_comp1 = lda_full.fit_transform(X_scaled, y)

# Вторая компонента - первая главная компонента из PCA, ортогональная к LDA
pca_for_lda = PCA(n_components=2)
X_pca_temp = pca_for_lda.fit_transform(X_scaled)
X_lda = np.column_stack([X_lda_comp1, X_pca_temp[:, 1]])

print(f"    ✓ LDA applied (1D discriminant + 1D PCA for visualization)")

# --- NCA (Neighborhood Components Analysis) ---
print("\n  [NCA] Neighborhood Components Analysis...")
print("    ⚠ NCA может занять несколько минут...")
nca = NCA(n_components=2, random_state=RANDOM_STATE, max_iter=100, verbose=0)

# NCA очень медленный, используем еще меньшую выборку если нужно
if len(X_scaled) > 5000:
    print(f"    Используем {5000} образцов для NCA...")
    idx_nca = np.random.choice(len(X_scaled), 5000, replace=False)
    X_nca = nca.fit_transform(X_scaled[idx_nca], y.iloc[idx_nca])
    y_nca = y.iloc[idx_nca]
else:
    X_nca = nca.fit_transform(X_scaled, y)
    y_nca = y

print(f"    ✓ NCA completed")

# ============================================================================
# 4. ВИЗУАЛИЗАЦИЯ
# ============================================================================

print("\n4. Создание визуализации...")

# Создаем фигуру с тремя subplot'ами
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Визуализация целевой переменной в двухмерном пространстве',
             fontsize=16, fontweight='bold', y=1.02)

# Цвета для классов
colors = ['#2ecc71', '#e74c3c']  # Зеленый для 0, Красный для 1
labels = ['No Churn (0)', 'Churn (1)']

# --- PCA Plot ---
ax = axes[0]
for class_val in [0, 1]:
    mask = y == class_val
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
              c=colors[class_val],
              label=labels[class_val],
              alpha=0.6,
              s=20,
              edgecolors='black',
              linewidth=0.3)

ax.set_xlabel(f'PC1 ({explained_var_pca[0]:.2%} variance)', fontsize=11)
ax.set_ylabel(f'PC2 ({explained_var_pca[1]:.2%} variance)', fontsize=11)
ax.set_title('PCA (Principal Component Analysis)', fontsize=13, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

# --- LDA Plot ---
ax = axes[1]
for class_val in [0, 1]:
    mask = y == class_val
    ax.scatter(X_lda[mask, 0], X_lda[mask, 1],
              c=colors[class_val],
              label=labels[class_val],
              alpha=0.6,
              s=20,
              edgecolors='black',
              linewidth=0.3)

ax.set_xlabel('LD1 (Linear Discriminant)', fontsize=11)
ax.set_ylabel('PC1 (PCA Component)', fontsize=11)
ax.set_title('LDA (Linear Discriminant Analysis)', fontsize=13, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

# --- NCA Plot ---
ax = axes[2]
for class_val in [0, 1]:
    mask = y_nca == class_val
    ax.scatter(X_nca[mask, 0], X_nca[mask, 1],
              c=colors[class_val],
              label=labels[class_val],
              alpha=0.6,
              s=20,
              edgecolors='black',
              linewidth=0.3)

ax.set_xlabel('NCA Component 1', fontsize=11)
ax.set_ylabel('NCA Component 2', fontsize=11)
ax.set_title('NCA (Neighborhood Components Analysis)', fontsize=13, fontweight='bold')
ax.legend(loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()

# Создаем директорию для сохранения
output_dir = Path("figures")
output_dir.mkdir(exist_ok=True)

# Сохраняем
output_file = output_dir / "target_visualization_pca_lda_nca.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n  ✓ Визуализация сохранена: {output_file}")

# Показываем
plt.show()

# ============================================================================
# 5. СТАТИСТИКА И ИНТЕРПРЕТАЦИЯ
# ============================================================================

print("\n" + "="*80)
print("ИНТЕРПРЕТАЦИЯ РЕЗУЛЬТАТОВ")
print("="*80)

print("\n[PCA] Principal Component Analysis:")
print("  • Несупервизированный метод (не использует метки классов)")
print("  • Находит направления максимальной дисперсии")
print(f"  • Первые 2 компоненты объясняют {explained_var_pca.sum():.2%} дисперсии")
print("  • Полезно для понимания общей структуры данных")

print("\n[LDA] Linear Discriminant Analysis:")
print("  • Супервизированный метод (использует метки классов)")
print("  • Максимизирует разделимость между классами")
print("  • Для бинарной классификации: 1 дискриминант + 1 PCA компонента")
print("  • Лучше для визуализации разделимости классов")

print("\n[NCA] Neighborhood Components Analysis:")
print("  • Супервизированный метод (использует метки классов)")
print("  • Оптимизирует метрику для kNN классификации")
print("  • Учитывает локальную структуру данных")
print("  • Наиболее вычислительно затратный метод")

print("\n" + "="*80)
print("РЕКОМЕНДАЦИИ:")
print("="*80)
print("1. PCA - для исследования общей структуры данных")
print("2. LDA - для максимальной разделимости классов")
print("3. NCA - для учета локальных паттернов и улучшения kNN")
print("\n✓ Визуализация завершена успешно!")
print("="*80)
