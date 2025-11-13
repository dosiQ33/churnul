#!/usr/bin/env python3
"""
Добавление финальных улучшений: PSI, Decile Analysis, XGBoost, LightGBM, Class Balancing
"""

import json
from pathlib import Path

def add_cell(cells, cell_type, content, position=None):
    """Добавить ячейку"""
    source_lines = content.split('\n') if isinstance(content, str) else content
    # Ensure each line ends with \n
    source_lines = [line + '\n' if not line.endswith('\n') else line for line in source_lines]

    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": source_lines
    }
    if cell_type == "code":
        cell["execution_count"] = None
        cell["outputs"] = []

    if position is None:
        cells.append(cell)
    else:
        cells.insert(position, cell)

    return cells

# Читаем ноутбук
notebook_path = Path("Churn_Model_Enhanced_v2.ipynb")
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f"Исходное количество ячеек: {len(cells)}")

# ============================================================================
# ДОБАВЛЕНИЕ: Helper Functions
# ============================================================================
print("\n1. Добавление helper functions...")

# Найти позицию перед МОДЕЛЬ 1
insert_pos = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'markdown' and 'МОДЕЛЬ 1' in ''.join(cell['source']):
        insert_pos = i
        break

if insert_pos:
    # Markdown
    cells = add_cell(cells, "markdown",
        "---\n# НОВОЕ: HELPER FUNCTIONS\n\nФункции для PSI, Decile Analysis и работы с моделями",
        insert_pos)

    # Code
    helper_code = """# ====================================================================================
# HELPER FUNCTIONS: PSI, DECILE ANALYSIS, METRICS
# ====================================================================================

def calculate_psi(expected, actual, buckets=10):
    \"\"\"
    Calculate Population Stability Index (PSI)

    PSI < 0.1: No significant change
    0.1 <= PSI < 0.2: Moderate change
    PSI >= 0.2: Significant change (требуется пересмотр модели)
    \"\"\"
    breakpoints = np.arange(0, buckets + 1) / buckets * 100
    breakpoints = np.percentile(expected, breakpoints)
    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_percents = pd.cut(expected, breakpoints, duplicates='drop').value_counts(normalize=True).sort_index()
    actual_percents = pd.cut(actual, breakpoints, duplicates='drop').value_counts(normalize=True).sort_index()

    # Ensure same bins
    expected_percents = expected_percents.reindex(actual_percents.index, fill_value=0.001)
    actual_percents = actual_percents.reindex(expected_percents.index, fill_value=0.001)

    psi_value = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))

    return psi_value

def calculate_decile_table(y_true, y_pred_proba, n_deciles=10):
    \"\"\"
    Создать таблицу метрик по децилям (перцентилям)

    Returns DataFrame с: percentile, count, target_count, target_rate,
                        precision_cum, recall_cum, lift
    \"\"\"
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred_proba': y_pred_proba
    })

    # Сортировка по вероятности (от высокой к низкой)
    df = df.sort_values('y_pred_proba', ascending=False).reset_index(drop=True)

    # Разбиение на децили
    df['decile'] = pd.qcut(df['y_pred_proba'], q=n_deciles, labels=False, duplicates='drop') + 1
    df['decile'] = n_deciles - df['decile'] + 1  # Reverse (1 = highest prob)

    # Агрегация
    decile_table = df.groupby('decile').agg(
        count=('y_true', 'size'),
        target_count=('y_true', 'sum'),
        min_prob=('y_pred_proba', 'min'),
        max_prob=('y_pred_proba', 'max')
    ).reset_index()

    decile_table['target_rate'] = decile_table['target_count'] / decile_table['count']

    # Cumulative
    decile_table['count_cum'] = decile_table['count'].cumsum()
    decile_table['target_count_cum'] = decile_table['target_count'].cumsum()
    decile_table['precision_cum'] = decile_table['target_count_cum'] / decile_table['count_cum']

    # Recall (cumulative)
    total_targets = df['y_true'].sum()
    decile_table['recall_cum'] = decile_table['target_count_cum'] / total_targets

    # Lift
    baseline_rate = total_targets / len(df)
    decile_table['lift'] = decile_table['target_rate'] / baseline_rate

    # Rename
    decile_table = decile_table.rename(columns={'decile': 'percentile'})

    return decile_table

def prepare_data_for_catboost(df, categorical_features, exclude_cols):
    \"\"\"Подготовка данных для CatBoost\"\"\"
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[config.TARGET_COLUMN].copy() if config.TARGET_COLUMN in df.columns else None

    # Конвертация категориальных
    for cat in categorical_features:
        if cat in X.columns:
            X[cat] = X[cat].astype(str).replace('nan', 'missing')

    # Индексы категориальных
    cat_indices = [i for i, c in enumerate(feature_cols) if c in categorical_features]

    return X, y, cat_indices

def prepare_data_for_xgboost_lightgbm(df, categorical_features, exclude_cols):
    \"\"\"Подготовка данных для XGBoost/LightGBM (label encoding)\"\"\"
    feature_cols = [c for c in df.columns if c not in exclude_cols]

    X = df[feature_cols].copy()
    y = df[config.TARGET_COLUMN].copy() if config.TARGET_COLUMN in df.columns else None

    # Label encoding для категориальных
    for cat in categorical_features:
        if cat in X.columns:
            le = LabelEncoder()
            X[cat] = le.fit_transform(X[cat].astype(str))

    return X, y

def calculate_class_weights(y):
    \"\"\"Расчет весов классов\"\"\"
    n_samples = len(y)
    n_classes = 2
    n_class_0 = (y == 0).sum()
    n_class_1 = (y == 1).sum()

    weight_0 = n_samples / (n_classes * n_class_0)
    weight_1 = n_samples / (n_classes * n_class_1)

    weights = np.ones(len(y))
    weights[y == 1] = weight_1
    weights[y == 0] = weight_0

    return weights, weight_0, weight_1

def find_optimal_threshold(y_true, y_pred_proba, metric='f1'):
    \"\"\"Поиск оптимального порога\"\"\"
    thresholds = np.arange(0.1, 0.9, 0.01)
    scores = []

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        scores.append(score)

    optimal_idx = np.argmax(scores)
    return thresholds[optimal_idx], scores[optimal_idx]

def calculate_all_metrics(y_true, y_pred_proba, y_pred, threshold, dataset_name=''):
    \"\"\"Расчет всех метрик\"\"\"
    metrics = {
        'dataset': dataset_name,
        'threshold': threshold,
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }
    metrics['gini'] = 2 * metrics['roc_auc'] - 1

    cm = confusion_matrix(y_true, y_pred)
    metrics['tn'] = cm[0, 0]
    metrics['fp'] = cm[0, 1]
    metrics['fn'] = cm[1, 0]
    metrics['tp'] = cm[1, 1]

    return metrics

print("\\n✓ Helper functions определены:")
print("  - calculate_psi: PSI расчет")
print("  - calculate_decile_table: Метрики по перцентилям")
print("  - prepare_data_for_catboost: Подготовка для CatBoost")
print("  - prepare_data_for_xgboost_lightgbm: Подготовка для XGBoost/LightGBM")
print("  - calculate_class_weights: Веса классов")
print("  - find_optimal_threshold: Оптимальный порог")
print("  - calculate_all_metrics: Все метрики")
"""
    cells = add_cell(cells, "code", helper_code, insert_pos + 1)
    print(f"   ✓ Helper functions добавлены на позиции {insert_pos}")

# ============================================================================
# Сохранение промежуточное
# ============================================================================
nb['cells'] = cells
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✓ Промежуточное сохранение: {len(cells)} ячеек")

print("\n" + "="*80)
print("SUMMARY (Part 1):")
print("  1. ✅ Helper functions добавлены")
print("="*80)
print("\nПродолжение будет в следующих скриптах...")
print("Следующие шаги:")
print("  - Добавить XGBoost и LightGBM модели")
print("  - Добавить PSI анализ")
print("  - Добавить Decile Analysis")
print("  - Добавить Class Balancing эксперименты")
