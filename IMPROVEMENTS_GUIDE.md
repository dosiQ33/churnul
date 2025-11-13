# РУКОВОДСТВО ПО УЛУЧШЕНИЯМ МОДЕЛИ ОТТОКА

## Обзор

Данный документ содержит детальные объяснения всех улучшений, внесенных в модель прогнозирования оттока клиентов.

### Файлы:
- **Churn_Model_Complete.ipynb** - оригинальная версия
- **Churn_Model_Enhanced_v2.ipynb** - улучшенная версия с базовыми доработками

---

## ЧТО УЖЕ СДЕЛАНО

### ✅ 1. Удаление segment_group после разделения

**Проблема:** После разделения данных по сегментам, признак `segment_group` становится константой внутри каждой модели и не несет информации.

**Решение:** Удалили `segment_group` из `CATEGORICAL_FEATURES` и из датафреймов после split.

**Код (уже добавлен в Churn_Model_Enhanced_v2.ipynb):**
```python
# В конфигурации
CATEGORICAL_FEATURES = ['obs_month', 'obs_quarter']  # УБРАЛИ segment_group!

# После разделения
if config.SEGMENT_COLUMN in seg1_train.columns:
    seg1_train = seg1_train.drop(columns=[config.SEGMENT_COLUMN])
    seg1_val = seg1_val.drop(columns=[config.SEGMENT_COLUMN])
    seg1_test = seg1_test.drop(columns=[config.SEGMENT_COLUMN])
```

### ✅ 2. Анализ корреляции признаков с таргетом

**Зачем:** Требуется для документации банка (раздел 3.5.4). Показывает важность признаков.

**Что добавлено:**
- Расчет корреляции всех числовых признаков с таргетом
- Визуализация ТОП-20 положительных и отрицательных корреляций
- Сохранение результатов в CSV

**Результат:** Файл `output/feature_target_correlations.csv` и график `figures/01a_correlation_with_target.png`

### ✅ 3. Helper Functions

Добавлены вспомогательные функции:
- `calculate_psi()` - для PSI анализа
- `calculate_decile_table()` - для метрик по перцентилям
- `prepare_data_for_catboost()` - подготовка данных для CatBoost
- `prepare_data_for_xgboost_lightgbm()` - для XGBoost/LightGBM
- `find_optimal_threshold()` - оптимальный порог
- `calculate_all_metrics()` - все метрики

---

## ЧТО НУЖНО ДОБАВИТЬ

### 🔄 4. Сравнение моделей: XGBoost и LightGBM

**Зачем:** Разные алгоритмы могут показать лучшие результаты на ваших данных.

**Код для добавления после обучения CatBoost:**

```python
# ====================================================================================
# МОДЕЛЬ 1: СРАВНЕНИЕ АЛГОРИТМОВ (CatBoost, XGBoost, LightGBM)
# ====================================================================================

print("\\n" + "="*80)
print("СРАВНЕНИЕ АЛГОРИТМОВ ДЛЯ МОДЕЛИ 1")
print("="*80)

# Подготовка данных для разных моделей
X_train_cb, y_train_1, cat_idx_1 = prepare_data_for_catboost(
    seg1_train, config.CATEGORICAL_FEATURES,
    config.ID_COLUMNS + [config.TARGET_COLUMN]
)
X_val_cb, y_val_1, _ = prepare_data_for_catboost(
    seg1_val, config.CATEGORICAL_FEATURES,
    config.ID_COLUMNS + [config.TARGET_COLUMN]
)
X_test_cb, y_test_1, _ = prepare_data_for_catboost(
    seg1_test, config.CATEGORICAL_FEATURES,
    config.ID_COLUMNS + [config.TARGET_COLUMN]
)

# Для XGBoost/LightGBM (label encoding)
X_train_xgb, _ = prepare_data_for_xgboost_lightgbm(
    seg1_train, config.CATEGORICAL_FEATURES,
    config.ID_COLUMNS + [config.TARGET_COLUMN]
)
X_val_xgb, _ = prepare_data_for_xgboost_lightgbm(
    seg1_val, config.CATEGORICAL_FEATURES,
    config.ID_COLUMNS + [config.TARGET_COLUMN]
)
X_test_xgb, _ = prepare_data_for_xgboost_lightgbm(
    seg1_test, config.CATEGORICAL_FEATURES,
    config.ID_COLUMNS + [config.TARGET_COLUMN]
)

# Class weights
weights_1, w0_1, w1_1 = calculate_class_weights(y_train_1)
scale_pos_weight = w1_1 / w0_1

# =============================================================================
# 1. CatBoost (уже обучена)
# =============================================================================
print("\\n1. CatBoost (обучена ранее)")

# =============================================================================
# 2. XGBoost
# =============================================================================
print("\\n2. Обучение XGBoost...")

model_xgb_1 = xgb.XGBClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=500,
    objective='binary:logistic',
    eval_metric='auc',
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=100,
    scale_pos_weight=scale_pos_weight,  # Балансировка классов
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    early_stopping_rounds=100
)

model_xgb_1.fit(
    X_train_xgb, y_train_1,
    eval_set=[(X_val_xgb, y_val_1)],
    verbose=100
)

# Predictions
y_val_pred_proba_xgb = model_xgb_1.predict_proba(X_val_xgb)[:, 1]
y_test_pred_proba_xgb = model_xgb_1.predict_proba(X_test_xgb)[:, 1]

# Optimal threshold
optimal_threshold_xgb, _ = find_optimal_threshold(y_val_1, y_val_pred_proba_xgb, 'f1')
y_test_pred_xgb = (y_test_pred_proba_xgb >= optimal_threshold_xgb).astype(int)

# Metrics
test_metrics_xgb = calculate_all_metrics(y_test_1, y_test_pred_proba_xgb, y_test_pred_xgb,
                                         optimal_threshold_xgb, 'Test (OOT)')

print(f"\\n✓ XGBoost обучен")
print(f"  Test ROC-AUC: {test_metrics_xgb['roc_auc']:.4f}")
print(f"  Test GINI: {test_metrics_xgb['gini']:.4f}")
print(f"  Test F1: {test_metrics_xgb['f1']:.4f}")

# =============================================================================
# 3. LightGBM
# =============================================================================
print("\\n3. Обучение LightGBM...")

model_lgb_1 = lgb.LGBMClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=500,
    objective='binary',
    metric='auc',
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=100,
    scale_pos_weight=scale_pos_weight,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)

model_lgb_1.fit(
    X_train_xgb, y_train_1,
    eval_set=[(X_val_xgb, y_val_1)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
)

# Predictions
y_val_pred_proba_lgb = model_lgb_1.predict_proba(X_val_xgb)[:, 1]
y_test_pred_proba_lgb = model_lgb_1.predict_proba(X_test_xgb)[:, 1]

# Optimal threshold
optimal_threshold_lgb, _ = find_optimal_threshold(y_val_1, y_val_pred_proba_lgb, 'f1')
y_test_pred_lgb = (y_test_pred_proba_lgb >= optimal_threshold_lgb).astype(int)

# Metrics
test_metrics_lgb = calculate_all_metrics(y_test_1, y_test_pred_proba_lgb, y_test_pred_lgb,
                                         optimal_threshold_lgb, 'Test (OOT)')

print(f"\\n✓ LightGBM обучен")
print(f"  Test ROC-AUC: {test_metrics_lgb['roc_auc']:.4f}")
print(f"  Test GINI: {test_metrics_lgb['gini']:.4f}")
print(f"  Test F1: {test_metrics_lgb['f1']:.4f}")

# =============================================================================
# Сравнение
# =============================================================================
print("\\n" + "="*80)
print("СРАВНЕНИЕ АЛГОРИТМОВ (Test OOT)")
print("="*80)

comparison_algorithms = pd.DataFrame([
    {
        'Algorithm': 'CatBoost',
        'ROC-AUC': test_metrics_1['roc_auc'],
        'GINI': test_metrics_1['gini'],
        'F1': test_metrics_1['f1'],
        'Precision': test_metrics_1['precision'],
        'Recall': test_metrics_1['recall']
    },
    {
        'Algorithm': 'XGBoost',
        'ROC-AUC': test_metrics_xgb['roc_auc'],
        'GINI': test_metrics_xgb['gini'],
        'F1': test_metrics_xgb['f1'],
        'Precision': test_metrics_xgb['precision'],
        'Recall': test_metrics_xgb['recall']
    },
    {
        'Algorithm': 'LightGBM',
        'ROC-AUC': test_metrics_lgb['roc_auc'],
        'GINI': test_metrics_lgb['gini'],
        'F1': test_metrics_lgb['f1'],
        'Precision': test_metrics_lgb['precision'],
        'Recall': test_metrics_lgb['recall']
    }
])

print(comparison_algorithms.to_string(index=False))

# Сохранить
comparison_algorithms.to_csv(config.OUTPUT_DIR / 'algorithm_comparison_model1.csv', index=False)
print("\\n✓ Сохранено: output/algorithm_comparison_model1.csv")
```

**Объяснение:**
- **CatBoost:** Хорошо работает с категориальными признаками, обрабатывает их нативно
- **XGBoost:** Быстр, часто дает лучшие результаты, требует label encoding категориальных
- **LightGBM:** Очень быстрый, эффективен для больших датасетов
- Все три используют `scale_pos_weight` для балансировки классов

---

### 🔄 5. PSI (Population Stability Index)

**Зачем:** Требуется в документации банка (раздел 3.5.4). Показывает стабильность распределения признаков между train и test.

**Код для добавления ПОСЛЕ preprocessing:**

```python
# ====================================================================================
# PSI ANALYSIS
# ====================================================================================

print("\\n" + "="*80)
print("PSI (POPULATION STABILITY INDEX) ANALYSIS")
print("="*80)

print("\\nРасчет PSI для проверки стабильности данных...")
print("PSI < 0.1: Нет значимых изменений")
print("0.1 <= PSI < 0.2: Умеренные изменения")
print("PSI >= 0.2: Значимые изменения (требуется пересмотр модели)\\n")

# Выбрать топ-20 важных числовых признаков по корреляции
numeric_features = [c for c in pipeline.final_features
                   if c not in config.CATEGORICAL_FEATURES]

# Расчет PSI для каждого признака
psi_results = []
for feature in numeric_features[:50]:  # топ-50 для скорости
    if feature in train_processed.columns and feature in test_processed.columns:
        try:
            psi_value = calculate_psi(
                train_processed[feature].values,
                test_processed[feature].values,
                buckets=10
            )
            psi_results.append({
                'feature': feature,
                'psi': psi_value,
                'status': 'OK' if psi_value < 0.1 else ('WARNING' if psi_value < 0.2 else 'CRITICAL')
            })
        except:
            pass

psi_df = pd.DataFrame(psi_results).sort_values('psi', ascending=False)

print("\\nТОП-20 признаков по PSI:")
print(psi_df.head(20).to_string(index=False))

# Визуализация
fig, ax = plt.subplots(figsize=(12, 8))
colors = psi_df.head(20)['status'].map({'OK': 'green', 'WARNING': 'orange', 'CRITICAL': 'red'})
ax.barh(range(len(psi_df.head(20))), psi_df.head(20)['psi'].values, color=colors, alpha=0.7)
ax.set_yticks(range(len(psi_df.head(20))))
ax.set_yticklabels(psi_df.head(20)['feature'].values, fontsize=9)
ax.set_xlabel('PSI Value')
ax.set_title('Population Stability Index (PSI) - Top 20', fontweight='bold')
ax.axvline(0.1, color='orange', linestyle='--', label='Warning threshold')
ax.axvline(0.2, color='red', linestyle='--', label='Critical threshold')
ax.legend()
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(config.FIGURES_DIR / '03_psi_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Сохранить
psi_df.to_csv(config.OUTPUT_DIR / 'psi_analysis.csv', index=False)
print("\\n✓ Сохранено: output/psi_analysis.csv")
print("✓ Сохранено: figures/03_psi_analysis.png")

# Summary
critical_count = len(psi_df[psi_df['status'] == 'CRITICAL'])
warning_count = len(psi_df[psi_df['status'] == 'WARNING'])
ok_count = len(psi_df[psi_df['status'] == 'OK'])

print(f"\\nСводка PSI:")
print(f"  OK (< 0.1): {ok_count} признаков")
print(f"  WARNING (0.1-0.2): {warning_count} признаков")
print(f"  CRITICAL (>= 0.2): {critical_count} признаков")

if critical_count > 0:
    print(f"\\n⚠️  ВНИМАНИЕ: {critical_count} признаков с критическим PSI!")
    print("Критические признаки:")
    print(psi_df[psi_df['status'] == 'CRITICAL']['feature'].tolist())

print("="*80)
```

---

### 🔄 6. Метрики по перцентилям (Decile Analysis + Lift)

**Зачем:** Требуется в документации банка (раздел 5.4). Показывает, насколько хорошо модель ранжирует клиентов.

**Код для добавления ПОСЛЕ оценки модели:**

```python
# ====================================================================================
# DECILE ANALYSIS & LIFT TABLE
# ====================================================================================

print("\\n" + "="*80)
print("DECILE ANALYSIS & LIFT (Test OOT)")
print("="*80)

# Расчет таблицы по децилям
decile_table = calculate_decile_table(y_test_1, y_test_pred_proba_1, n_deciles=10)

print("\\nТаблица метрик по перцентилям (deciles):")
print(decile_table.to_string(index=False))

# Визуализация
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Churn Rate по децилям
ax = axes[0, 0]
ax.bar(decile_table['percentile'], decile_table['target_rate'] * 100,
       color='steelblue', alpha=0.7, edgecolor='black')
ax.set_xlabel('Decile (1=highest risk)')
ax.set_ylabel('Churn Rate (%)')
ax.set_title('Churn Rate по децилям', fontweight='bold')
ax.set_xticks(decile_table['percentile'])
for i, v in enumerate(decile_table['target_rate'] * 100):
    ax.text(decile_table['percentile'].iloc[i], v, f'{v:.2f}%',
           ha='center', va='bottom', fontsize=9)

# 2. Lift
ax = axes[0, 1]
ax.bar(decile_table['percentile'], decile_table['lift'],
       color='green', alpha=0.7, edgecolor='black')
ax.axhline(1.0, color='red', linestyle='--', label='Baseline')
ax.set_xlabel('Decile (1=highest risk)')
ax.set_ylabel('Lift')
ax.set_title('Lift по децилям', fontweight='bold')
ax.set_xticks(decile_table['percentile'])
ax.legend()
for i, v in enumerate(decile_table['lift']):
    ax.text(decile_table['percentile'].iloc[i], v, f'{v:.2f}',
           ha='center', va='bottom', fontsize=9)

# 3. Cumulative Precision
ax = axes[1, 0]
ax.plot(decile_table['percentile'], decile_table['precision_cum'] * 100,
       marker='o', color='purple', linewidth=2, markersize=8)
ax.set_xlabel('Decile (1=highest risk)')
ax.set_ylabel('Cumulative Precision (%)')
ax.set_title('Cumulative Precision', fontweight='bold')
ax.set_xticks(decile_table['percentile'])
ax.grid(alpha=0.3)

# 4. Cumulative Recall
ax = axes[1, 1]
ax.plot(decile_table['percentile'], decile_table['recall_cum'] * 100,
       marker='o', color='orange', linewidth=2, markersize=8)
ax.set_xlabel('Decile (1=highest risk)')
ax.set_ylabel('Cumulative Recall (%)')
ax.set_title('Cumulative Recall', fontweight='bold')
ax.set_xticks(decile_table['percentile'])
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / '04_decile_analysis_model1.png', dpi=150, bbox_inches='tight')
plt.show()

# Сохранить
decile_table.to_csv(config.OUTPUT_DIR / 'decile_analysis_model1.csv', index=False)
print("\\n✓ Сохранено: output/decile_analysis_model1.csv")
print("✓ Сохранено: figures/04_decile_analysis_model1.png")

print(f"\\nКлючевые показатели:")
print(f"  Top 10% клиентов (decile 1):")
print(f"    Churn Rate: {decile_table.iloc[0]['target_rate']*100:.2f}%")
print(f"    Lift: {decile_table.iloc[0]['lift']:.2f}x")
print(f"  Top 30% клиентов (deciles 1-3):")
print(f"    Cumulative Recall: {decile_table.iloc[2]['recall_cum']*100:.2f}%")
print(f"    Cumulative Precision: {decile_table.iloc[2]['precision_cum']*100:.2f}%")

print("="*80)
```

**Объяснение метрик:**
- **Percentile/Decile:** Группы клиентов, 1 = самые рисковые (highest probability)
- **Target Rate:** Процент оттока в каждом децпле
- **Lift:** Во сколько раз target rate в группе выше базового (>1 = хорошо)
- **Precision (cum):** Точность среди всех предсказанных как churn до этого дециля
- **Recall (cum):** Какой % всех churn мы поймали до этого дециля

---

### 🔄 7. Балансировка классов (Undersampling, SMOTE)

**Зачем:** У вас сильный дисбаланс (1:65 для модели 1, 1:200 для модели 2). Class weights помогают, но можно улучшить F1.

**Код для добавления КАК ЭКСПЕРИМЕНТ:**

```python
# ====================================================================================
# ЭКСПЕРИМЕНТЫ С БАЛАНСИРОВКОЙ КЛАССОВ
# ====================================================================================

print("\\n" + "="*80)
print("ЭКСПЕРИМЕНТЫ: БАЛАНСИРОВКА КЛАССОВ")
print("="*80)

print("\\nТестируем 4 подхода:")
print("  1. Baseline (Class Weights) - уже обучено")
print("  2. Random Undersampling")
print("  3. SMOTE (Oversampling)")
print("  4. Hybrid (SMOTE + Undersampling)")

# Baseline - уже есть
print(f"\\n1. BASELINE (Class Weights):")
print(f"   Test F1: {test_metrics_1['f1']:.4f}")
print(f"   Test Recall: {test_metrics_1['recall']:.4f}")
print(f"   Test Precision: {test_metrics_1['precision']:.4f}")

# =============================================================================
# 2. Random Undersampling
# =============================================================================
print("\\n2. RANDOM UNDERSAMPLING...")

rus = RandomUnderSampler(random_state=42, sampling_strategy=0.3)  # 1:3 ratio
X_train_rus, y_train_rus = rus.fit_resample(X_train_1, y_train_1)

print(f"   Размер после undersampling: {len(X_train_rus):,}")
print(f"   Распределение: {Counter(y_train_rus)}")

# Обучение CatBoost без class weights (данные уже сбалансированы)
model_rus = CatBoostClassifier(
    **{k: v for k, v in config.CATBOOST_PARAMS.items()},
    verbose=0
)

pool_rus = Pool(X_train_rus, y_train_rus, cat_features=cat_idx_1)
model_rus.fit(pool_rus, eval_set=val_pool_1, plot=False)

# Eval
y_test_pred_proba_rus = model_rus.predict_proba(test_pool_1)[:, 1]
threshold_rus, _ = find_optimal_threshold(y_val_1,
                                         model_rus.predict_proba(val_pool_1)[:, 1], 'f1')
y_test_pred_rus = (y_test_pred_proba_rus >= threshold_rus).astype(int)
metrics_rus = calculate_all_metrics(y_test_1, y_test_pred_proba_rus, y_test_pred_rus,
                                   threshold_rus, 'Undersampling')

print(f"   Test F1: {metrics_rus['f1']:.4f}")
print(f"   Test Recall: {metrics_rus['recall']:.4f}")
print(f"   Test Precision: {metrics_rus['precision']:.4f}")

# =============================================================================
# 3. SMOTE
# =============================================================================
print("\\n3. SMOTE (Oversampling)...")

# SMOTE работает только с числовыми данными
# Временно закодируем категориальные
X_train_for_smote = X_train_1.copy()
for cat in config.CATEGORICAL_FEATURES:
    if cat in X_train_for_smote.columns:
        le = LabelEncoder()
        X_train_for_smote[cat] = le.fit_transform(X_train_for_smote[cat])

smote = SMOTE(random_state=42, sampling_strategy=0.3)
X_train_smote, y_train_smote = smote.fit_resample(X_train_for_smote, y_train_1)

# Обратно в строки для CatBoost
X_train_smote_cb = X_train_smote.copy()
for cat in config.CATEGORICAL_FEATURES:
    if cat in X_train_smote_cb.columns:
        X_train_smote_cb[cat] = X_train_smote_cb[cat].astype(str)

print(f"   Размер после SMOTE: {len(X_train_smote):,}")
print(f"   Распределение: {Counter(y_train_smote)}")

# Обучение
model_smote = CatBoostClassifier(
    **{k: v for k, v in config.CATBOOST_PARAMS.items()},
    verbose=0
)

pool_smote = Pool(X_train_smote_cb, y_train_smote, cat_features=cat_idx_1)
model_smote.fit(pool_smote, eval_set=val_pool_1, plot=False)

# Eval
y_test_pred_proba_smote = model_smote.predict_proba(test_pool_1)[:, 1]
threshold_smote, _ = find_optimal_threshold(y_val_1,
                                           model_smote.predict_proba(val_pool_1)[:, 1], 'f1')
y_test_pred_smote = (y_test_pred_proba_smote >= threshold_smote).astype(int)
metrics_smote = calculate_all_metrics(y_test_1, y_test_pred_proba_smote, y_test_pred_smote,
                                     threshold_smote, 'SMOTE')

print(f"   Test F1: {metrics_smote['f1']:.4f}")
print(f"   Test Recall: {metrics_smote['recall']:.4f}")
print(f"   Test Precision: {metrics_smote['precision']:.4f}")

# =============================================================================
# 4. Hybrid (SMOTE + Undersampling)
# =============================================================================
print("\\n4. HYBRID (SMOTE minority + Undersample majority)...")

# Сначала SMOTE
smote_hybrid = SMOTE(random_state=42, sampling_strategy=0.15)
X_temp, y_temp = smote_hybrid.fit_resample(X_train_for_smote, y_train_1)

# Потом Undersampling
rus_hybrid = RandomUnderSampler(random_state=42, sampling_strategy=0.5)
X_train_hybrid, y_train_hybrid = rus_hybrid.fit_resample(X_temp, y_temp)

# Обратно для CatBoost
X_train_hybrid_cb = X_train_hybrid.copy()
for cat in config.CATEGORICAL_FEATURES:
    if cat in X_train_hybrid_cb.columns:
        X_train_hybrid_cb[cat] = X_train_hybrid_cb[cat].astype(str)

print(f"   Размер после hybrid: {len(X_train_hybrid):,}")
print(f"   Распределение: {Counter(y_train_hybrid)}")

# Обучение
model_hybrid = CatBoostClassifier(
    **{k: v for k, v in config.CATBOOST_PARAMS.items()},
    verbose=0
)

pool_hybrid = Pool(X_train_hybrid_cb, y_train_hybrid, cat_features=cat_idx_1)
model_hybrid.fit(pool_hybrid, eval_set=val_pool_1, plot=False)

# Eval
y_test_pred_proba_hybrid = model_hybrid.predict_proba(test_pool_1)[:, 1]
threshold_hybrid, _ = find_optimal_threshold(y_val_1,
                                            model_hybrid.predict_proba(val_pool_1)[:, 1], 'f1')
y_test_pred_hybrid = (y_test_pred_proba_hybrid >= threshold_hybrid).astype(int)
metrics_hybrid = calculate_all_metrics(y_test_1, y_test_pred_proba_hybrid, y_test_pred_hybrid,
                                      threshold_hybrid, 'Hybrid')

print(f"   Test F1: {metrics_hybrid['f1']:.4f}")
print(f"   Test Recall: {metrics_hybrid['recall']:.4f}")
print(f"   Test Precision: {metrics_hybrid['precision']:.4f}")

# =============================================================================
# Сравнение
# =============================================================================
print("\\n" + "="*80)
print("СРАВНЕНИЕ БАЛАНСИРОВКИ (Test OOT)")
print("="*80)

balancing_comparison = pd.DataFrame([
    {
        'Method': 'Baseline (Class Weights)',
        'F1': test_metrics_1['f1'],
        'Precision': test_metrics_1['precision'],
        'Recall': test_metrics_1['recall'],
        'ROC-AUC': test_metrics_1['roc_auc']
    },
    {
        'Method': 'Random Undersampling',
        'F1': metrics_rus['f1'],
        'Precision': metrics_rus['precision'],
        'Recall': metrics_rus['recall'],
        'ROC-AUC': metrics_rus['roc_auc']
    },
    {
        'Method': 'SMOTE',
        'F1': metrics_smote['f1'],
        'Precision': metrics_smote['precision'],
        'Recall': metrics_smote['recall'],
        'ROC-AUC': metrics_smote['roc_auc']
    },
    {
        'Method': 'Hybrid (SMOTE+Under)',
        'F1': metrics_hybrid['f1'],
        'Precision': metrics_hybrid['precision'],
        'Recall': metrics_hybrid['recall'],
        'ROC-AUC': metrics_hybrid['roc_auc']
    }
])

print(balancing_comparison.to_string(index=False))

# Сохранить
balancing_comparison.to_csv(config.OUTPUT_DIR / 'balancing_comparison_model1.csv', index=False)
print("\\n✓ Сохранено: output/balancing_comparison_model1.csv")

# Рекомендация
best_method = balancing_comparison.loc[balancing_comparison['F1'].idxmax(), 'Method']
print(f"\\n💡 РЕКОМЕНДАЦИЯ: Лучший F1 показал метод '{best_method}'")

print("="*80)
```

**Объяснение:**
- **Class Weights:** Увеличивает важность меньшинства при обучении
- **Undersampling:** Удаляет часть большинства класса (быстро, но теряем данные)
- **SMOTE:** Синтезирует новые примеры меньшинства (не теряем данные, но может overfitting)
- **Hybrid:** Комбинация (баланс между потерей данных и overfitting)

---

## ИТОГОВАЯ СТРУКТУРА УЛУЧШЕННОГО НОУТБУКА

1. ✅ Импорт библиотек (добавлены xgboost, lightgbm, imblearn)
2. ✅ Конфигурация (segment_group удален из CATEGORICAL_FEATURES)
3. ✅ Загрузка данных
4. ✅ EDA
5. ✅ **НОВОЕ:** Анализ корреляции с таргетом
6. ✅ Temporal Split
7. ✅ Gap Removal
8. ✅ Preprocessing
9. ✅ Разделение по сегментам + удаление segment_group
10. ✅ **НОВОЕ:** Helper Functions
11. 🔄 **ДОБАВИТЬ:** Модель 1 - CatBoost (есть)
12. 🔄 **ДОБАВИТЬ:** Модель 1 - XGBoost
13. 🔄 **ДОБАВИТЬ:** Модель 1 - LightGBM
14. 🔄 **ДОБАВИТЬ:** Сравнение алгоритмов
15. 🔄 **ДОБАВИТЬ:** PSI Analysis
16. 🔄 **ДОБАВИТЬ:** Decile Analysis для лучшей модели
17. 🔄 **ДОБАВИТЬ:** Эксперименты с балансировкой
18. 🔄 Повторить 11-17 для Модели 2
19. 🔄 Финальное сравнение
20. 🔄 Сохранение лучших моделей

---

## РЕКОМЕНДАЦИИ

### Для Модели 1 (Small Business):
- ✅ Текущие результаты хорошие (GINI 0.78, ROC-AUC 0.89)
- 🎯 F1 можно улучшить с помощью SMOTE или Hybrid балансировки
- 🎯 Попробовать XGBoost - часто дает лучший F1

### Для Модели 2 (Middle + Large Business):
- ⚠️ F1 очень низкий (0.1157) из-за сильного дисбаланса 1:200
- 🎯 ОБЯЗАТЕЛЬНО попробовать SMOTE или Hybrid
- 🎯 Рассмотреть focal loss в CatBoost
- 🎯 Возможно, объединить с Моделью 1 или использовать transfer learning

### Для документации банка:
- ✅ Корреляция с таргетом - готова
- 🔄 PSI - добавить
- 🔄 Decile/Lift analysis - добавить
- 🔄 Сравнение алгоритмов - добавить

---

## ФАЙЛЫ РЕЗУЛЬТАТОВ

После выполнения всех улучшений у вас будут:

### CSV файлы (output/):
- `feature_target_correlations.csv` - корреляции с таргетом
- `psi_analysis.csv` - PSI для всех признаков
- `decile_analysis_model1.csv` - метрики по перцентилям
- `decile_analysis_model2.csv`
- `algorithm_comparison_model1.csv` - сравнение CatBoost/XGBoost/LightGBM
- `algorithm_comparison_model2.csv`
- `balancing_comparison_model1.csv` - сравнение методов балансировки
- `balancing_comparison_model2.csv`
- `models_comparison.csv` - итоговое сравнение

### Графики (figures/):
- `01_eda_target.png` - EDA таргета
- `01a_correlation_with_target.png` - корреляция
- `02_models_comparison.png` - сравнение моделей
- `03_psi_analysis.png` - PSI
- `04_decile_analysis_model1.png` - decile/lift
- `04_decile_analysis_model2.png`
- `05_balancing_comparison.png` - сравнение балансировки

### Модели (models/):
- Лучшие модели для каждого сегмента с метаданными

---

## КАК ИСПОЛЬЗОВАТЬ

1. **Откройте `Churn_Model_Enhanced_v2.ipynb`** - базовые улучшения уже есть
2. **Копируйте код из этого README** секциями 4-7 в нужные места ноутбука
3. **Запустите весь ноутбук** и проанализируйте результаты
4. **Выберите лучшую конфигурацию** для каждого сегмента
5. **Заполните документацию** банка используя сгенерированные файлы

---

## КОНТРОЛЬНЫЙ СПИСОК

- [x] segment_group удален из конфигурации
- [x] Анализ корреляции добавлен
- [x] segment_group удаляется после split
- [x] Helper functions добавлены
- [ ] XGBoost добавлен
- [ ] LightGBM добавлен
- [ ] PSI анализ добавлен
- [ ] Decile analysis добавлен
- [ ] Эксперименты с балансировкой выполнены
- [ ] Документация заполнена

---

**Удачи с улучшением модели! 🚀**
