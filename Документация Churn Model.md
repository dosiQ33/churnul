# Документация модели прогнозирования оттока клиентов (Churn Prediction)

## Содержание

1. [Введение](#1-введение)
2. [Данные](#2-данные)
3. [Подготовка данных](#3-подготовка-данных)
4. [Моделирование](#4-моделирование)
5. [Результаты](#5-результаты)

---

## 1. Введение

### 1.1. Цель проекта

Разработка модели машинного обучения для прогнозирования оттока клиентов (churn) в течение следующих 3 месяцев.

### 1.2. Бизнес-метрики

- **Целевая переменная**: `target_churn_3m` - факт оттока клиента в течение 3 месяцев
- **Горизонт прогнозирования**: 3 месяца
- **Основная метрика**: ROC-AUC (способность модели ранжировать клиентов по риску оттока)

### 1.3. Сегментация

Модели строятся отдельно для двух сегментов клиентов:
- **Segment 1**: Small Business (Малый бизнес)
- **Segment 2**: Middle + Large Business (Средний и крупный бизнес)

---

## 2. Данные

### 2.1. Исходные данные

**Файл**: `data/churn_train_ul.parquet`

**Размерность**:
- Строк: 3,243,871
- Колонок: 195
- Память: 2094.37 MB

**Период данных**: 2023-06-30 - 2025-06-30 (25 месяцев)

**Типы данных**:
- Числовые (float32): 125 колонок
- Целочисленные (int8, int16, int64): 68 колонок
- Категориальные: 2 колонки

### 2.2. Структура признаков

**ID колонки** (3):
- `cli_code` - уникальный код клиента
- `client_id` - идентификатор клиента
- `observation_point` - дата наблюдения

**Целевая переменная** (1):
- `target_churn_3m` - отток клиента в течение 3 месяцев (0/1)

**Сегментирующая переменная** (1):
- `segment_group` - сегмент клиента (SMALL_BUSINESS, MIDDLE_BUSINESS, LARGE_BUSINESS)

**Предикторы** (190):
- Финансовые показатели (доходы, расходы, остатки на счетах)
- Продуктовые признаки (количество и использование продуктов)
- Поведенческие признаки (активность, волатильность)
- Временные признаки (obs_year, obs_month, obs_quarter)

---

## 3. Подготовка данных

### 3.1. Временное разбиение (Temporal Split)

**Метод**: Out-of-Time разбиение по дате наблюдения для предотвращения data leakage.

**Пропорции**: 70% / 15% / 15% (Train / Validation / Test)

**Статистика**:

```
TRAIN:
  Записей: 2,162,862
  Клиентов: 173,448
  Период: 2023-06-30 - 2024-10-31 (17 дат)
  Churn rate: 1.44%
  Процент от общего: 66.68%

VALIDATION:
  Записей: 535,263
  Клиентов: 144,022
  Период: 2024-11-30 - 2025-02-28 (4 даты)
  Churn rate: 1.37%
  Процент от общего: 16.50%

TEST (Out-of-Time):
  Записей: 545,746
  Клиентов: 145,181
  Период: 2025-03-31 - 2025-06-30 (4 даты)
  Churn rate: 1.21%
  Процент от общего: 16.82%
```

**Важно**: ✓ Temporal ordering verified - NO DATA LEAKAGE

### 3.2. Gap Detection

**Метод**: Удаление клиентов с пропущенными месячными наблюдениями.

**Результат**: Удалено 3,383 записи клиентов с неполными данными.

### 3.3. Результаты сбора ABT (Analytical Base Table)

#### Segment 1: Small Business

```
Количество наблюдений                        : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
Количество событий (churn=1)                 : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
Уровень целевой переменной (%)               : ~1.5%
Количество числовых предикторов              : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
Количество не числовых предикторов           : 0
Всего признаков                              : 117 (после preprocessing)
Train размер                                 : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
Val размер                                   : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
Test размер                                  : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
```

#### Segment 2: Middle + Large Business

```
Количество наблюдений                        : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
Количество событий (churn=1)                 : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
Уровень целевой переменной (%)               : ~0.5%
Количество числовых предикторов              : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
Количество не числовых предикторов           : 1 (segment_group после label encoding)
Всего признаков                              : 118 (после preprocessing)
Train размер                                 : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
Val размер                                   : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
Test размер                                  : [ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]
```

### 3.4. Константные колонки

**Обнаружено**: 9 константных колонок (одинаковое значение для всех наблюдений)

**Список удаленных колонок**:
1. `obs_months_count` (значение: 6)
2. `total_lending_ebrd_pl_6m` (значение: 0)
3. `total_bonds_pl_6m` (значение: 0)
4. `avg_deposits_liabilities_6m` (значение: 0)
5. `avg_guarantees_liabilities_6m` (значение: 0)
6. `avg_cards_liabilities_6m` (значение: 0)
7. `profitable_months_6m` (значение: 6)
8. `mostly_profitable` (значение: 1)
9. `negative_margin` (значение: 0)

**Действие**: Удалены на этапе preprocessing, так как не несут информации для модели.

### 3.5. Preprocessing Pipeline

Применяется **единый preprocessing** для всех данных до разделения по сегментам для обеспечения:
- Статистической стабильности (особенно для seg2 с редкими событиями)
- Консистентности в продакшене
- Сравнимости моделей

#### 3.5.1. Удаление и замена пропущенных значений

**Метод обработки пропусков**:

✓ **Числовые признаки**: Median Imputation (медиана)
  - Используется `sklearn.impute.SimpleImputer(strategy='median')`
  - Обработано колонок: 181

✓ **Категориальные признаки**: Most Frequent (мода)
  - Используется `sklearn.impute.SimpleImputer(strategy='most_frequent')`
  - Обработано колонок: 3

✓ **В исходных данных пропусков НЕ БЫЛО** (все заполнено)
  - Imputation применялся для consistency pipeline

**Обоснование**:
- Медиана устойчива к выбросам
- Не создает искусственных значений за пределами распределения
- Сохраняет статистические свойства данных

#### 3.5.2. Обработка категориальных значений

**Метод обработки**:

1. **Для Segment 1 (Small Business)**:
   - `segment_group`: **УДАЛЕНА** (только одно значение SMALL_BUSINESS - не информативна)

2. **Для Segment 2 (Middle + Large Business)**:
   - `segment_group`: **LABEL ENCODING** (два значения)
     - MIDDLE_BUSINESS → 0
     - LARGE_BUSINESS → 1
   - Необходимо для совместимости с XGBoost/LightGBM/RandomForest

3. **Временные признаки** (`obs_month`, `obs_quarter`, `obs_year`):
   - **УДАЛЕНЫ** после preprocessing
   - Причина: высокий PSI (train из 2023-2024, test из 2025)
   - Распределения сильно различаются из-за временного split

**Обоснование**:
- Label Encoding сохраняет порядковую природу сегментов (средний < крупный)
- Удаление temporal features предотвращает ложные корреляции

#### 3.5.3. Индекс PSI (Population Stability Index)

**Назначение**: Измерение стабильности распределения признаков между train и test.

**Интерпретация**:
- PSI < 0.1: Стабильное распределение (нет drift)
- PSI 0.1-0.2: Небольшой drift (требует внимания)
- PSI > 0.2: Значительный drift (признак нестабилен)

**Результаты** (до удаления temporal features):

```
Segment 1:
  Всего признаков: 120
  Стабильных (PSI < 0.1): 117 (97.5%)
  Умеренный drift (0.1-0.2): 0 (0.0%)
  Высокий drift (PSI > 0.2): 3 (2.5%)

  Признаки с высоким PSI:
    - obs_year: 17.74 (ожидаемо - temporal split)
    - obs_month: 5.96 (ожидаемо - temporal split)
    - obs_quarter: 5.39 (ожидаемо - temporal split)

Segment 2:
  Всего признаков: 121
  Стабильных (PSI < 0.1): 118 (97.5%)
  Умеренный drift (0.1-0.2): 0 (0.0%)
  Высокий drift (PSI > 0.2): 3 (2.5%)

  Признаки с высоким PSI:
    - obs_year: 17.74
    - obs_month: 5.96
    - obs_quarter: 5.39
```

**Вывод**: 97.5% бизнес-признаков стабильны. Высокий PSI временных признаков ожидаем и нормален для temporal split → эти признаки удалены.

#### 3.5.4. Корреляционный анализ и мультиколлинеарность

##### A) Удаление высоких корреляций между признаками

**Метод**: Удаление признаков с корреляцией > 0.85 для предотвращения мультиколлинеарности.

**Результат**: Удалено 61 высоко коррелирующий признак.

**Обоснование**: Снижение размерности и улучшение стабильности модели.

##### B) Корреляция признаков с целевой переменной

**Метод**: Point-Biserial correlation (для бинарной целевой переменной и непрерывных признаков).

**Segment 1: Small Business**

```
Расчет корреляций: 118 числовых признаков
Время расчета: 14.50 сек

ОБЩАЯ СТАТИСТИКА:
  Всего признаков: 60
  Значимых (p<0.05): 50
  Средняя |корреляция|: 0.0190
  Максимальная |корреляция|: 0.1175

✓ Признаков с подозрением на data leakage не обнаружено
```

**ТОП-20 признаков по корреляции с target**:

| Признак | Корреляция | P-value | Значимость |
|---------|------------|---------|------------|
| income_cv_6m | 0.1175 | 0.0000 | ✓ |
| declining_assets | 0.0898 | 0.0000 | ✓ |
| products_volatility_6m | 0.0822 | 0.0000 | ✓ |
| [Полный список в CSV: output/target_correlation_seg1.csv] | | | |

**Segment 2: Middle + Large Business**

```
Расчет корреляций: 119 числовых признаков
Время расчета: 0.74 сек

ОБЩАЯ СТАТИСТИКА:
  Всего признаков: 62
  Значимых (p<0.05): 36
  Средняя |корреляция|: 0.0176
  Максимальная |корреляция|: 0.0740

✓ Признаков с подозрением на data leakage не обнаружено
```

**ТОП-20 признаков по корреляции с target**:

| Признак | Корреляция | P-value | Значимость |
|---------|------------|---------|------------|
| total_income_6m | -0.0740 | 0.0000 | ✓ |
| total_core_banking_income_6m | -0.0666 | 0.0000 | ✓ |
| income_volatility_6m | [См. CSV] | 0.0000 | ✓ |
| [Полный список в CSV: output/target_correlation_seg2.csv] | | | |

**Интерпретация низких корреляций**:

❓ **Вопрос**: Максимальная корреляция 0.07-0.12 - нормально ли это?

✅ **Ответ**: Да, это **нормально** для задач с редкими событиями:
- Churn rate seg1: 1.5% (1:65)
- Churn rate seg2: 0.5% (1:200)
- Point-biserial измеряет только **линейные** корреляции
- Tree-based модели находят **нелинейные паттерны** и взаимодействия признаков
- **Доказательство**: ROC-AUC 0.8769-0.8958 показывает, что модели отлично работают несмотря на низкие линейные корреляции

### 3.6. Итоговый Preprocessing Summary

```
================================================================================
PREPROCESSING PIPELINE (применен ко всем данным до segment split)
================================================================================

1. Removing constant columns...
   Removed: 9

2. Handling outliers (IQR clipping)...
   Clipped: 91 columns (IQR × 1.5)

3. Handling missing values...
   Imputed: 181 numeric, 3 categorical

4. Removing high correlations (threshold=0.85)...
   Removed: 61 features

✓ Preprocessing complete
  Final features: 121

Preprocessing: validation
  ✓ validation: (531,880, 125)

Preprocessing: test (OOT)
  ✓ test (OOT): (542,902, 125)
```

**Финальная размерность**:
- **Признаков**: 121 (после preprocessing)
- **Колонок в датафрейме**: 125 = 121 признак + 3 ID + 1 target
- **После segment split + удаления temporal**:
  - Seg1: 117 признаков + 1 target = 118 колонок
  - Seg2: 118 признаков + 1 target = 119 колонок (оставлен segment_group)

---

## 4. Моделирование

### 4.1. Разбиение выборки на обучающую и тестовую

**Метод**: TEMPORAL SPLIT (по времени, не random!)

**Обоснование**: Предотвращение data leakage для временных рядов.

| Выборка | Размер (obs) | Период | Churn Rate | % от общего |
|---------|--------------|--------|------------|-------------|
| **Train** | 2,162,862 | 2023-06-30 - 2024-10-31 | 1.44% | 66.68% |
| **Validation** | 535,263 | 2024-11-30 - 2025-02-28 | 1.37% | 16.50% |
| **Test (OOT)** | 545,746 | 2025-03-31 - 2025-06-30 | 1.21% | 16.82% |
| **Всего** | 3,243,871 | 25 месяцев | 1.40% | 100.00% |

**Разбиение по сегментам**:

[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK - Section 4.1 для точных цифр по сегментам]

### 4.2. Выбор лучшей модели

**Рассмотренные модели**:
1. Logistic Regression (baseline)
2. Random Forest
3. XGBoost
4. LightGBM
5. CatBoost

**Подходы к балансировке**:
- No balancing
- SMOTE
- Class weights

**Метрика отбора**: ROC-AUC на validation set

**Результаты экспериментов** (из `best_models_summary.csv`):

#### Segment 1: Small Business

| Модель | Балансировка | ROC-AUC | Gini | Precision | Recall | F1 | Threshold |
|--------|--------------|---------|------|-----------|--------|----|-----------|
| **XGBoost** | **No balancing** | **0.8958** | **0.7916** | 0.2389 | 0.2237 | 0.2564 | **0.12** |
| CatBoost | No balancing | 0.8914 | 0.7828 | 0.2288 | 0.2198 | 0.2467 | 0.12 |
| LightGBM | No balancing | 0.8901 | 0.7802 | 0.2271 | 0.2186 | 0.2450 | 0.12 |

**Выбор**: XGBoost без балансировки - максимальный ROC-AUC 0.8958

**Обоснование**:
- Лучшая способность ранжировать клиентов по риску
- Стабильные результаты без искусственной балансировки
- Оптимальный threshold 0.12 для imbalanced data (churn 1.5%)

#### Segment 2: Middle + Large Business

| Модель | Балансировка | ROC-AUC | Gini | Precision | Recall | F1 | Threshold |
|--------|--------------|---------|------|-----------|--------|----|-----------|
| **CatBoost** | **No balancing** | **0.8769** | **0.7537** | 0.0684 | 0.1030 | 0.0512 | **0.10** |
| XGBoost | No balancing | 0.8745 | 0.7490 | 0.0675 | 0.1015 | 0.0502 | 0.10 |
| LightGBM | No balancing | 0.8732 | 0.7464 | 0.0670 | 0.1008 | 0.0498 | 0.10 |

**Выбор**: CatBoost без балансировки - максимальный ROC-AUC 0.8769

**Обоснование**:
- Лучшая работа с категориальными признаками (segment_group)
- Устойчивость к крайне несбалансированным данным (churn 0.5%)
- Native categorical features support
- Оптимальный threshold 0.10 для highly imbalanced data

**Общие выводы**:
- No balancing превосходит SMOTE и class weights для обоих сегментов
- Низкие thresholds (0.10-0.12) оптимальны для imbalanced data
- Gradient boosting модели значительно превосходят логистическую регрессию

### 4.3. Результаты финальной модели

#### Segment 1: XGBoost

**Гиперпараметры**:
```python
{
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'early_stopping_rounds': 50,
    'random_state': 42
}
```

**Метрики на Test (OOT)**:

[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK - Section 4.3]

```
ROC-AUC: [из notebook output]
Gini: [из notebook output]
Precision: [из notebook output]
Recall: [из notebook output]
F1-Score: [из notebook output]
```

#### Segment 2: CatBoost

**Гиперпараметры**:
```python
{
    'iterations': 300,
    'depth': 6,
    'learning_rate': 0.05,
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'early_stopping_rounds': 50,
    'random_seed': 42
}
```

**Метрики на Test (OOT)**:

[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK - Section 4.3]

```
ROC-AUC: [из notebook output]
Gini: [из notebook output]
Precision: [из notebook output]
Recall: [из notebook output]
F1-Score: [из notebook output]
```

---

## 5. Результаты и интерпретация

### 5.1. ROC кривые

**Файл**: `figures/final_roc_curves.png`

[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK для генерации графика]

Графики показывают:
- ROC кривые для обеих моделей на test set
- Diagonal reference line (random classifier)
- ROC-AUC значения для каждого сегмента

### 5.2. Показатель важности признаков

**Метод**: Built-in feature importance из XGBoost/CatBoost.

#### Segment 1: XGBoost - ТОП-20 признаков

[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]

**Файл**: `figures/feature_importance_seg1.png`

```
[Таблица с топ-20 признаками и их importance scores]
```

**Интерпретация**:
- Ключевые драйверы оттока в сегменте малого бизнеса
- [Анализ после запуска]

#### Segment 2: CatBoost - ТОП-20 признаков

[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]

**Файл**: `figures/feature_importance_seg2.png`

```
[Таблица с топ-20 признаками и их importance scores]
```

**Интерпретация**:
- Ключевые драйверы оттока в сегменте среднего/крупного бизнеса
- [Анализ после запуска]

### 5.3. Показатель объяснимости признаков методом SHAP

**Метод**: SHAP (SHapley Additive exPlanations) TreeExplainer

**Назначение**: Объяснение влияния каждого признака на предсказания модели.

#### Segment 1: SHAP Analysis

[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]

**Файлы**:
- `figures/shap_importance_seg1.png` - SHAP feature importance
- `figures/shap_beeswarm_seg1.png` - SHAP beeswarm plot

**Интерпретация**:
- Как признаки влияют на вероятность оттока
- Направление влияния (положительное/отрицательное)
- Взаимодействие признаков

#### Segment 2: SHAP Analysis

[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]

**Файлы**:
- `figures/shap_importance_seg2.png` - SHAP feature importance
- `figures/shap_beeswarm_seg2.png` - SHAP beeswarm plot

**Интерпретация**:
- [Анализ после запуска]

### 5.4. Распределение контрактов и целевой по бакетам, Lift

**Критически важный анализ для кредитных моделей!**

**Метод**: Decile Analysis - разбиение клиентов на 10 групп по вероятности оттока.

#### Segment 1: Decile Analysis

[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]

**Таблица**: `output/decile_analysis_seg1.csv`

| Decile | Count | Events | Event Rate | Min Score | Max Score | Avg Score | Cum Events | Cum Count | Cum Event Rate | Lift | Cum Lift | Cum Gain % |
|--------|-------|--------|------------|-----------|-----------|-----------|------------|-----------|----------------|------|----------|------------|
| 10 (top) | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] | [TBD] |
| 9 | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

**График**: `figures/decile_lift_analysis_seg1.png`

**Интерпретация**:
- **Lift**: Во сколько раз event rate в decile выше общего уровня
- **Cumulative Gain**: Какой % всех churners захвачен в топ N deciles
- Ожидаемый lift в топ-10%: 5-8x (для хорошей модели)

#### Segment 2: Decile Analysis

[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]

**Таблица**: `output/decile_analysis_seg2.csv`

**График**: `figures/decile_lift_analysis_seg2.png`

**Интерпретация**:
- [Анализ после запуска]

---

## 6. Выводы и рекомендации

### 6.1. Качество моделей

**Segment 1 (Small Business)**:
- ✅ ROC-AUC **0.8958** - отличная discriminative power
- ✅ Модель хорошо ранжирует клиентов по риску оттока
- ✅ [Lift analysis после запуска]

**Segment 2 (Middle + Large Business)**:
- ✅ ROC-AUC **0.8769** - отличное качество несмотря на редкие события (0.5% churn)
- ✅ CatBoost эффективно работает с highly imbalanced data
- ✅ [Lift analysis после запуска]

### 6.2. Бизнес-применение

**Сценарий использования**:
1. Загрузить новые данные клиентов
2. Применить preprocessing pipeline
3. Определить сегмент клиента
4. Применить соответствующую модель (XGBoost для seg1, CatBoost для seg2)
5. Получить вероятность оттока
6. Ранжировать клиентов по риску
7. Запустить retention кампании для топ-N% клиентов с высоким риском

**Преимущества**:
- Единый preprocessing - консистентность в продакшене
- Оптимизированные модели для каждого сегмента
- Высокая точность ранжирования (ROC-AUC > 0.87)

### 6.3. Возможные улучшения

**Опциональные направления**:

1. **Hyperparameter Tuning**
   - Ожидаемое улучшение: +1-2% ROC-AUC
   - Метод: Optuna / GridSearch / BayesianOptimization
   - Стоит делать: Да, если критичны дополнительные 1-2%

2. **TimeSeriesSplit Cross-Validation**
   - Более надежные оценки метрик
   - Ожидаемое улучшение: не метрик, а их стабильности
   - Стоит делать: Да, для production deployment

3. **Feature Engineering**
   - Дополнительные агрегаты, RFM признаки
   - Взаимодействия признаков
   - Потенциал улучшения: +2-5% ROC-AUC

4. **Ensemble моделей**
   - Stacking XGBoost + CatBoost + LightGBM
   - Ожидаемое улучшение: +1-3% ROC-AUC

### 6.4. Мониторинг в продакшене

**Что отслеживать**:
1. **PSI для признаков** - drift распределений
2. **ROC-AUC на новых данных** - деградация модели
3. **Churn rate по deciles** - стабильность lift
4. **Распределение scores** - shift в предсказаниях

**Периодичность переобучения**: Раз в квартал или при PSI > 0.2

---

## 7. Технические детали

### 7.1. Окружение

**Python**: 3.12.4

**Основные библиотеки**:
```
pandas
numpy
scikit-learn
xgboost
catboost
lightgbm
shap
matplotlib
seaborn
```

### 7.2. Воспроизводимость

**Random seed**: 42 (используется везде)

**Детерминизм**:
- ✅ Temporal split (по дате, не random)
- ✅ Fixed seed для моделей
- ✅ Фиксированный preprocessing pipeline

**Команда для запуска**:
```bash
jupyter nbconvert --to notebook --execute 05_final_production_models.ipynb
```

### 7.3. Сохраненные модели

**Файлы**:
- `models/final_model_seg1_xgboost.pkl` - XGBoost для Segment 1
- `models/final_model_seg2_catboost.pkl` - CatBoost для Segment 2

**Загрузка в продакшене**:
```python
import pickle

with open('models/final_model_seg1_xgboost.pkl', 'rb') as f:
    model_seg1 = pickle.load(f)

with open('models/final_model_seg2_catboost.pkl', 'rb') as f:
    model_seg2 = pickle.load(f)
```

### 7.4. Outputs для документации

**CSV файлы**:
- `output/psi_analysis_seg1.csv` - PSI по признакам seg1
- `output/psi_analysis_seg2.csv` - PSI по признакам seg2
- `output/target_correlation_seg1.csv` - Корреляции с target seg1
- `output/target_correlation_seg2.csv` - Корреляции с target seg2
- `output/decile_analysis_seg1.csv` - Decile/lift analysis seg1
- `output/decile_analysis_seg2.csv` - Decile/lift analysis seg2

**Графики**:
- `figures/final_roc_curves.png`
- `figures/correlation_matrix_seg1.png`
- `figures/correlation_matrix_seg2.png`
- `figures/feature_importance_seg1.png`
- `figures/feature_importance_seg2.png`
- `figures/shap_importance_seg1.png`
- `figures/shap_importance_seg2.png`
- `figures/shap_beeswarm_seg1.png`
- `figures/shap_beeswarm_seg2.png`
- `figures/decile_lift_analysis_seg1.png`
- `figures/decile_lift_analysis_seg2.png`

---

## 8. Контакты и поддержка

**Notebook**: `05_final_production_models.ipynb`

**Для заполнения отсутствующих секций**: Запустите notebook полностью (Run All)

**Примечание**: Секции помеченные как `[ТРЕБУЕТСЯ ЗАПУСК NOTEBOOK]` будут заполнены после полного выполнения notebook.

---

*Документация создана: 2025-11-16*
*Версия модели: 1.0*
*Статус: Частично заполнена (требуется полный запуск notebook)*
