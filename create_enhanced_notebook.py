#!/usr/bin/env python3
"""
Создание полного улучшенного ноутбука Churn_Model_Enhanced.ipynb
"""

import json
from pathlib import Path

# Базовый скелет ноутбука
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.8.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

def add_markdown_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [text] if isinstance(text, list) else [text]
    }

def add_code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code] if isinstance(code, list) else [code]
    }

# Начинаем добавлять ячейки
cells = []

# ============================================================================
# ЗАГОЛОВОК
# ============================================================================
cells.append(add_markdown_cell("""# МОДЕЛЬ ПРОГНОЗИРОВАНИЯ ОТТОКА КЛИЕНТОВ БАНКА
## Улучшенная версия с полным анализом

===============================================================================

**Дата:** 2025-01-13
**Версия:** 2.0 (Enhanced)
**Алгоритмы:** CatBoost, XGBoost, LightGBM

## УЛУЧШЕНИЯ:
1. ✅ Удаление segment_group после разделения по сегментам
2. ✅ Сравнение 3 алгоритмов: CatBoost, XGBoost, LightGBM
3. ✅ Анализ корреляции фичей с таргетом
4. ✅ PSI (Population Stability Index) анализ
5. ✅ Метрики по перцентилям (Decile Analysis, Lift, Cumulative Precision/Recall)
6. ✅ Техники балансировки классов: Undersampling, SMOTE, Class Weights
7. ✅ Полная документация для банка

## ОСОБЕННОСТИ:
- Две модели по сегментам:
  * Модель 1: Малый бизнес (SMALL_BUSINESS)
  * Модель 2: Средний + Крупный бизнес (MIDDLE + LARGE_BUSINESS)
- Temporal Split (Train/Val/Test-OOT)
- Полная воспроизводимость (random_seed=42)

==============================================================================="""))

cells.append(add_markdown_cell("# 1. ИМПОРТ БИБЛИОТЕК И КОНФИГУРАЦИЯ"))

cells.append(add_code_cell("""# ====================================================================================
# ИМПОРТ БИБЛИОТЕК
# ====================================================================================

import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
import json
import pickle
import time
import gc

# Данные
import numpy as np
import pandas as pd
from scipy import stats

# Визуализация
import matplotlib.pyplot as plt
import seaborn as sns

# ML
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier, Pool
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    classification_report, confusion_matrix,
    accuracy_score, f1_score, precision_score, recall_score
)

# Балансировка классов
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from collections import Counter

# Настройки
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.4f' % x)

print("="*80)
print("CHURN PREDICTION MODEL - УЛУЧШЕННЫЙ ПАЙПЛАЙН v2.0")
print("="*80)
print(f"✓ Библиотеки импортированы")
print(f"  Pandas: {pd.__version__}")
print(f"  NumPy: {np.__version__}")
print(f"  XGBoost: {xgb.__version__}")
print(f"  LightGBM: {lgb.__version__}")
print(f"  Дата запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*80)"""))

print("Генерация ноутбука...")
print(f"Добавлено {len(cells)} начальных ячеек")

# Сохраняем промежуточно
notebook['cells'] = cells
output_path = Path("Churn_Model_Enhanced_FULL.ipynb")

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"✓ Создан начальный ноутбук: {output_path}")
print("Продолжение следует...")
