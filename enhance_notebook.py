#!/usr/bin/env python3
"""
Улучшение ноутбука: добавление всех новых функций
"""

import json
from pathlib import Path

def add_cell(cells, cell_type, content, position=None):
    """Добавить ячейку"""
    cell = {
        "cell_type": cell_type,
        "metadata": {},
        "source": [content] if isinstance(content, list) else [content]
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
# МОДИФИКАЦИЯ 1: Изменить конфигурацию (убрать segment_group)
# ============================================================================
print("\n1. Модификация конфигурации...")
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code' and any('CATEGORICAL_FEATURES' in line for line in cell['source']):
        # Найти и заменить строку
        new_source = []
        for line in cell['source']:
            if 'CATEGORICAL_FEATURES' in line and 'segment_group' in line:
                # Заменить
                new_source.append("    CATEGORICAL_FEATURES = ['obs_month', 'obs_quarter']  # УБРАЛИ segment_group!\n")
                print(f"   ✓ Заменена строка CATEGORICAL_FEATURES в ячейке {i}")
            else:
                new_source.append(line)
        cell['source'] = new_source

        # Добавить warning
        if "segment_group удален" not in ''.join(cell['source']):
            cell['source'].append("\nprint(f\"\\n⚠️  ВАЖНО: segment_group удален из CATEGORICAL_FEATURES!\")\n")

# ============================================================================
# МОДИФИКАЦИЯ 2: Добавить анализ корреляции ПОСЛЕ EDA
# ============================================================================
print("\n2. Добавление анализа корреляции...")

# Найти позицию после EDA (после ячейки с визуализацией target)
insert_pos = None
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code' and 'figures/01_eda_target.png' in ''.join(cell['source']):
        insert_pos = i + 1
        break

if insert_pos:
    # Добавить markdown
    cells = add_cell(cells, "markdown",
        "---\\n# НОВОЕ: АНАЛИЗ КОРРЕЛЯЦИИ С ТАРГЕТОМ\\n\\nВажно для документации банка (раздел 3.5.4)",
        insert_pos)

    # Добавить код
    corr_code = """# ====================================================================================
# КОРРЕЛЯЦИЯ С ТАРГЕТОМ
# ====================================================================================

print("\\n" + "="*80)
print("АНАЛИЗ КОРРЕЛЯЦИИ ПРИЗНАКОВ С ТАРГЕТОМ")
print("="*80)

# Только числовые признаки
numeric_cols = train_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in config.ID_COLUMNS + [config.TARGET_COLUMN]]

print(f"\\nЧисловых признаков: {len(numeric_cols)}")

# Расчет корреляции
correlations = []
for col in numeric_cols:
    try:
        corr = train_df[[col, config.TARGET_COLUMN]].corr().iloc[0, 1]
        correlations.append({'feature': col, 'correlation': corr})
    except:
        pass

corr_df = pd.DataFrame(correlations)
corr_df['abs_correlation'] = corr_df['correlation'].abs()
corr_df = corr_df.sort_values('abs_correlation', ascending=False)

print(f"\\nТОП-20 признаков по корреляции с таргетом:")
print(corr_df.head(20)[['feature', 'correlation']].to_string(index=False))

# Визуализация
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Top 20 positive
top_positive = corr_df.nlargest(20, 'correlation')
axes[0].barh(range(len(top_positive)), top_positive['correlation'].values, color='green', alpha=0.7)
axes[0].set_yticks(range(len(top_positive)))
axes[0].set_yticklabels(top_positive['feature'].values, fontsize=8)
axes[0].set_xlabel('Correlation')
axes[0].set_title('ТОП-20 Положительных Корреляций с Target', fontweight='bold')
axes[0].invert_yaxis()

# Top 20 negative
top_negative = corr_df.nsmallest(20, 'correlation')
axes[1].barh(range(len(top_negative)), top_negative['correlation'].values, color='red', alpha=0.7)
axes[1].set_yticks(range(len(top_negative)))
axes[1].set_yticklabels(top_negative['feature'].values, fontsize=8)
axes[1].set_xlabel('Correlation')
axes[1].set_title('ТОП-20 Отрицательных Корреляций с Target', fontweight='bold')
axes[1].invert_yaxis()

plt.tight_layout()
plt.savefig(config.FIGURES_DIR / '01a_correlation_with_target.png', dpi=150, bbox_inches='tight')
plt.show()

print("\\n✓ Сохранено: figures/01a_correlation_with_target.png")

# Сохранить для отчета
corr_df.to_csv(config.OUTPUT_DIR / 'feature_target_correlations.csv', index=False)
print("✓ Сохранено: output/feature_target_correlations.csv")

print("="*80)
"""
    cells = add_cell(cells, "code", corr_code, insert_pos + 1)
    print(f"   ✓ Добавлен анализ корреляции на позиции {insert_pos}")

# ============================================================================
# МОДИФИКАЦИЯ 3: Убрать segment_group после разделения
# ============================================================================
print("\n3. Модификация разделения по сегментам...")
for i, cell in enumerate(cells):
    if cell['cell_type'] == 'code' and 'SEGMENT SPLIT' in ''.join(cell['source']):
        # Добавить код удаления segment_group
        removal_code = """
# УДАЛЯЕМ SEGMENT_GROUP - он больше не нужен!
if config.SEGMENT_COLUMN in seg1_train.columns:
    seg1_train = seg1_train.drop(columns=[config.SEGMENT_COLUMN])
    seg1_val = seg1_val.drop(columns=[config.SEGMENT_COLUMN])
    seg1_test = seg1_test.drop(columns=[config.SEGMENT_COLUMN])
    print(f"\\n✓ УДАЛЕН {config.SEGMENT_COLUMN} из модели 1 (константа внутри сегмента)")

"""

        # Вставить после создания seg1_test
        source = ''.join(cell['source'])
        if 'УДАЛЕН' not in source and 'seg1_test' in source:
            lines = cell['source']
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if 'seg1_test' in line and '.copy()' in line:
                    new_lines.append(removal_code)
            cell['source'] = new_lines
            print(f"   ✓ Добавлено удаление segment_group в ячейке {i}")

        # То же для seg2
        if 'seg2_test' in source and 'УДАЛЕН' not in source:
            removal_code_2 = """
# УДАЛЯЕМ SEGMENT_GROUP из модели 2
if config.SEGMENT_COLUMN in seg2_train.columns:
    seg2_train = seg2_train.drop(columns=[config.SEGMENT_COLUMN])
    seg2_val = seg2_val.drop(columns=[config.SEGMENT_COLUMN])
    seg2_test = seg2_test.drop(columns=[config.SEGMENT_COLUMN])
    print(f"\\n✓ УДАЛЕН {config.SEGMENT_COLUMN} из модели 2")

print("\\n" + "="*80)
print("ОБЪЯСНЕНИЕ: segment_group удален, так как после разделения данных")
print("он является константой внутри каждой модели и не несет информации.")
print("Он был полезен только для разделения на два сегмента.")
print("="*80)
"""
            lines = cell['source']
            new_lines = []
            for line in lines:
                new_lines.append(line)
                if 'seg2_test' in line and '.copy()' in line:
                    new_lines.append(removal_code_2)
            cell['source'] = new_lines

# ============================================================================
# Сохранение
# ============================================================================
nb['cells'] = cells
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\n✓ Модифицировано! Итого ячеек: {len(cells)}")
print(f"✓ Сохранено: {notebook_path}")

print("\n" + "="*80)
print("SUMMARY:")
print("  1. ✅ segment_group убран из CATEGORICAL_FEATURES")
print("  2. ✅ Добавлен анализ корреляции с таргетом")
print("  3. ✅ segment_group удаляется после разделения по сегментам")
print("="*80)
