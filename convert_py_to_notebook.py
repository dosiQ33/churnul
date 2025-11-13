"""
Конвертирует Python скрипт в Jupyter notebook
Поддерживает специальные комментарии:
- # %% [markdown] - начало markdown ячейки
- # %% - начало code ячейки
"""
import json
import re
from pathlib import Path


def convert_py_to_notebook(py_file, nb_file):
    """Конвертирует .py файл в .ipynb"""

    with open(py_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Разбиение на ячейки
    cells = []
    current_cell = {'type': 'code', 'content': []}

    lines = content.split('\n')
    i = 0

    while i < len(lines):
        line = lines[i]

        # Проверка маркеров
        if line.strip().startswith('# %% [markdown]'):
            # Сохранить текущую ячейку
            if current_cell['content']:
                cells.append(current_cell)

            # Начать markdown ячейку
            current_cell = {'type': 'markdown', 'content': []}
            i += 1

            # Собрать markdown content
            while i < len(lines) and not lines[i].strip().startswith('# %%'):
                md_line = lines[i]
                # Убрать leading #
                if md_line.startswith('# '):
                    md_line = md_line[2:]
                elif md_line.startswith('#'):
                    md_line = md_line[1:]

                current_cell['content'].append(md_line)
                i += 1

        elif line.strip() == '# %%':
            # Сохранить текущую ячейку
            if current_cell['content']:
                cells.append(current_cell)

            # Начать code ячейку
            current_cell = {'type': 'code', 'content': []}
            i += 1

        else:
            current_cell['content'].append(line)
            i += 1

    # Добавить последнюю ячейку
    if current_cell['content']:
        cells.append(current_cell)

    # Создать notebook structure
    notebook = {
        'cells': [],
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3'
            },
            'language_info': {
                'codemirror_mode': {'name': 'ipython', 'version': 3},
                'file_extension': '.py',
                'mimetype': 'text/x-python',
                'name': 'python',
                'nbconvert_exporter': 'python',
                'pygments_lexer': 'ipython3',
                'version': '3.8.0'
            }
        },
        'nbformat': 4,
        'nbformat_minor': 4
    }

    # Конвертировать ячейки
    for cell_data in cells:
        # Очистить пустые строки в начале и конце
        while cell_data['content'] and not cell_data['content'][0].strip():
            cell_data['content'].pop(0)
        while cell_data['content'] and not cell_data['content'][-1].strip():
            cell_data['content'].pop()

        if not cell_data['content']:
            continue

        # Добавить \n в конец каждой строки
        source = [line + '\n' for line in cell_data['content']]
        if source and not source[-1].endswith('\n'):
            source[-1] = source[-1].rstrip('\n')

        cell = {
            'cell_type': cell_data['type'],
            'metadata': {},
            'source': source
        }

        if cell_data['type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []

        notebook['cells'].append(cell)

    # Сохранить notebook
    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)

    print(f"✓ Конвертация завершена")
    print(f"  Исходный файл: {py_file}")
    print(f"  Notebook: {nb_file}")
    print(f"  Ячеек: {len(notebook['cells'])}")

    # Статистика
    n_markdown = sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')
    n_code = sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')
    print(f"    - Markdown: {n_markdown}")
    print(f"    - Code: {n_code}")


if __name__ == "__main__":
    py_file = Path("churn_model_complete_pipeline.py")
    nb_file = Path("Churn_Model_Complete.ipynb")

    if not py_file.exists():
        print(f"✗ Файл не найден: {py_file}")
        exit(1)

    convert_py_to_notebook(py_file, nb_file)

    print(f"\n✓ Готово! Откройте notebook: jupyter notebook {nb_file}")
