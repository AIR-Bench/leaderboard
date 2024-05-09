import pytest
from src.display.utils import fields, AutoEvalColumn, COLS, COLS_LITE, TYPES, EVAL_COLS, BENCHMARK_COLS


def test_fields():
    for c in fields(AutoEvalColumn):
        print(c.name)


def test_macro_variables():
    print(f'COLS: {COLS}')
    print(f'COLS_LITE: {COLS_LITE}')
    print(f'TYPES: {TYPES}')
    print(f'EVAL_COLS: {EVAL_COLS}')
    print(f'BENCHMARK_COLS: {BENCHMARK_COLS}')
