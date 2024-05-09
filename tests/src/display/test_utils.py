import pytest
from src.display.utils import fields, AutoEvalColumnQA, AutoEvalColumnLongDoc, COLS, COLS_LITE, TYPES, EVAL_COLS, QA_BENCHMARK_COLS, LONG_DOC_BENCHMARK_COLS


def test_fields():
    for c in fields(AutoEvalColumnQA):
        print(c)


def test_macro_variables():
    print(f'COLS: {COLS}')
    print(f'COLS_LITE: {COLS_LITE}')
    print(f'TYPES: {TYPES}')
    print(f'EVAL_COLS: {EVAL_COLS}')
    print(f'BENCHMARK_COLS: {QA_BENCHMARK_COLS}')
