import pytest
from src.display.utils import fields, AutoEvalColumnQA, AutoEvalColumnLongDoc, COLS_QA, COLS_LONG_DOC, COLS_LITE, TYPES, EVAL_COLS, QA_BENCHMARK_COLS, LONG_DOC_BENCHMARK_COLS


def test_fields():
    for c in fields(AutoEvalColumnQA):
        print(c)


def test_macro_variables():
    print(f'COLS_QA: {COLS_QA}')
    print(f'COLS_LONG_DOC: {COLS_LONG_DOC}')
    print(f'COLS_LITE: {COLS_LITE}')
    print(f'TYPES: {TYPES}')
    print(f'EVAL_COLS: {EVAL_COLS}')
    print(f'QA_BENCHMARK_COLS: {QA_BENCHMARK_COLS}')
    print(f'LONG_DOC_BENCHMARK_COLS: {LONG_DOC_BENCHMARK_COLS}')
