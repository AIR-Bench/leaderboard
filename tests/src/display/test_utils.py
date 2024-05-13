import pytest
from src.display.utils import fields, AutoEvalColumnQA, COLS_QA, COLS_LONG_DOC, COLS_LITE, TYPES_QA, TYPES_LONG_DOC, QA_BENCHMARK_COLS, LONG_DOC_BENCHMARK_COLS, get_default_auto_eval_column_dict


def test_fields():
    for c in fields(AutoEvalColumnQA):
        print(c)


def test_macro_variables():
    print(f'COLS_QA: {COLS_QA}')
    print(f'COLS_LONG_DOC: {COLS_LONG_DOC}')
    print(f'COLS_LITE: {COLS_LITE}')
    print(f'TYPES_QA: {TYPES_QA}')
    print(f'TYPES_LONG_DOC: {TYPES_LONG_DOC}')
    print(f'QA_BENCHMARK_COLS: {QA_BENCHMARK_COLS}')
    print(f'LONG_DOC_BENCHMARK_COLS: {LONG_DOC_BENCHMARK_COLS}')


def test_get_default_auto_eval_column_dict():
    auto_eval_column_dict_list = get_default_auto_eval_column_dict()
    assert len(auto_eval_column_dict_list) == 9

