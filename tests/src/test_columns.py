import pytest

from src.benchmarks import LongDocBenchmarks, QABenchmarks
from src.columns import (
    COL_NAME_AVG,
    COL_NAME_RANK,
    COL_NAME_RERANKING_MODEL,
    COL_NAME_RETRIEVAL_MODEL,
    COL_NAME_REVISION,
    COL_NAME_TIMESTAMP,
    get_default_auto_eval_column_dict,
    get_default_col_names_and_types,
    get_fixed_col_names_and_types,
    make_autoevalcolumn,
)

# Ref: https://github.com/AIR-Bench/AIR-Bench/blob/4b27b8a8f2047a963805fcf6fb9d74be51ec440c/docs/available_tasks.md
# 24.05
# | Task | dev | test |
# | ---- | --- | ---- |
# | Long-Doc | 4 | 11 |
# | QA | 54 | 53 |
#
# 24.04
# | Task | test |
# | ---- | ---- |
# | Long-Doc | 15 |
# | QA | 13 |


@pytest.fixture()
def expected_col_names():
    return [
        "rank",
        "retrieval_model",
        "reranking_model",
        "revision",
        "timestamp",
        "average",
        "retrieval_model_link",
        "reranking_model_link",
        "is_anonymous",
    ]


@pytest.fixture()
def expected_hidden_col_names():
    return [
        "retrieval_model_link",
        "reranking_model_link",
        "is_anonymous",
    ]


def test_get_default_auto_eval_column_dict(expected_col_names, expected_hidden_col_names):
    col_list = get_default_auto_eval_column_dict()
    assert len(col_list) == 9
    hidden_cols = []
    for col_tuple, expected_col in zip(col_list, expected_col_names):
        col, _, col_content = col_tuple
        assert col == expected_col
        if col_content.hidden:
            hidden_cols.append(col)
    assert hidden_cols == expected_hidden_col_names


def test_get_fixed_col_names_and_types():
    col_names, col_types = get_fixed_col_names_and_types()
    assert len(col_names) == 6
    assert len(col_types) == 6
    expected_col_and_type = [
        (COL_NAME_RANK, "number"),
        (COL_NAME_RETRIEVAL_MODEL, "markdown"),
        (COL_NAME_RERANKING_MODEL, "markdown"),
        (COL_NAME_REVISION, "markdown"),
        (COL_NAME_TIMESTAMP, "date"),
        (COL_NAME_AVG, "number"),
    ]
    for col_name, col_type, (c_name, c_type) in zip(col_names, col_types, expected_col_and_type):
        assert col_name == c_name
        assert col_type == c_type


@pytest.mark.parametrize(
    "benchmarks, expected_benchmark_len",
    [
        (QABenchmarks, {"air_bench_2404": 13, "air_bench_2405": 53}),
        (LongDocBenchmarks, {"air_bench_2404": 15, "air_bench_2405": 11}),
    ],
)
def test_make_autoevalcolumn(benchmarks, expected_benchmark_len, expected_col_names):
    expected_default_attrs = frozenset(expected_col_names)
    for benchmark in benchmarks:
        TestEvalColumn = make_autoevalcolumn("TestEvalColumn", benchmark)
        attrs = []
        for k, v in TestEvalColumn.__dict__.items():
            if not k.startswith("__"):
                attrs.append(k)
        attrs = frozenset(attrs)
        assert expected_default_attrs.issubset(attrs)
        benchmark_attrs = attrs.difference(expected_default_attrs)
        assert len(benchmark_attrs) == expected_benchmark_len[benchmark.name]


@pytest.mark.parametrize(
    "benchmarks, expected_benchmark_len",
    [
        (QABenchmarks, {"air_bench_2404": 13, "air_bench_2405": 53}),
        (LongDocBenchmarks, {"air_bench_2404": 15, "air_bench_2405": 11}),
    ],
)
def test_get_default_col_names_and_types(
    benchmarks, expected_benchmark_len, expected_col_names, expected_hidden_col_names
):
    default_col_len = len(expected_col_names)
    hidden_col_len = len(expected_hidden_col_names)
    for benchmark in benchmarks:
        col_names, col_types = get_default_col_names_and_types(benchmark)
        assert len(col_names) == expected_benchmark_len[benchmark.name] + default_col_len - hidden_col_len
