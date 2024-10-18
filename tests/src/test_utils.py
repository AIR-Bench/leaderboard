from pathlib import Path

import pandas as pd
import pytest

from src.columns import COL_NAME_RERANKING_MODEL, COL_NAME_RETRIEVAL_MODEL
from src.models import TaskType, model_hyperlink
from src.utils import (
    _update_df_elem,
    calculate_mean,
    filter_models,
    filter_queries,
    get_default_cols,
    get_leaderboard_df,
    get_selected_cols,
    remove_html,
    select_columns,
)

cur_fp = Path(__file__)

NUM_QA_BENCHMARKS_24_05 = 53
NUM_DOC_BENCHMARKS_24_05 = 11
NUM_QA_BENCHMARKS_24_04 = 13
NUM_DOC_BENCHMARKS_24_04 = 15


@pytest.fixture
def toy_df():
    return pd.DataFrame(
        {
            "Retrieval Method": ["bge-m3", "bge-m3", "jina-embeddings-v2-base", "jina-embeddings-v2-base"],
            "Reranking Model": ["bge-reranker-v2-m3", "NoReranker", "bge-reranker-v2-m3", "NoReranker"],
            "Rank 🏆": [1, 2, 3, 4],
            "Revision": ["123", "234", "345", "456"],
            "Submission Date": ["", "", "", ""],
            "Average ⬆️": [0.6, 0.4, 0.3, 0.2],
            "wiki_en": [0.8, 0.7, 0.2, 0.1],
            "wiki_zh": [0.4, 0.1, 0.4, 0.3],
            "news_en": [0.8, 0.7, 0.2, 0.1],
            "news_zh": [0.4, 0.1, 0.2, 0.3],
            "Anonymous Submission": [False, False, False, True],
        }
    )


def test_remove_html():
    model_name = "jina-embeddings-v3"
    html_str = model_hyperlink("https://jina.ai", model_name)
    output_str = remove_html(html_str)
    assert output_str == model_name


def test_calculate_mean():
    valid_row = [1, 3]
    invalid_row = [2, pd.NA]
    df = pd.DataFrame([valid_row, invalid_row], columns=["a", "b"])
    result = list(df.apply(calculate_mean, axis=1))
    assert result[0] == sum(valid_row) / 2
    assert result[1] == -1


@pytest.mark.parametrize(
    "models, expected",
    [
        (["model1", "model3"], 2),
        (["model1", "model_missing"], 1),
        (["model1", "model2", "model3"], 3),
        (
            [
                "model1",
            ],
            1,
        ),
        ([], 3),
    ],
)
def test_filter_models(models, expected):
    df = pd.DataFrame(
        {
            COL_NAME_RERANKING_MODEL: [
                "model1",
                "model2",
                "model3",
            ],
            "col2": [1, 2, 3],
        }
    )
    output_df = filter_models(df, models)
    assert len(output_df) == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ("model1;model3", 2),
        ("model1;model4", 1),
        ("model1;model2;model3", 3),
        ("model1", 1),
        ("", 3),
    ],
)
def test_filter_queries(query, expected):
    df = pd.DataFrame(
        {
            COL_NAME_RETRIEVAL_MODEL: [
                "model1",
                "model2",
                "model3",
            ],
            COL_NAME_RERANKING_MODEL: [
                "model4",
                "model5",
                "model6",
            ],
        }
    )
    output_df = filter_queries(query, df)
    assert len(output_df) == expected


@pytest.mark.parametrize(
    "task_type, slug, add_fix_cols, expected",
    [
        (TaskType.qa, "air_bench_2404", True, NUM_QA_BENCHMARKS_24_04),
        (TaskType.long_doc, "air_bench_2404", True, NUM_DOC_BENCHMARKS_24_04),
        (TaskType.qa, "air_bench_2405", False, NUM_QA_BENCHMARKS_24_05),
        (TaskType.long_doc, "air_bench_2405", False, NUM_DOC_BENCHMARKS_24_05),
    ],
)
def test_get_default_cols(task_type, slug, add_fix_cols, expected):
    attr_cols = ["Rank 🏆", "Retrieval Method", "Reranking Model", "Revision", "Submission Date", "Average ⬆️"]
    cols, types = get_default_cols(task_type, slug)
    cols_set = frozenset(cols)
    attrs_set = frozenset(attr_cols)
    if add_fix_cols:
        assert attrs_set.issubset(cols_set)
    benchmark_cols = list(cols_set.difference(attrs_set))
    assert len(benchmark_cols) == expected


@pytest.mark.parametrize(
    "task_type, domains, languages, expected",
    [
        (
            TaskType.qa,
            ["wiki", "news"],
            [
                "zh",
            ],
            ["wiki_zh", "news_zh"],
        ),
        (
            TaskType.qa,
            [
                "law",
            ],
            ["zh", "en"],
            ["law_en"],
        ),
        (
            TaskType.long_doc,
            ["healthcare"],
            ["zh", "en"],
            [
                "healthcare_en_pubmed_100k_200k_1",
                "healthcare_en_pubmed_100k_200k_2",
                "healthcare_en_pubmed_100k_200k_3",
                "healthcare_en_pubmed_40k_50k_5_merged",
                "healthcare_en_pubmed_30k_40k_10_merged",
            ],
        ),
    ],
)
def test_get_selected_cols(task_type, domains, languages, expected):
    slug = "air_bench_2404"
    cols = get_selected_cols(task_type, slug, domains, languages)
    assert sorted(cols) == sorted(expected)


@pytest.mark.parametrize("reset_rank", [False])
def test_select_columns(toy_df, reset_rank):
    expected = [
        "Rank 🏆",
        "Retrieval Method",
        "Reranking Model",
        "Revision",
        "Submission Date",
        "Average ⬆️",
        "news_zh",
    ]
    df_result = select_columns(toy_df, ["news"], ["zh"], version_slug="air_bench_2404", reset_ranking=reset_rank)
    assert len(df_result.columns) == len(expected)
    if reset_rank:
        assert df_result["Average ⬆️"].equals(df_result["news_zh"])
    else:
        assert df_result["Average ⬆️"].equals(toy_df["Average ⬆️"])


@pytest.mark.parametrize(
    "reset_rank, show_anony",
    [
        (False, True),
        (True, True),
        (True, False),
    ],
)
def test__update_df_elem(toy_df, reset_rank, show_anony):
    df = _update_df_elem(TaskType.qa, "AIR-Bench_24.04", toy_df, ["news"], ["zh"], [], "", show_anony, reset_rank)
    if show_anony:
        assert df.shape[0] == 4
    else:
        assert df.shape[0] == 3
    if show_anony:
        if reset_rank:
            assert df["Average ⬆️"].equals(df["news_zh"])
        else:
            assert df["Average ⬆️"].equals(toy_df["Average ⬆️"])


@pytest.mark.parametrize(
    "version, task_type",
    [
        ("AIR-Bench_24.04", TaskType.qa),
        ("AIR-Bench_24.04", TaskType.long_doc),
        ("AIR-Bench_24.05", TaskType.qa),
        ("AIR-Bench_24.05", TaskType.long_doc),
    ],
)
def test_get_leaderboard_df(version, task_type):
    from src.loaders import load_raw_eval_results
    from src.models import LeaderboardDataStore, get_safe_name

    raw_data = load_raw_eval_results(cur_fp.parents[1] / f"toydata/eval_results/{version}")
    ds = LeaderboardDataStore(version, get_safe_name(version), raw_data=raw_data)
    df = get_leaderboard_df(ds, task_type, "ndcg_at_10")
    assert df.shape[0] == 1
