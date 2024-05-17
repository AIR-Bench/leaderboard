import pandas as pd
import pytest

from src.utils import filter_models, search_table, filter_queries, select_columns, update_table_long_doc, get_iso_format_timestamp, get_default_cols, update_table
from src.display.utils import COL_NAME_IS_ANONYMOUS, COL_NAME_REVISION, COL_NAME_TIMESTAMP, COL_NAME_RERANKING_MODEL, COL_NAME_RETRIEVAL_MODEL, COL_NAME_RANK, COL_NAME_AVG


@pytest.fixture
def toy_df():
    return pd.DataFrame(
        {
            "Retrieval Model": [
                "bge-m3",
                "bge-m3",
                "jina-embeddings-v2-base",
                "jina-embeddings-v2-base"
            ],
            "Reranking Model": [
                "bge-reranker-v2-m3",
                "NoReranker",
                "bge-reranker-v2-m3",
                "NoReranker"
            ],
            "Average ⬆️": [0.6, 0.4, 0.3, 0.2],
            "wiki_en": [0.8, 0.7, 0.2, 0.1],
            "wiki_zh": [0.4, 0.1, 0.4, 0.3],
            "news_en": [0.8, 0.7, 0.2, 0.1],
            "news_zh": [0.4, 0.1, 0.4, 0.3],
        }
    )


@pytest.fixture
def toy_df_long_doc():
    return pd.DataFrame(
        {
            "Retrieval Model": [
                "bge-m3",
                "bge-m3",
                "jina-embeddings-v2-base",
                "jina-embeddings-v2-base"
            ],
            "Reranking Model": [
                "bge-reranker-v2-m3",
                "NoReranker",
                "bge-reranker-v2-m3",
                "NoReranker"
            ],
            "Average ⬆️": [0.6, 0.4, 0.3, 0.2],
            "law_en_lex_files_300k_400k": [0.4, 0.1, 0.4, 0.3],
            "law_en_lex_files_400k_500k": [0.8, 0.7, 0.2, 0.1],
            "law_en_lex_files_500k_600k": [0.8, 0.7, 0.2, 0.1],
            "law_en_lex_files_600k_700k": [0.4, 0.1, 0.4, 0.3],
        }
    )
def test_filter_models(toy_df):
    df_result = filter_models(toy_df, ["bge-reranker-v2-m3", ])
    assert len(df_result) == 2
    assert df_result.iloc[0]["Reranking Model"] == "bge-reranker-v2-m3"


def test_search_table(toy_df):
    df_result = search_table(toy_df, "jina")
    assert len(df_result) == 2
    assert df_result.iloc[0]["Retrieval Model"] == "jina-embeddings-v2-base"


def test_filter_queries(toy_df):
    df_result = filter_queries("jina", toy_df)
    assert len(df_result) == 2
    assert df_result.iloc[0]["Retrieval Model"] == "jina-embeddings-v2-base"


def test_select_columns(toy_df):
    df_result = select_columns(toy_df, ['news',], ['zh',])
    assert len(df_result.columns) == 4
    assert df_result['Average ⬆️'].equals(df_result['news_zh'])


def test_update_table_long_doc(toy_df_long_doc):
    df_result = update_table_long_doc(toy_df_long_doc, ['law',], ['en',], ["bge-reranker-v2-m3", ], "jina")
    print(df_result)


def test_get_iso_format_timestamp():
    timestamp_config, timestamp_fn = get_iso_format_timestamp()
    assert len(timestamp_fn) == 14
    assert len(timestamp_config) == 20
    assert timestamp_config[-1] == "Z"


def test_get_default_cols():
    cols, types = get_default_cols("qa")
    for c, t in zip(cols, types):
        print(f"type({c}): {t}")
    assert len(frozenset(cols)) == len(cols)


def test_update_table():
    df = pd.DataFrame(
        {
            COL_NAME_IS_ANONYMOUS: [False, False, False],
            COL_NAME_REVISION: ["a1", "a2", "a3"],
            COL_NAME_TIMESTAMP: ["2024-05-12T12:24:02Z"] * 3,
            COL_NAME_RERANKING_MODEL: ["NoReranker"] * 3,
            COL_NAME_RETRIEVAL_MODEL: ["Foo"] * 3,
            COL_NAME_RANK: [1, 2, 3],
            COL_NAME_AVG: [0.1, 0.2, 0.3],  # unsorted values
            "wiki_en": [0.1, 0.2, 0.3]
        }
    )
    results = update_table(df, "wiki", "en", ["NoReranker"], "", show_anonymous=False, reset_ranking=False, show_revision_and_timestamp=False)
    # keep the RANK as the same regardless of the unsorted averages
    assert results[COL_NAME_RANK].to_list() == [1, 2, 3]

