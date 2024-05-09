import pandas as pd
import pytest

from utils import filter_models, search_table, filter_queries, select_columns


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