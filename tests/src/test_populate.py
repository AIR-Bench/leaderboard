from src.populate import get_leaderboard_df
from pathlib import Path

cur_fp = Path(__file__)


def test_get_leaderboard_df():
    requests_path = cur_fp.parents[1] / "toydata" / "test_requests"
    results_path = cur_fp.parents[1] / "toydata" / "test_results"
    cols = ['Retrieval Model', 'Reranking Model', 'Average ⬆️', 'wiki_en', 'wiki_zh',]
    benchmark_cols = ['wiki_en', 'wiki_zh',]
    raw_data, df = get_leaderboard_df(results_path, requests_path, cols, benchmark_cols)
    assert df.shape[0] == 2
    # the results contains only one embedding model
    for i in range(2):
        assert df["Retrieval Model"][i] == "bge-m3"
    # the results contains only two reranking model
    assert df["Reranking Model"][0] == "bge-reranker-v2-m3"
    assert df["Reranking Model"][1] == "NoReranker"
    assert df["Average ⬆️"][0] > df["Average ⬆️"][1]
    assert not df[['Average ⬆️', 'wiki_en', 'wiki_zh',]].isnull().values.any()


