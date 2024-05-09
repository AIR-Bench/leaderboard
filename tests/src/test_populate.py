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
    assert df["Retrieval Model"][0] == "bge-m3"
    assert df["Retrieval Model"][1] == "bge-m3"
    assert df["Reranking Model"][0] == "NoReranker"
    assert df["Reranking Model"][1] == "bge-reranker-v2-m3"
    assert not df[['Average ⬆️', 'wiki_en', 'wiki_zh',]].isnull().values.any()


