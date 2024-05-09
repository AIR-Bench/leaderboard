from src.populate import get_leaderboard_df
from pathlib import Path

cur_fp = Path(__file__)

def test_get_leaderboard_df():
    requests_path = cur_fp.parents[2] / "toydata" / "test_requests"
    results_path = cur_fp.parents[2] / "toydata" / "test_results"
    cols = []
    benchmark_cols = []
    COLS = [c.name for c in fields(AutoEvalColumn) if not c.hidden]
    get_leaderboard_df(results_path, requests_path, cols, benchmark_cols)