from pathlib import Path

import pandas as pd
import pytest

from src.loaders import load_eval_results, load_leaderboard_datastore, load_raw_eval_results

cur_fp = Path(__file__)


@pytest.mark.parametrize("version", ["AIR-Bench_24.04", "AIR-Bench_24.05"])
def test_load_raw_eval_results(version):
    raw_data = load_raw_eval_results(cur_fp.parents[1] / f"toydata/eval_results/{version}")
    assert len(raw_data) == 1
    full_eval_result = raw_data[0]
    expected_attr = [
        "eval_name",
        "retrieval_model",
        "reranking_model",
        "retrieval_model_link",
        "reranking_model_link",
        "results",
        "timestamp",
        "revision",
        "is_anonymous",
    ]
    result_attr = [k for k in full_eval_result.__dict__.keys() if k[:2] != "__" and k[-2:] != "__"]
    assert sorted(expected_attr) == sorted(result_attr)


@pytest.mark.parametrize("version", ["AIR-Bench_24.04", "AIR-Bench_24.05"])
def test_load_leaderboard_datastore(version):
    file_path = cur_fp.parents[1] / f"toydata/eval_results/{version}"
    datastore = load_leaderboard_datastore(file_path, version)
    for k, v in datastore.__dict__.items():
        if k[:2] != "__" and k[-2:] != "__":
            if isinstance(v, list):
                assert v
            elif isinstance(v, pd.DataFrame):
                assert not v.empty


def test_load_eval_results():
    file_path = cur_fp.parents[1] / "toydata/eval_results/"
    datastore_dict = load_eval_results(file_path)
    assert len(datastore_dict) == 2
