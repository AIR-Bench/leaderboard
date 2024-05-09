from pathlib import Path

from src.leaderboard.read_evals import FullEvalResult, get_raw_eval_results, get_request_file_for_model

cur_fp = Path(__file__)


def test_init_from_json_file():
    json_fp = cur_fp.parents[2] / "toydata" / "test_data.json"
    full_eval_result = FullEvalResult.init_from_json_file(json_fp)
    assert len(full_eval_result.results) == 6


def test_to_dict():
    json_fp = cur_fp.parents[2] / "toydata" / "test_data.json"
    full_eval_result = FullEvalResult.init_from_json_file(json_fp)
    result_dict = full_eval_result.to_dict(task='qa', metric='ndcg_at_1')
    assert len(result_dict) == 2


def test_get_request_file_for_model():
    requests_path = cur_fp.parents[2] / "toydata" / "test_requests"
    request_file = get_request_file_for_model(requests_path, "bge-m3", "bge-reranker-v2-m3")
    # only load the latest finished results
    assert Path(request_file).name.removeprefix("eval_request_").removesuffix(".json") == "2023-11-21T18-10-08"


def test_get_raw_eval_results():
    requests_path = cur_fp.parents[2] / "toydata" / "test_requests"
    results_path = cur_fp.parents[2] / "toydata" / "test_results" / "bge-m3"
    results = get_raw_eval_results(results_path, requests_path)
    # only load the latest results
    assert len(results) == 2
    assert results[0].date == "2023-12-21T18:10:08"
    assert results[0].eval_name == "bge-m3_NoReranker"
    assert len(results[0].results) == 3
    assert results[1].eval_name == "bge-m3_bge-reranker-v2-m3"
    assert results[1].date == "2023-11-21T18:10:08"
    assert len(results[1].results) == 6
