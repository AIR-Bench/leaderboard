from pathlib import Path

from src.leaderboard.read_evals import FullEvalResult, get_raw_eval_results

cur_fp = Path(__file__)


def test_init_from_json_file():
    json_fp = cur_fp.parents[2] / "toydata" / "test_data.json"
    full_eval_result = FullEvalResult.init_from_json_file(json_fp)
    assert len(full_eval_result.results) == 6


def test_to_dict():
    json_fp = cur_fp.parents[2] / "toydata" / "test_data.json"
    full_eval_result = FullEvalResult.init_from_json_file(json_fp)
    result_list = full_eval_result.to_dict(task='qa', metric='ndcg_at_1')
    assert len(result_list) == 1
    result_dict = result_list[0]
    assert result_dict["Retrieval Model"] == "bge-m3"
    assert result_dict["Reranking Model"] == "bge-reranker-v2-m3"
    assert result_dict["wiki_en"] is not None
    assert result_dict["wiki_zh"] is not None


def test_get_raw_eval_results():
    results_path = cur_fp.parents[2] / "toydata" / "test_results" / "bge-m3"
    results = get_raw_eval_results(results_path)
    # only load the latest results
    assert len(results) == 2
    assert results[0].eval_name == "bge-m3_NoReranker"
    assert len(results[0].results) == 6
    assert results[1].eval_name == "bge-m3_bge-reranker-v2-m3"
    assert len(results[1].results) == 6
