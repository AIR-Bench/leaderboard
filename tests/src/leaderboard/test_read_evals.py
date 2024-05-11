from pathlib import Path

from src.leaderboard.read_evals import FullEvalResult, get_raw_eval_results, get_leaderboard_df

cur_fp = Path(__file__)


def test_init_from_json_file():
    json_fp = cur_fp.parents[2] / "toydata" / "test_data.json"
    full_eval_result = FullEvalResult.init_from_json_file(json_fp)
    num_different_task_domain_lang_metric_dataset_combination = 6
    assert len(full_eval_result.results) == \
           num_different_task_domain_lang_metric_dataset_combination
    assert full_eval_result.retrieval_model == "bge-m3"
    assert full_eval_result.reranking_model == "bge-reranker-v2-m3"


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
    results_path = cur_fp.parents[2] / "toydata" / "eval_results" / "AIR-Bench_24.04"
    results = get_raw_eval_results(results_path)
    # only load the latest results
    assert len(results) == 4
    assert results[0].eval_name == "bge-base-en-v1.5_NoReranker"
    assert len(results[0].results) == 70
    assert results[0].eval_name == "bge-base-en-v1.5_bge-reranker-v2-m3"
    assert len(results[1].results) == 70


def test_get_leaderboard_df():
    results_path = cur_fp.parents[2] / "toydata" / "eval_results" / "AIR-Bench_24.04"
    raw_data = get_raw_eval_results(results_path)
    df = get_leaderboard_df(raw_data, 'qa', 'ndcg_at_3')
    assert df.shape[0] == 4
    # the results contain only one embedding model
    # for i in range(4):
    #     assert df["Retrieval Model"][i] == "bge-m3"
    # # the results contain only two reranking model
    # assert df["Reranking Model"][0] == "bge-reranker-v2-m3"
    # assert df["Reranking Model"][1] == "NoReranker"
    # assert df["Average ⬆️"][0] > df["Average ⬆️"][1]
    # assert not df[['Average ⬆️', 'wiki_en', 'wiki_zh', ]].isnull().values.any()


def test_get_leaderboard_df_long_doc():
    results_path = cur_fp.parents[2] / "toydata" / "test_results"
    raw_data = get_raw_eval_results(results_path)
    df = get_leaderboard_df(raw_data, 'long-doc', 'ndcg_at_1')
    assert df.shape[0] == 2
    # the results contain only one embedding model
    for i in range(2):
        assert df["Retrieval Model"][i] == "bge-m3"
    # the results contains only two reranking model
    assert df["Reranking Model"][0] == "bge-reranker-v2-m3"
    assert df["Reranking Model"][1] == "NoReranker"
    assert df["Average ⬆️"][0] > df["Average ⬆️"][1]
    assert not df[['Average ⬆️', 'law_en_lex_files_500k_600k', ]].isnull().values.any()
