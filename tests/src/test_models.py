from pathlib import Path

import pytest

from src.models import EvalResult, FullEvalResult

cur_fp = Path(__file__)


# Ref: https://github.com/AIR-Bench/AIR-Bench/blob/4b27b8a8f2047a963805fcf6fb9d74be51ec440c/docs/available_tasks.md
# 24.05
# | Task | dev | test |
# | ---- | --- | ---- |
# | Long-Doc | 4 | 11 |
# | QA | 54 | 53 |
#
# 24.04
# | Task | test |
# | ---- | ---- |
# | Long-Doc | 15 |
# | QA | 13 |
NUM_QA_BENCHMARKS_24_05 = 53
NUM_DOC_BENCHMARKS_24_05 = 11
NUM_QA_BENCHMARKS_24_04 = 13
NUM_DOC_BENCHMARKS_24_04 = 15


def test_eval_result():
    EvalResult(
        eval_name="eval_name",
        retrieval_model="bge-m3",
        reranking_model="NoReranking",
        results=[{"domain": "law", "lang": "en", "dataset": "lex_files_500K-600K", "value": 0.45723}],
        task="qa",
        metric="ndcg_at_3",
        timestamp="2024-05-14T03:09:08Z",
        revision="1e243f14bd295ccdea7a118fe847399d",
        is_anonymous=True,
    )


@pytest.mark.parametrize(
    "file_path",
    [
        "AIR-Bench_24.04/bge-m3/jina-reranker-v2-base-multilingual/results.json",
        "AIR-Bench_24.05/bge-m3/NoReranker/results.json",
    ],
)
def test_full_eval_result_init_from_json_file(file_path):
    json_fp = cur_fp.parents[1] / "toydata/eval_results/" / file_path
    full_eval_result = FullEvalResult.init_from_json_file(json_fp)
    assert json_fp.parents[0].stem == full_eval_result.reranking_model
    assert json_fp.parents[1].stem == full_eval_result.retrieval_model
    assert len(full_eval_result.results) == 70


@pytest.mark.parametrize(
    "file_path, task, expected_num_results",
    [
        ("AIR-Bench_24.04/bge-m3/jina-reranker-v2-base-multilingual/results.json", "qa", NUM_QA_BENCHMARKS_24_04),
        (
            "AIR-Bench_24.04/bge-m3/jina-reranker-v2-base-multilingual/results.json",
            "long-doc",
            NUM_DOC_BENCHMARKS_24_04,
        ),
        ("AIR-Bench_24.05/bge-m3/NoReranker/results.json", "qa", NUM_QA_BENCHMARKS_24_05),
        ("AIR-Bench_24.05/bge-m3/NoReranker/results.json", "long-doc", NUM_DOC_BENCHMARKS_24_05),
    ],
)
def test_full_eval_result_to_dict(file_path, task, expected_num_results):
    json_fp = cur_fp.parents[1] / "toydata/eval_results/" / file_path
    full_eval_result = FullEvalResult.init_from_json_file(json_fp)
    result_dict_list = full_eval_result.to_dict(task)
    assert len(result_dict_list) == 1
    result = result_dict_list[0]
    attr_list = frozenset(
        [
            "eval_name",
            "Retrieval Method",
            "Reranking Model",
            "Retrieval Model LINK",
            "Reranking Model LINK",
            "Revision",
            "Submission Date",
            "Anonymous Submission",
        ]
    )
    result_cols = list(result.keys())
    assert len(result_cols) == (expected_num_results + len(attr_list))
