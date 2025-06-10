import os.path
from pathlib import Path
from typing import Dict, List, Union

import pandas as pd

from src.columns import COL_NAME_IS_ANONYMOUS, COL_NAME_REVISION, COL_NAME_TIMESTAMP
from src.envs import BENCHMARK_VERSION_LIST, DEFAULT_METRIC_LONG_DOC, DEFAULT_METRIC_QA
from src.models import FullEvalResult, LeaderboardDataStore, TaskType, get_safe_name
from src.utils import get_default_cols, get_leaderboard_df, reset_rank

pd.options.mode.copy_on_write = True


def load_raw_eval_results(results_path: Union[Path, str]) -> List[FullEvalResult]:
    """
    Load the evaluation results from a json file
    """
    model_result_filepaths = []
    for root, dirs, files in os.walk(results_path):
        if len(files) == 0:
            continue

        # select the latest results
        for file in files:
            if not (file.startswith("results") and file.endswith(".json")):
                print(f"skip {file}")
                continue
            model_result_filepaths.append(os.path.join(root, file))

    eval_results = {}
    for model_result_filepath in model_result_filepaths:
        # create evaluation results
        try:
            eval_result = FullEvalResult.init_from_json_file(model_result_filepath)
        except UnicodeDecodeError:
            print(f"loading file failed since UnicodeDecodeError. {model_result_filepath}")
            continue
        except IndexError:
            print(f"loading file failed since IndexError. {model_result_filepath}")
            continue
        print(f"file loaded: {model_result_filepath}")
        timestamp = eval_result.timestamp
        eval_results[timestamp] = eval_result

    results = []
    for k, v in eval_results.items():
        try:
            v.to_dict()
            results.append(v)
        except KeyError:
            print(f"loading failed: {k}")
            continue
    return results


def load_leaderboard_datastore(file_path, version) -> LeaderboardDataStore:
    ds = LeaderboardDataStore(version, get_safe_name(version))
    ds.raw_data = load_raw_eval_results(file_path)
    print(f"raw data: {len(ds.raw_data)}")

    ds.qa_raw_df = get_leaderboard_df(ds, TaskType.qa, DEFAULT_METRIC_QA)
    print(f"QA data loaded: {ds.qa_raw_df.shape}")
    ds.qa_fmt_df = ds.qa_raw_df.copy()
    qa_cols, ds.qa_types = get_default_cols(TaskType.qa, ds.slug, add_fix_cols=True)
    # by default, drop the anonymous submissions
    ds.qa_fmt_df = ds.qa_fmt_df[~ds.qa_fmt_df[COL_NAME_IS_ANONYMOUS]][qa_cols]
    # reset the rank after dropping the anonymous submissions
    ds.qa_fmt_df = reset_rank(ds.qa_fmt_df)
    ds.qa_fmt_df.drop([COL_NAME_REVISION, COL_NAME_TIMESTAMP], axis=1, inplace=True)

    ds.doc_raw_df = get_leaderboard_df(ds, TaskType.long_doc, DEFAULT_METRIC_LONG_DOC)
    print(f"Long-Doc data loaded: {len(ds.doc_raw_df)}")
    ds.doc_fmt_df = ds.doc_raw_df.copy()
    doc_cols, ds.doc_types = get_default_cols(TaskType.long_doc, ds.slug, add_fix_cols=True)
    # by default, drop the anonymous submissions
    ds.doc_fmt_df = ds.doc_fmt_df[~ds.doc_fmt_df[COL_NAME_IS_ANONYMOUS]][doc_cols]
    # reset the rank after dropping the anonymous submissions
    ds.doc_fmt_df = reset_rank(ds.doc_fmt_df)
    ds.doc_fmt_df.drop([COL_NAME_REVISION, COL_NAME_TIMESTAMP], axis=1, inplace=True)

    ds.reranking_models = sorted(list(frozenset([eval_result.reranking_model for eval_result in ds.raw_data])))
    return ds


def load_eval_results(file_path: Union[str, Path]) -> Dict[str, LeaderboardDataStore]:
    output = {}
    for version in BENCHMARK_VERSION_LIST:
        fn = f"{file_path}/{version}"
        output[version] = load_leaderboard_datastore(fn, version)
    return output
