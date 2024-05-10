import json
import os

import pandas as pd

from src.display.formatting import has_no_nan_values, make_clickable_model
from src.display.utils import AutoEvalColumnQA, AutoEvalColumnLongDoc, EvalQueueColumn
from src.leaderboard.read_evals import get_raw_eval_results, EvalResult, FullEvalResult
from typing import Tuple, List


def get_leaderboard_df(raw_data: List[FullEvalResult], cols: list, benchmark_cols: list, task: str, metric: str) -> pd.DataFrame:
    """Creates a dataframe from all the individual experiment results"""
    all_data_json = []
    for v in raw_data:
        all_data_json += v.to_dict(task=task, metric=metric)
    df = pd.DataFrame.from_records(all_data_json)
    print(f'dataframe created: {df.shape}')

    # calculate the average score for selected benchmarks
    _benchmark_cols = frozenset(benchmark_cols).intersection(frozenset(df.columns.to_list()))
    if task == 'qa':
        df[AutoEvalColumnQA.average.name] = df[list(_benchmark_cols)].mean(axis=1).round(decimals=2)
        df = df.sort_values(by=[AutoEvalColumnQA.average.name], ascending=False)
    elif task == "long_doc":
        df[AutoEvalColumnLongDoc.average.name] = df[list(_benchmark_cols)].mean(axis=1).round(decimals=2)
        df = df.sort_values(by=[AutoEvalColumnLongDoc.average.name], ascending=False)

    df.reset_index(inplace=True)

    _cols = frozenset(cols).intersection(frozenset(df.columns.to_list()))
    df = df[_cols].round(decimals=2)

    # filter out if any of the benchmarks have not been produced
    df = df[has_no_nan_values(df, _benchmark_cols)]
    return df


def get_evaluation_queue_df(save_path: str, cols: list) -> list[pd.DataFrame]:
    """Creates the different dataframes for the evaluation queues requests"""
    # entries = [entry for entry in os.listdir(save_path) if not entry.startswith(".")]
    # all_evals = []
    #
    # for entry in entries:
    #     if ".json" in entry:
    #         file_path = os.path.join(save_path, entry)
    #         with open(file_path) as fp:
    #             data = json.load(fp)
    #
    #         data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
    #         data[EvalQueueColumn.revision.name] = data.get("revision", "main")
    #
    #         all_evals.append(data)
    #     elif ".md" not in entry:
    #         # this is a folder
    #         sub_entries = [e for e in os.listdir(f"{save_path}/{entry}") if not e.startswith(".")]
    #         for sub_entry in sub_entries:
    #             file_path = os.path.join(save_path, entry, sub_entry)
    #             with open(file_path) as fp:
    #                 data = json.load(fp)
    #
    #             data[EvalQueueColumn.model.name] = make_clickable_model(data["model"])
    #             data[EvalQueueColumn.revision.name] = data.get("revision", "main")
    #             all_evals.append(data)
    #
    # pending_list = [e for e in all_evals if e["status"] in ["PENDING", "RERUN"]]
    # running_list = [e for e in all_evals if e["status"] == "RUNNING"]
    # finished_list = [e for e in all_evals if e["status"].startswith("FINISHED") or e["status"] == "PENDING_NEW_EVAL"]
    # df_pending = pd.DataFrame.from_records(pending_list, columns=cols)
    # df_running = pd.DataFrame.from_records(running_list, columns=cols)
    # df_finished = pd.DataFrame.from_records(finished_list, columns=cols)
    cols = ["Retrieval Model", "Submitted Time", "Status"]
    df_finished = pd.DataFrame(
        {
            "Retrieval Model": ["bge-m3", "jina-embeddings-v2"],
            "Submitted Time": ["2024-05-01 12:34:20", "2024-05-02 12:34:20"],
            "Status": ["FINISHED", "FINISHED"]
        }
    )
    df_running = pd.DataFrame(
        {
            "Retrieval Model": ["bge-m3", "jina-embeddings-v2"],
            "Submitted Time": ["2024-05-01 12:34:20", "2024-05-02 12:34:20"],
            "Status": ["RUNNING", "RUNNING"]
        }
    )
    df_pending = pd.DataFrame(
        {
            "Retrieval Model": ["bge-m3", "jina-embeddings-v2"],
            "Submitted Time": ["2024-05-01 12:34:20", "2024-05-02 12:34:20"],
            "Status": ["PENDING", "PENDING"]
        }
    )
    return df_finished, df_running, df_pending
