import json
from typing import List
import os
from datetime import datetime
from pathlib import Path

import pytz

import pandas as pd

from src.benchmarks import BENCHMARK_COLS_QA, BENCHMARK_COLS_LONG_DOC, BenchmarksQA, BenchmarksLongDoc
from src.display.utils import AutoEvalColumnQA, AutoEvalColumnLongDoc, COLS_QA, COLS_LONG_DOC, COL_NAME_RANK, COL_NAME_AVG, COL_NAME_RERANKING_MODEL, COL_NAME_RETRIEVAL_MODEL
from src.leaderboard.read_evals import FullEvalResult, get_leaderboard_df
from src.envs import API, SEARCH_RESULTS_REPO, CACHE_PATH


def filter_models(df: pd.DataFrame, reranking_query: list) -> pd.DataFrame:
    return df.loc[df["Reranking Model"].isin(reranking_query)]


def filter_queries(query: str, filtered_df: pd.DataFrame) -> pd.DataFrame:
    final_df = []
    if query != "":
        queries = [q.strip() for q in query.split(";")]
        for _q in queries:
            _q = _q.strip()
            if _q != "":
                temp_filtered_df = search_table(filtered_df, _q)
                if len(temp_filtered_df) > 0:
                    final_df.append(temp_filtered_df)
        if len(final_df) > 0:
            filtered_df = pd.concat(final_df)
            filtered_df = filtered_df.drop_duplicates(
                subset=[
                    AutoEvalColumnQA.retrieval_model.name,
                    AutoEvalColumnQA.reranking_model.name,
                ]
            )

    return filtered_df


def search_table(df: pd.DataFrame, query: str) -> pd.DataFrame:
    return df[(df[AutoEvalColumnQA.retrieval_model.name].str.contains(query, case=False))]


def get_default_cols(task: str, columns: list, add_fix_cols: bool=True) -> list:
    if task == "qa":
        cols = list(frozenset(COLS_QA).intersection(frozenset(BENCHMARK_COLS_QA)).intersection(frozenset(columns)))
    elif task == "long-doc":
        cols = list(frozenset(COLS_LONG_DOC).intersection(frozenset(BENCHMARK_COLS_LONG_DOC)).intersection(frozenset(columns)))
    else:
        raise NotImplemented
    if add_fix_cols:
        cols = FIXED_COLS + cols
    return cols

FIXED_COLS = [
        COL_NAME_RANK,
        COL_NAME_RETRIEVAL_MODEL,
        COL_NAME_RERANKING_MODEL,
        COL_NAME_AVG,
    ]

def select_columns(df: pd.DataFrame, domain_query: list, language_query: list, task: str = "qa") -> pd.DataFrame:
    cols = get_default_cols(task=task, columns=df.columns, add_fix_cols=False)
    selected_cols = []
    for c in cols:
        if task == "qa":
            eval_col = BenchmarksQA[c].value
        elif task == "long-doc":
            eval_col = BenchmarksLongDoc[c].value
        if eval_col.domain not in domain_query:
            continue
        if eval_col.lang not in language_query:
            continue
        selected_cols.append(c)
    # We use COLS to maintain sorting
    filtered_df = df[FIXED_COLS + selected_cols]
    filtered_df[COL_NAME_AVG] = filtered_df[selected_cols].mean(axis=1).round(decimals=2)
    filtered_df.sort_values(by=[COL_NAME_AVG], ascending=False, inplace=True)
    filtered_df.reset_index(inplace=True, drop=True)
    filtered_df[COL_NAME_RANK] = filtered_df[COL_NAME_AVG].rank(ascending=False, method="min")

    return filtered_df


def update_table(
        hidden_df: pd.DataFrame,
        domains: list,
        langs: list,
        reranking_query: list,
        query: str,
):
    filtered_df = filter_models(hidden_df, reranking_query)
    filtered_df = filter_queries(query, filtered_df)
    df = select_columns(filtered_df, domains, langs)
    return df


def update_table_long_doc(
        hidden_df: pd.DataFrame,
        domains: list,
        langs: list,
        reranking_query: list,
        query: str,
):
    filtered_df = filter_models(hidden_df, reranking_query)
    filtered_df = filter_queries(query, filtered_df)
    df = select_columns(filtered_df, domains, langs, task='long_doc')
    return df


def update_metric(
        raw_data: List[FullEvalResult],
        task: str,
        metric: str,
        domains: list,
        langs: list,
        reranking_model: list,
        query: str,
) -> pd.DataFrame:
    if task == 'qa':
        leaderboard_df = get_leaderboard_df(raw_data, task=task, metric=metric)
        return update_table(
            leaderboard_df,
            domains,
            langs,
            reranking_model,
            query
        )
    elif task == "long-doc":
        leaderboard_df = get_leaderboard_df(raw_data, task=task, metric=metric)
        return update_table_long_doc(
            leaderboard_df,
            domains,
            langs,
            reranking_model,
            query
        )


def upload_file(
        filepath: str, model: str, model_url: str, version: str="AIR-Bench_24.04"):
    print(f"file uploaded: {filepath}")
    # model = "bge-small-en-v1.5"
    # version = "AIR-Bench_24.04"
    if not filepath.endswith(".zip"):
        print(f"file uploading aborted. wrong file type: {filepath}")
        return filepath

    # rename the uploaded file
    input_fp = Path(filepath)
    timezone = pytz.timezone('UTC')
    timestamp = datetime.now(timezone).strftime('%Y%m%d%H%M%S')
    output_fn = f"{timestamp}-{input_fp.name}"
    input_folder_path = input_fp.parent
    API.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=f"{version}/{model}/{output_fn}",
        repo_id=SEARCH_RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"feat: submit {model} to evaluate")

    output_config_fn = f"{output_fn.removesuffix('.zip')}.json"
    output_config = {
        "model_name": f"{model}",
        "model_url": f"{model_url}",
        "version": f"{version}"
    }
    with open(input_folder_path / output_config_fn, "w") as f:
        json.dump(output_config, f, ensure_ascii=False)
    API.upload_file(
        path_or_fileobj=input_folder_path / output_config_fn,
        path_in_repo= f"{version}/{model}/{output_config_fn}",
        repo_id=SEARCH_RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"feat: submit {model} config")
    return filepath
