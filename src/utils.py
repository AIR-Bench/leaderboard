import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from src.benchmarks import BENCHMARK_COLS_QA, BENCHMARK_COLS_LONG_DOC, BenchmarksQA, BenchmarksLongDoc
from src.display.formatting import styled_message, styled_error
from src.display.utils import COLS_QA, TYPES_QA, COLS_LONG_DOC, TYPES_LONG_DOC, COL_NAME_RANK, COL_NAME_AVG, \
    COL_NAME_RERANKING_MODEL, COL_NAME_RETRIEVAL_MODEL, COL_NAME_IS_ANONYMOUS, COL_NAME_TIMESTAMP, COL_NAME_REVISION, get_default_auto_eval_column_dict
from src.envs import API, SEARCH_RESULTS_REPO, LATEST_BENCHMARK_VERSION
from src.read_evals import FullEvalResult, get_leaderboard_df, calculate_mean

import re


def remove_html(input_str):
    # Regular expression for finding HTML tags
    clean = re.sub(r'<.*?>', '', input_str)
    return clean


def filter_models(df: pd.DataFrame, reranking_query: list) -> pd.DataFrame:
    if not reranking_query:
        return df
    else:
        return df.loc[df[COL_NAME_RERANKING_MODEL].apply(remove_html).isin(reranking_query)]


def filter_queries(query: str, df: pd.DataFrame) -> pd.DataFrame:
    filtered_df = df.copy()
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
                    COL_NAME_RETRIEVAL_MODEL,
                    COL_NAME_RERANKING_MODEL,
                ]
            )
    return filtered_df


def search_table(df: pd.DataFrame, query: str) -> pd.DataFrame:
    return df[(df[COL_NAME_RETRIEVAL_MODEL].str.contains(query, case=False))]


def get_default_cols(task: str, columns: list=[], add_fix_cols: bool=True) -> list:
    cols = []
    types = []
    if task == "qa":
        cols_list = COLS_QA
        types_list = TYPES_QA
        benchmark_list = BENCHMARK_COLS_QA
    elif task == "long-doc":
        cols_list = COLS_LONG_DOC
        types_list = TYPES_LONG_DOC
        benchmark_list = BENCHMARK_COLS_LONG_DOC
    else:
        raise NotImplemented
    for col_name, col_type in zip(cols_list, types_list):
        if col_name not in benchmark_list:
            continue
        if len(columns) > 0 and col_name not in columns:
            continue
        cols.append(col_name)
        types.append(col_type)

    if add_fix_cols:
        _cols = []
        _types = []
        for col_name, col_type in zip(cols, types):
            if col_name in FIXED_COLS:
                continue
            _cols.append(col_name)
            _types.append(col_type)
        cols = FIXED_COLS + _cols
        types = FIXED_COLS_TYPES + _types
    return cols, types


fixed_cols = get_default_auto_eval_column_dict()[:-3]

FIXED_COLS = [c.name for _, _, c in fixed_cols]
FIXED_COLS_TYPES = [c.type for _, _, c in fixed_cols]


def select_columns(
        df: pd.DataFrame,
        domain_query: list,
        language_query: list,
        task: str = "qa",
        reset_ranking: bool = True
) -> pd.DataFrame:
    cols, _ = get_default_cols(task=task, columns=df.columns, add_fix_cols=False)
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
    if reset_ranking:
        filtered_df[COL_NAME_AVG] = filtered_df[selected_cols].apply(calculate_mean, axis=1).round(decimals=2)
        filtered_df.sort_values(by=[COL_NAME_AVG], ascending=False, inplace=True)
        filtered_df.reset_index(inplace=True, drop=True)
        filtered_df = reset_rank(filtered_df)

    return filtered_df


def _update_table(
        task: str,
        hidden_df: pd.DataFrame,
        domains: list,
        langs: list,
        reranking_query: list,
        query: str,
        show_anonymous: bool,
        reset_ranking: bool = True,
        show_revision_and_timestamp: bool = False
):
    filtered_df = hidden_df.copy()
    if not show_anonymous:
        filtered_df = filtered_df[~filtered_df[COL_NAME_IS_ANONYMOUS]]
    filtered_df = filter_models(filtered_df, reranking_query)
    filtered_df = filter_queries(query, filtered_df)
    filtered_df = select_columns(filtered_df, domains, langs, task, reset_ranking)
    if not show_revision_and_timestamp:
        filtered_df.drop([COL_NAME_REVISION, COL_NAME_TIMESTAMP], axis=1, inplace=True)
    return filtered_df


def update_table(
        hidden_df: pd.DataFrame,
        domains: list,
        langs: list,
        reranking_query: list,
        query: str,
        show_anonymous: bool,
        show_revision_and_timestamp: bool = False,
        reset_ranking: bool = True
):
    return _update_table(
        "qa", hidden_df, domains, langs, reranking_query, query, show_anonymous, reset_ranking, show_revision_and_timestamp)


def update_table_long_doc(
        hidden_df: pd.DataFrame,
        domains: list,
        langs: list,
        reranking_query: list,
        query: str,
        show_anonymous: bool,
        show_revision_and_timestamp: bool = False,
        reset_ranking: bool = True

):
    return _update_table(
        "long-doc", hidden_df, domains, langs, reranking_query, query, show_anonymous, reset_ranking, show_revision_and_timestamp)


def update_metric(
        raw_data: List[FullEvalResult],
        task: str,
        metric: str,
        domains: list,
        langs: list,
        reranking_model: list,
        query: str,
        show_anonymous: bool = False,
        show_revision_and_timestamp: bool = False,
) -> pd.DataFrame:
    if task == 'qa':
        leaderboard_df = get_leaderboard_df(raw_data, task=task, metric=metric)
        return update_table(
            leaderboard_df,
            domains,
            langs,
            reranking_model,
            query,
            show_anonymous,
            show_revision_and_timestamp
        )
    elif task == "long-doc":
        leaderboard_df = get_leaderboard_df(raw_data, task=task, metric=metric)
        return update_table_long_doc(
            leaderboard_df,
            domains,
            langs,
            reranking_model,
            query,
            show_anonymous,
            show_revision_and_timestamp
        )


def upload_file(filepath: str):
    if not filepath.endswith(".zip"):
        print(f"file uploading aborted. wrong file type: {filepath}")
        return filepath
    return filepath



def get_iso_format_timestamp():
    # Get the current timestamp with UTC as the timezone
    current_timestamp = datetime.now(timezone.utc)

    # Remove milliseconds by setting microseconds to zero
    current_timestamp = current_timestamp.replace(microsecond=0)

    # Convert to ISO 8601 format and replace the offset with 'Z'
    iso_format_timestamp = current_timestamp.isoformat().replace('+00:00', 'Z')
    filename_friendly_timestamp = current_timestamp.strftime('%Y%m%d%H%M%S')
    return iso_format_timestamp, filename_friendly_timestamp


def calculate_file_md5(file_path):
    md5 = hashlib.md5()

    with open(file_path, 'rb') as f:
        while True:
            data = f.read(4096)
            if not data:
                break
            md5.update(data)

    return md5.hexdigest()


def submit_results(
        filepath: str,
        model: str,
        model_url: str,
        reranking_model: str="",
        reranking_model_url: str="",
        version: str=LATEST_BENCHMARK_VERSION,
        is_anonymous=False):
    if not filepath.endswith(".zip"):
        return styled_error(f"file uploading aborted. wrong file type: {filepath}")

    # validate model
    if not model:
        return styled_error("failed to submit. Model name can not be empty.")

    # validate model url
    if not is_anonymous:
        if not model_url.startswith("https://") and not model_url.startswith("http://"):
            # TODO: retrieve the model page and find the model name on the page
            return styled_error(
                f"failed to submit. Model url must start with `https://` or `http://`. Illegal model url: {model_url}")
        if reranking_model != "NoReranker":
            if not reranking_model_url.startswith("https://") and not reranking_model_url.startswith("http://"):
                return styled_error(
                    f"failed to submit. Model url must start with `https://` or `http://`. Illegal model url: {model_url}")

    # rename the uploaded file
    input_fp = Path(filepath)
    revision = calculate_file_md5(filepath)
    timestamp_config, timestamp_fn = get_iso_format_timestamp()
    output_fn = f"{timestamp_fn}-{revision}.zip"
    input_folder_path = input_fp.parent

    if not reranking_model:
        reranking_model = 'NoReranker'
    
    API.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=f"{version}/{model}/{reranking_model}/{output_fn}",
        repo_id=SEARCH_RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"feat: submit {model} to evaluate")

    output_config_fn = f"{output_fn.removesuffix('.zip')}.json"
    output_config = {
        "model_name": f"{model}",
        "model_url": f"{model_url}",
        "reranker_name": f"{reranking_model}",
        "reranker_url": f"{reranking_model_url}",
        "version": f"{version}",
        "is_anonymous": is_anonymous,
        "revision": f"{revision}",
        "timestamp": f"{timestamp_config}"
    }
    with open(input_folder_path / output_config_fn, "w") as f:
        json.dump(output_config, f, indent=4, ensure_ascii=False)
    API.upload_file(
        path_or_fileobj=input_folder_path / output_config_fn,
        path_in_repo=f"{version}/{model}/{reranking_model}/{output_config_fn}",
        repo_id=SEARCH_RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"feat: submit {model} + {reranking_model} config")
    return styled_message(
        f"Thanks for submission!\n"
        f"Retrieval method: {model}\nReranking model: {reranking_model}\nSubmission revision: {revision}"
    )


def reset_rank(df):
    df[COL_NAME_RANK] = df[COL_NAME_AVG].rank(ascending=False, method="min")
    return df
