import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import pandas as pd

from src.benchmarks import BENCHMARK_COLS_QA, BENCHMARK_COLS_LONG_DOC, BenchmarksQA, BenchmarksLongDoc
from src.display.formatting import styled_message, styled_error
from src.display.utils import COLS_QA, TYPES_QA, COLS_LONG_DOC, TYPES_LONG_DOC, COL_NAME_RANK, COL_NAME_AVG, \
    COL_NAME_RERANKING_MODEL, COL_NAME_RETRIEVAL_MODEL, COL_NAME_IS_ANONYMOUS, get_default_auto_eval_column_dict
from src.envs import API, SEARCH_RESULTS_REPO
from src.read_evals import FullEvalResult, get_leaderboard_df


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
                    COL_NAME_RETRIEVAL_MODEL,
                    COL_NAME_RERANKING_MODEL,
                ]
            )

    return filtered_df


def search_table(df: pd.DataFrame, query: str) -> pd.DataFrame:
    return df[(df[COL_NAME_RETRIEVAL_MODEL].str.contains(query, case=False))]


def get_default_cols(task: str, columns: list = [], add_fix_cols: bool = True) -> list:
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
        cols = FIXED_COLS + cols
        types = FIXED_COLS_TYPES + types
    return cols, types


fixed_cols = get_default_auto_eval_column_dict()[:-2]

FIXED_COLS = [c.name for _, _, c in fixed_cols]
FIXED_COLS_TYPES = [c.type for _, _, c in fixed_cols]


def select_columns(df: pd.DataFrame, domain_query: list, language_query: list, task: str = "qa") -> pd.DataFrame:
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
    filtered_df[COL_NAME_AVG] = filtered_df[selected_cols].mean(axis=1, numeric_only=True).round(decimals=2)
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
        show_anonymous: bool
):
    print(f"shown_anonymous: {show_anonymous}")
    filtered_df = hidden_df
    if not show_anonymous:
        print(filtered_df[COL_NAME_IS_ANONYMOUS])
        filtered_df = filtered_df[~filtered_df[COL_NAME_IS_ANONYMOUS]]
        print(f"filtered_df: {len(filtered_df)}")
    filtered_df = filter_models(filtered_df, reranking_query)
    filtered_df = filter_queries(query, filtered_df)
    df = select_columns(filtered_df, domains, langs)
    return df


def update_table_long_doc(
        hidden_df: pd.DataFrame,
        domains: list,
        langs: list,
        reranking_query: list,
        query: str,
        # show_anonymous: bool
):
    filtered_df = filter_models(hidden_df, reranking_query)
    filtered_df = filter_queries(query, filtered_df)
    df = select_columns(filtered_df, domains, langs, task='long_doc')
    # if not show_anonymous:
    #     df = df[~df[COL_NAME_IS_ANONYMOUS]]
    return df


def update_metric(
        raw_data: List[FullEvalResult],
        task: str,
        metric: str,
        domains: list,
        langs: list,
        reranking_model: list,
        query: str,
        show_anonymous: bool
) -> pd.DataFrame:
    if task == 'qa':
        leaderboard_df = get_leaderboard_df(raw_data, task=task, metric=metric)
        return update_table(
            leaderboard_df,
            domains,
            langs,
            reranking_model,
            query,
            show_anonymous
        )
    elif task == "long-doc":
        leaderboard_df = get_leaderboard_df(raw_data, task=task, metric=metric)
        return update_table_long_doc(
            leaderboard_df,
            domains,
            langs,
            reranking_model,
            query,
            # show_anonymous
        )


def upload_file(filepath: str):
    if not filepath.endswith(".zip"):
        print(f"file uploading aborted. wrong file type: {filepath}")
        return filepath
    return filepath


from huggingface_hub import ModelCard
from huggingface_hub.utils import EntryNotFoundError


def get_iso_format_timestamp():
    # Get the current timestamp with UTC as the timezone
    current_timestamp = datetime.now(timezone.utc)

    # Remove milliseconds by setting microseconds to zero
    current_timestamp = current_timestamp.replace(microsecond=0)

    # Convert to ISO 8601 format and replace the offset with 'Z'
    iso_format_timestamp = current_timestamp.isoformat().replace('+00:00', 'Z')
    filename_friendly_timestamp = current_timestamp.strftime('%Y%m%d%H%M%S')
    return iso_format_timestamp, filename_friendly_timestamp


def submit_results(filepath: str, model: str, model_url: str, version: str = "AIR-Bench_24.04", is_anonymous=False):
    if not filepath.endswith(".zip"):
        return styled_error(f"file uploading aborted. wrong file type: {filepath}")

    # validate model
    if not model:
        return styled_error("failed to submit. Model name can not be empty.")

    # validate model url
    if not model_url.startswith("https://huggingface.co/"):
        return styled_error(
            f"failed to submit. Model url must be a link to a valid HuggingFace model on HuggingFace space. Illegal model url: {model_url}")

    # validate model card
    repo_id = model_url.removeprefix("https://huggingface.co/")
    try:
        card = ModelCard.load(repo_id)
    except EntryNotFoundError as e:
        print(e)
        return styled_error(
            f"failed to submit. Model url must be a link to a valid HuggingFace model on HuggingFace space. Could not get model {repo_id}")

    # rename the uploaded file
    input_fp = Path(filepath)
    revision = input_fp.name.removesuffix(".zip")
    timestamp_config, timestamp_fn = get_iso_format_timestamp()
    output_fn = f"{timestamp_fn}-{input_fp.name}"
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
        "version": f"{version}",
        "is_anonymous": f"{is_anonymous}",
        "revision": f"{revision}",
        "timestamp": f"{timestamp_config}"
    }
    with open(input_folder_path / output_config_fn, "w") as f:
        json.dump(output_config, f, ensure_ascii=False)
    API.upload_file(
        path_or_fileobj=input_folder_path / output_config_fn,
        path_in_repo=f"{version}/{model}/{output_config_fn}",
        repo_id=SEARCH_RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"feat: submit {model} config")
    return styled_message(
        f"Thanks for submission!\nSubmission revision: {revision}"
    )
