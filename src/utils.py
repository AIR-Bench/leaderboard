import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.benchmarks import LongDocBenchmarks, QABenchmarks
from src.columns import (
    COL_NAME_AVG,
    COL_NAME_IS_ANONYMOUS,
    COL_NAME_RANK,
    COL_NAME_RERANKING_MODEL,
    COL_NAME_RETRIEVAL_MODEL,
    COL_NAME_REVISION,
    COL_NAME_TIMESTAMP,
    get_default_col_names_and_types,
    get_fixed_col_names_and_types,
)
from src.envs import API, LATEST_BENCHMARK_VERSION, SEARCH_RESULTS_REPO
from src.models import TaskType, get_safe_name


def calculate_mean(row):
    if pd.isna(row).any():
        return -1
    else:
        return row.mean()


def remove_html(input_str):
    # Regular expression for finding HTML tags
    clean = re.sub(r"<.*?>", "", input_str)
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


def get_default_cols(task: TaskType, version_slug, add_fix_cols: bool = True) -> tuple:
    cols = []
    types = []
    if task == TaskType.qa:
        benchmarks = QABenchmarks[version_slug]
    elif task == TaskType.long_doc:
        benchmarks = LongDocBenchmarks[version_slug]
    else:
        raise NotImplementedError
    cols_list, types_list = get_default_col_names_and_types(benchmarks)
    benchmark_list = [c.value.col_name for c in list(benchmarks.value)]
    for col_name, col_type in zip(cols_list, types_list):
        if col_name not in benchmark_list:
            continue
        cols.append(col_name)
        types.append(col_type)
    if add_fix_cols:
        _cols = []
        _types = []
        fixed_cols, fixed_cols_types = get_fixed_col_names_and_types()
        for col_name, col_type in zip(cols, types):
            if col_name in fixed_cols:
                continue
            _cols.append(col_name)
            _types.append(col_type)
        cols = fixed_cols + _cols
        types = fixed_cols_types + _types
    return cols, types


def get_selected_cols(task, version_slug, domains, languages):
    cols, _ = get_default_cols(task=task, version_slug=version_slug, add_fix_cols=False)
    selected_cols = []
    for c in cols:
        if task == TaskType.qa:
            eval_col = QABenchmarks[version_slug].value[c].value
        elif task == TaskType.long_doc:
            eval_col = LongDocBenchmarks[version_slug].value[c].value
        else:
            raise NotImplementedError
        if eval_col.domain not in domains:
            continue
        if eval_col.lang not in languages:
            continue
        selected_cols.append(c)
    # We use COLS to maintain sorting
    return selected_cols


def select_columns(
    df: pd.DataFrame,
    domains: list,
    languages: list,
    task: TaskType = TaskType.qa,
    reset_ranking: bool = True,
    version_slug: str = None,
) -> pd.DataFrame:
    selected_cols = get_selected_cols(task, version_slug, domains, languages)
    fixed_cols, _ = get_fixed_col_names_and_types()
    filtered_df = df[fixed_cols + selected_cols]
    filtered_df.replace({"": pd.NA}, inplace=True)
    if reset_ranking:
        filtered_df[COL_NAME_AVG] = filtered_df[selected_cols].apply(calculate_mean, axis=1).round(decimals=2)
        filtered_df.sort_values(by=[COL_NAME_AVG], ascending=False, inplace=True)
        filtered_df.reset_index(inplace=True, drop=True)
        filtered_df = reset_rank(filtered_df)
    return filtered_df


def _update_df_elem(
    task: TaskType,
    version: str,
    source_df: pd.DataFrame,
    domains: list,
    langs: list,
    reranking_query: list,
    query: str,
    show_anonymous: bool,
    reset_ranking: bool = True,
    show_revision_and_timestamp: bool = False,
):
    filtered_df = source_df.copy()
    if not show_anonymous:
        filtered_df = filtered_df[~filtered_df[COL_NAME_IS_ANONYMOUS]]
    filtered_df = filter_models(filtered_df, reranking_query)
    filtered_df = filter_queries(query, filtered_df)
    filtered_df = select_columns(filtered_df, domains, langs, task, reset_ranking, get_safe_name(version))
    if not show_revision_and_timestamp:
        filtered_df.drop([COL_NAME_REVISION, COL_NAME_TIMESTAMP], axis=1, inplace=True)
    return filtered_df


def update_doc_df_elem(
    version: str,
    hidden_df: pd.DataFrame,
    domains: list,
    langs: list,
    reranking_query: list,
    query: str,
    show_anonymous: bool,
    show_revision_and_timestamp: bool = False,
    reset_ranking: bool = True,
):
    return _update_df_elem(
        TaskType.long_doc,
        version,
        hidden_df,
        domains,
        langs,
        reranking_query,
        query,
        show_anonymous,
        reset_ranking,
        show_revision_and_timestamp,
    )


def update_metric(
    datastore,
    task: TaskType,
    metric: str,
    domains: list,
    langs: list,
    reranking_model: list,
    query: str,
    show_anonymous: bool = False,
    show_revision_and_timestamp: bool = False,
) -> pd.DataFrame:
    if task == TaskType.qa:
        update_func = update_qa_df_elem
    elif task == TaskType.long_doc:
        update_func = update_doc_df_elem
    else:
        raise NotImplementedError
    df_elem = get_leaderboard_df(datastore, task=task, metric=metric)
    version = datastore.version
    return update_func(
        version,
        df_elem,
        domains,
        langs,
        reranking_model,
        query,
        show_anonymous,
        show_revision_and_timestamp,
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
    iso_format_timestamp = current_timestamp.isoformat().replace("+00:00", "Z")
    filename_friendly_timestamp = current_timestamp.strftime("%Y%m%d%H%M%S")
    return iso_format_timestamp, filename_friendly_timestamp


def calculate_file_md5(file_path):
    md5 = hashlib.md5()

    with open(file_path, "rb") as f:
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
    reranking_model: str = "",
    reranking_model_url: str = "",
    version: str = LATEST_BENCHMARK_VERSION,
    is_anonymous=False,
):
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
                f"failed to submit. Model url must start with `https://` or `http://`. Illegal model url: {model_url}"
            )
        if reranking_model != "NoReranker":
            if not reranking_model_url.startswith("https://") and not reranking_model_url.startswith("http://"):
                return styled_error(
                    f"failed to submit. Model url must start with `https://` or `http://`. Illegal model url: {model_url}"
                )

    # rename the uploaded file
    input_fp = Path(filepath)
    revision = calculate_file_md5(filepath)
    timestamp_config, timestamp_fn = get_iso_format_timestamp()
    output_fn = f"{timestamp_fn}-{revision}.zip"
    input_folder_path = input_fp.parent

    if not reranking_model:
        reranking_model = "NoReranker"

    API.upload_file(
        path_or_fileobj=filepath,
        path_in_repo=f"{version}/{model}/{reranking_model}/{output_fn}",
        repo_id=SEARCH_RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"feat: submit {model} to evaluate",
    )

    output_config_fn = f"{output_fn.removesuffix('.zip')}.json"
    output_config = {
        "model_name": f"{model}",
        "model_url": f"{model_url}",
        "reranker_name": f"{reranking_model}",
        "reranker_url": f"{reranking_model_url}",
        "version": f"{version}",
        "is_anonymous": is_anonymous,
        "revision": f"{revision}",
        "timestamp": f"{timestamp_config}",
    }
    with open(input_folder_path / output_config_fn, "w") as f:
        json.dump(output_config, f, indent=4, ensure_ascii=False)
    API.upload_file(
        path_or_fileobj=input_folder_path / output_config_fn,
        path_in_repo=f"{version}/{model}/{reranking_model}/{output_config_fn}",
        repo_id=SEARCH_RESULTS_REPO,
        repo_type="dataset",
        commit_message=f"feat: submit {model} + {reranking_model} config",
    )
    return styled_message(
        f"Thanks for submission!\n"
        f"Retrieval method: {model}\nReranking model: {reranking_model}\nSubmission revision: {revision}"
    )


def reset_rank(df):
    df[COL_NAME_RANK] = df[COL_NAME_AVG].rank(ascending=False, method="min")
    return df


def get_leaderboard_df(datastore, task: TaskType, metric: str) -> pd.DataFrame:
    """
    Creates a dataframe from all the individual experiment results
    """
    # load the selected metrics into a DataFrame from the raw json
    all_data_json = []
    for v in datastore.raw_data:
        all_data_json += v.to_dict(task=task.value, metric=metric)
    df = pd.DataFrame.from_records(all_data_json)

    # calculate the average scores for selected task
    if task == TaskType.qa:
        benchmarks = QABenchmarks[datastore.slug]
    elif task == TaskType.long_doc:
        benchmarks = LongDocBenchmarks[datastore.slug]
    else:
        raise NotImplementedError
    valid_cols = frozenset(df.columns.to_list())
    benchmark_cols = []
    for t in list(benchmarks.value):
        if t.value.col_name not in valid_cols:
            continue
        benchmark_cols.append(t.value.col_name)

    # filter out the columns that are not in the data
    df[COL_NAME_AVG] = df[list(benchmark_cols)].apply(calculate_mean, axis=1).round(decimals=2)
    df.sort_values(by=[COL_NAME_AVG], ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)

    # filter out columns that are not in the data
    display_cols = [COL_NAME_IS_ANONYMOUS, COL_NAME_AVG]
    default_cols, _ = get_default_col_names_and_types(benchmarks)
    for col in default_cols:
        if col in valid_cols:
            display_cols.append(col)
    df = df[display_cols].round(decimals=2)

    # rank the scores
    df = reset_rank(df)

    # shorten the revision
    df[COL_NAME_REVISION] = df[COL_NAME_REVISION].str[:6]

    return df


def set_listeners(
    task: TaskType,
    target_df,
    source_df,
    search_bar,
    version,
    selected_domains,
    selected_langs,
    selected_rerankings,
    show_anonymous,
    show_revision_and_timestamp,
):
    if task == TaskType.qa:
        update_table_func = update_qa_df_elem
    elif task == TaskType.long_doc:
        update_table_func = update_doc_df_elem
    else:
        raise NotImplementedError
    selector_list = [selected_domains, selected_langs, selected_rerankings, search_bar, show_anonymous]
    search_bar_args = [
        source_df,
        version,
    ] + selector_list
    selector_args = (
        [version, source_df]
        + selector_list
        + [
            show_revision_and_timestamp,
        ]
    )
    # Set search_bar listener
    search_bar.submit(update_table_func, search_bar_args, target_df)

    # Set column-wise listener
    for selector in selector_list:
        selector.change(
            update_table_func,
            selector_args,
            target_df,
            queue=True,
        )


def update_qa_df_elem(
    version: str,
    hidden_df: pd.DataFrame,
    domains: list,
    langs: list,
    reranking_query: list,
    query: str,
    show_anonymous: bool,
    show_revision_and_timestamp: bool = False,
    reset_ranking: bool = True,
):
    return _update_df_elem(
        TaskType.qa,
        version,
        hidden_df,
        domains,
        langs,
        reranking_query,
        query,
        show_anonymous,
        reset_ranking,
        show_revision_and_timestamp,
    )


def styled_error(error):
    return f"<p style='color: red; font-size: 20px; text-align: center;'>{error}</p>"


def styled_message(message):
    return f"<p style='color: green; font-size: 20px; text-align: center;'>{message}</p>"
