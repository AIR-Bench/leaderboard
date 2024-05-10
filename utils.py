from typing import List

import pandas as pd

from src.benchmarks import BENCHMARK_COLS_QA, BENCHMARK_COLS_LONG_DOC, BenchmarksQA, BenchmarksLongDoc
from src.display.utils import AutoEvalColumnQA, AutoEvalColumnLongDoc, COLS_QA, COLS_LONG_DOC
from src.leaderboard.read_evals import FullEvalResult, get_leaderboard_df


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


def select_columns(df: pd.DataFrame, domain_query: list, language_query: list, task: str = "qa") -> pd.DataFrame:
    if task == "qa":
        always_here_cols = [
            AutoEvalColumnQA.retrieval_model.name,
            AutoEvalColumnQA.reranking_model.name,
            AutoEvalColumnQA.average.name
        ]
        cols = list(frozenset(COLS_QA).intersection(frozenset(BENCHMARK_COLS_QA)))
    elif task == "long_doc":
        always_here_cols = [
            AutoEvalColumnLongDoc.retrieval_model.name,
            AutoEvalColumnLongDoc.reranking_model.name,
            AutoEvalColumnLongDoc.average.name
        ]
        cols = list(frozenset(COLS_LONG_DOC).intersection(frozenset(BENCHMARK_COLS_LONG_DOC)))
    selected_cols = []
    for c in cols:
        if c not in df.columns:
            continue
        if task == "qa":
            eval_col = BenchmarksQA[c].value
        elif task == "long_doc":
            eval_col = BenchmarksLongDoc[c].value
        if eval_col.domain not in domain_query:
            continue
        if eval_col.lang not in language_query:
            continue
        selected_cols.append(c)
    # We use COLS to maintain sorting
    filtered_df = df[always_here_cols + selected_cols]
    filtered_df[always_here_cols[2]] = filtered_df[selected_cols].mean(axis=1).round(decimals=2)
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
    elif task == 'long_doc':
        leaderboard_df = get_leaderboard_df(raw_data, task=task, metric=metric)
        return update_table_long_doc(
            leaderboard_df,
            domains,
            langs,
            reranking_model,
            query
        )


def upload_file(files):
    file_paths = [file.name for file in files]
    print(f"file uploaded: {file_paths}")
    # for fp in file_paths:
    #     # upload the file
    #     print(file_paths)
    #     HfApi(token="").upload_file(...)
    #     os.remove(fp)
    return file_paths
