import pandas as pd

from src.display.utils import AutoEvalColumnQA, COLS
from src.benchmarks import BENCHMARK_COLS_QA, BenchmarksQA
from src.leaderboard.read_evals import FullEvalResult
from typing import List
from src.populate import get_leaderboard_df
from src.display.utils import COLS, QA_BENCHMARK_COLS


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


def select_columns(df: pd.DataFrame, domain_query: list, language_query: list) -> pd.DataFrame:
    always_here_cols = [
        AutoEvalColumnQA.retrieval_model.name,
        AutoEvalColumnQA.reranking_model.name,
        AutoEvalColumnQA.average.name
    ]
    selected_cols = []
    for c in COLS:
        if c not in df.columns:
            continue
        if c not in BENCHMARK_COLS_QA:
            continue
        eval_col = BenchmarksQA[c].value
        if eval_col.domain not in domain_query:
            continue
        if eval_col.lang not in language_query:
            continue
        selected_cols.append(c)
    # We use COLS to maintain sorting
    filtered_df = df[always_here_cols + selected_cols]
    filtered_df[AutoEvalColumnQA.average.name] = filtered_df[selected_cols].mean(axis=1).round(decimals=2)
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


def update_metric(
        raw_data: List[FullEvalResult],
        metric: str,
        domains: list,
        langs: list,
        reranking_model: list,
        query: str,
) -> pd.DataFrame:
    leaderboard_df = get_leaderboard_df(raw_data, COLS, QA_BENCHMARK_COLS, task='qa', metric=metric)
    hidden_df = leaderboard_df
    return update_table(
        hidden_df,
        domains,
        langs,
        reranking_model,
        query
    )