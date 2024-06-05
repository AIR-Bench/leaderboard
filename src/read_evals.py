import json
import os.path
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import pandas as pd

from src.benchmarks import get_safe_name
from src.display.utils import (
    COL_NAME_RERANKING_MODEL,
    COL_NAME_RETRIEVAL_MODEL,
    COL_NAME_RERANKING_MODEL_LINK,
    COL_NAME_RETRIEVAL_MODEL_LINK,
    COL_NAME_REVISION,
    COL_NAME_TIMESTAMP,
    COL_NAME_IS_ANONYMOUS,
    COLS_QA,
    QA_BENCHMARK_COLS,
    COLS_LONG_DOC,
    LONG_DOC_BENCHMARK_COLS,
    COL_NAME_AVG,
    COL_NAME_RANK
)

from src.display.formatting import make_clickable_model

pd.options.mode.copy_on_write = True

def calculate_mean(row):
    if pd.isna(row).any():
        return 0
    else:
        return row.mean()


@dataclass
class EvalResult:
    """
    Evaluation result of a single embedding model with a specific reranking model on benchmarks over different
    domains, languages, and datasets
    """
    eval_name: str  # name of the evaluation, [retrieval_model]_[reranking_model]_[metric]
    retrieval_model: str
    reranking_model: str
    results: list  # results on all the benchmarks stored as dict
    task: str
    metric: str
    timestamp: str = ""  # submission timestamp
    revision: str = ""
    is_anonymous: bool = False


@dataclass
class FullEvalResult:
    """
    Evaluation result of a single embedding model with a specific reranking model on benchmarks over different tasks
    """
    eval_name: str  # name of the evaluation, [retrieval_model]_[reranking_model]
    retrieval_model: str
    reranking_model: str
    retrieval_model_link: str
    reranking_model_link: str
    results: List[EvalResult]  # results on all the EvalResults over different tasks and metrics.
    timestamp: str = ""
    revision: str = ""
    is_anonymous: bool = False

    @classmethod
    def init_from_json_file(cls, json_filepath):
        """
        Initiate from the result json file for a single model.
        The json file will be written only when the status is FINISHED.
        """
        with open(json_filepath) as fp:
            model_data = json.load(fp)

        # store all the results for different metrics and tasks
        result_list = []
        retrieval_model_link = ""
        reranking_model_link = ""
        revision = ""
        for item in model_data:
            config = item.get("config", {})
            # eval results for different metrics
            results = item.get("results", [])
            retrieval_model_link = config["retrieval_model_link"]
            if config["reranking_model_link"] is None:
                reranking_model_link = ""
            else:
                reranking_model_link = config["reranking_model_link"]
            eval_result = EvalResult(
                eval_name=f"{config['retrieval_model']}_{config['reranking_model']}_{config['metric']}",
                retrieval_model=config["retrieval_model"],
                reranking_model=config["reranking_model"],
                results=results,
                task=config["task"],
                metric=config["metric"],
                timestamp=config.get("timestamp", "2024-05-12T12:24:02Z"),
                revision=config.get("revision", "3a2ba9dcad796a48a02ca1147557724e"),
                is_anonymous=config.get("is_anonymous", False)
            )
            result_list.append(eval_result)
        return cls(
            eval_name=f"{result_list[0].retrieval_model}_{result_list[0].reranking_model}",
            retrieval_model=result_list[0].retrieval_model,
            reranking_model=result_list[0].reranking_model,
            retrieval_model_link=retrieval_model_link,
            reranking_model_link=reranking_model_link,
            results=result_list,
            timestamp=result_list[0].timestamp,
            revision=result_list[0].revision,
            is_anonymous=result_list[0].is_anonymous
        )

    def to_dict(self, task='qa', metric='ndcg_at_3') -> List:
        """
        Convert the results in all the EvalResults over different tasks and metrics. The output is a list of dict compatible with the dataframe UI
        """
        results = defaultdict(dict)
        for eval_result in self.results:
            if eval_result.metric != metric:
                continue
            if eval_result.task != task:
                continue
            results[eval_result.eval_name]["eval_name"] = eval_result.eval_name
            results[eval_result.eval_name][COL_NAME_RETRIEVAL_MODEL] = (
                make_clickable_model(self.retrieval_model, self.retrieval_model_link))
            results[eval_result.eval_name][COL_NAME_RERANKING_MODEL] = (
                make_clickable_model(self.reranking_model, self.reranking_model_link))
            results[eval_result.eval_name][COL_NAME_RETRIEVAL_MODEL_LINK] = self.retrieval_model_link
            results[eval_result.eval_name][COL_NAME_RERANKING_MODEL_LINK] = self.reranking_model_link
            results[eval_result.eval_name][COL_NAME_REVISION] = self.revision
            results[eval_result.eval_name][COL_NAME_TIMESTAMP] = self.timestamp
            results[eval_result.eval_name][COL_NAME_IS_ANONYMOUS] = self.is_anonymous

            # print(f'result loaded: {eval_result.eval_name}')
            for result in eval_result.results:
                # add result for each domain, language, and dataset
                domain = result["domain"]
                lang = result["lang"]
                dataset = result["dataset"]
                value = result["value"] * 100
                if dataset == 'default':
                    benchmark_name = f"{domain}_{lang}"
                else:
                    benchmark_name = f"{domain}_{lang}_{dataset}"
                results[eval_result.eval_name][get_safe_name(benchmark_name)] = value
        return [v for v in results.values()]


def get_raw_eval_results(results_path: str) -> List[FullEvalResult]:
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
                print(f'skip {file}')
                continue
            model_result_filepaths.append(os.path.join(root, file))

    eval_results = {}
    for model_result_filepath in model_result_filepaths:
        # create evaluation results
        try:
            eval_result = FullEvalResult.init_from_json_file(model_result_filepath)
        except UnicodeDecodeError as e:
            print(f"loading file failed. {model_result_filepath}")
            continue
        print(f'file loaded: {model_result_filepath}')
        eval_name = eval_result.eval_name
        eval_results[eval_name] = eval_result

    results = []
    for k, v in eval_results.items():
        try:
            v.to_dict()
            results.append(v)
        except KeyError:
            print(f"loading failed: {k}")
            continue
    return results


def get_leaderboard_df(raw_data: List[FullEvalResult], task: str, metric: str) -> pd.DataFrame:
    """
    Creates a dataframe from all the individual experiment results
    """
    cols = [COL_NAME_IS_ANONYMOUS, ]
    if task == "qa":
        cols += COLS_QA
        benchmark_cols = QA_BENCHMARK_COLS
    elif task == "long-doc":
        cols += COLS_LONG_DOC
        benchmark_cols = LONG_DOC_BENCHMARK_COLS
    else:
        raise NotImplemented
    all_data_json = []
    for v in raw_data:
        all_data_json += v.to_dict(task=task, metric=metric)
    df = pd.DataFrame.from_records(all_data_json)
    # print(f'dataframe created: {df.shape}')

    _benchmark_cols = frozenset(benchmark_cols).intersection(frozenset(df.columns.to_list()))

    # calculate the average score for selected benchmarks
    df[COL_NAME_AVG] = df[list(_benchmark_cols)].apply(calculate_mean, axis=1).round(decimals=2)
    df.sort_values(by=[COL_NAME_AVG], ascending=False, inplace=True)
    df.reset_index(inplace=True, drop=True)

    _cols = frozenset(cols).intersection(frozenset(df.columns.to_list()))
    df = df[_cols].round(decimals=2)

    # filter out if any of the benchmarks have not been produced
    df[COL_NAME_RANK] = df[COL_NAME_AVG].rank(ascending=False, method="min")

    # shorten the revision
    df[COL_NAME_REVISION] = df[COL_NAME_REVISION].str[:6]
    return df
