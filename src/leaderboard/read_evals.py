import json
import os.path
from collections import defaultdict
from dataclasses import dataclass
from typing import List

import dateutil.parser._parser
import pandas as pd

from src.benchmarks import get_safe_name
from src.display.formatting import has_no_nan_values
from src.display.utils import COL_NAME_RERANKING_MODEL, COL_NAME_RETRIEVAL_MODEL, COLS_QA, QA_BENCHMARK_COLS, \
    COLS_LONG_DOC, LONG_DOC_BENCHMARK_COLS, COL_NAME_AVG


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


@dataclass
class FullEvalResult:
    """
    Evaluation result of a single embedding model with a specific reranking model on benchmarks over different tasks
    """
    eval_name: str  # name of the evaluation, [retrieval_model]_[reranking_model]
    retrieval_model: str
    reranking_model: str
    results: List[EvalResult]  # results on all the EvalResults over different tasks and metrics.
    date: str = ""

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
        for item in model_data:
            config = item.get("config", {})
            # eval results for different metrics
            results = item.get("results", [])
            eval_result = EvalResult(
                eval_name=f"{config['retrieval_model']}_{config['reranking_model']}_{config['metric']}",
                retrieval_model=config["retrieval_model"],
                reranking_model=config["reranking_model"],
                results=results,
                task=config["task"],
                metric=config["metric"]
            )
            result_list.append(eval_result)
        return cls(
            eval_name=f"{result_list[0].retrieval_model}_{result_list[0].reranking_model}",
            retrieval_model=result_list[0].retrieval_model,
            reranking_model=result_list[0].reranking_model,
            results=result_list
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
            results[eval_result.eval_name][COL_NAME_RETRIEVAL_MODEL] = self.retrieval_model
            results[eval_result.eval_name][COL_NAME_RERANKING_MODEL] = self.reranking_model

            print(f'result loaded: {eval_result.eval_name}')
            for result in eval_result.results:
                # add result for each domain, language, and dataset
                domain = result["domain"]
                lang = result["lang"]
                dataset = result["dataset"]
                value = result["value"]
                if task == 'qa':
                    benchmark_name = f"{domain}_{lang}"
                elif task == 'long_doc':
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
        try:
            files.sort(key=lambda x: x.removesuffix(".json").removeprefix("results_")[:-7], reverse=True)
        except dateutil.parser._parser.ParserError:
            files = [files[-1]]

        # select the latest results
        for file in files:
            model_result_filepaths.append(os.path.join(root, file))

    eval_results = {}
    for model_result_filepath in model_result_filepaths:
        # create evaluation results
        eval_result = FullEvalResult.init_from_json_file(model_result_filepath)
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
    if task == "qa":
        cols = COLS_QA
        benchmark_cols = QA_BENCHMARK_COLS
    elif task == "long_doc":
        cols = COLS_LONG_DOC
        benchmark_cols = LONG_DOC_BENCHMARK_COLS
    else:
        raise NotImplemented
    all_data_json = []
    for v in raw_data:
        all_data_json += v.to_dict(task=task, metric=metric)
    df = pd.DataFrame.from_records(all_data_json)
    print(f'dataframe created: {df.shape}')

    # calculate the average score for selected benchmarks
    _benchmark_cols = frozenset(benchmark_cols).intersection(frozenset(df.columns.to_list()))
    df[COL_NAME_AVG] = df[list(_benchmark_cols)].mean(axis=1).round(decimals=2)
    df = df.sort_values(by=[COL_NAME_AVG], ascending=False)
    df.reset_index(inplace=True)

    _cols = frozenset(cols).intersection(frozenset(df.columns.to_list()))
    df = df[_cols].round(decimals=2)

    # filter out if any of the benchmarks have not been produced
    df = df[has_no_nan_values(df, _benchmark_cols)]
    return df
