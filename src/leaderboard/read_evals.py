import glob
from collections import defaultdict
import json
import os.path
from dataclasses import dataclass
from typing import List

import dateutil.parser._parser

from src.display.utils import AutoEvalColumnQA
from src.benchmarks import get_safe_name


@dataclass
class EvalResult:
    """Full evaluation result of a single embedding model
    """
    eval_name: str  # name of the evaluation, [retrieval_model]_[reranking_model]_[metric]
    retrieval_model: str
    reranking_model: str
    results: list  # results on all the benchmarks over different domains, languages, and datasets. Use benchmark.name as the key
    task: str
    metric: str
    timestamp: str = ""  # submission timestamp


@dataclass
class FullEvalResult:
    eval_name: str  # name of the evaluation, [retrieval_model]_[reranking_model]
    retrieval_model: str
    reranking_model: str
    results: List[EvalResult]  # results on all the EvalResults over different tasks and metrics.
    date: str = ""

    @classmethod
    def init_from_json_file(cls, json_filepath):
        """Initiate from the result json file for a single model.
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
        """Convert FullEvalResult to a list of dict compatible with our dataframe UI
        """
        results = defaultdict(dict)
        for eval_result in self.results:
            if eval_result.metric != metric:
                # print(f'result skipped: {metric} != {eval_result.metric}')
                continue
            if eval_result.task != task:
                # print(f'result skipped: {task} != {eval_result.task}')
                continue
            results[eval_result.eval_name]["eval_name"] = eval_result.eval_name
            results[eval_result.eval_name][AutoEvalColumnQA.retrieval_model.name] = self.retrieval_model
            results[eval_result.eval_name][AutoEvalColumnQA.reranking_model.name] = self.reranking_model

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


def get_request_file_for_model(requests_path, retrieval_model_name, reranking_model_name):
    """
    Load the request status from a json file
    """
    request_files = os.path.join(
        requests_path,
        f"{retrieval_model_name}",
        f"{reranking_model_name}",
        "eval_request_*.json",
    )
    request_files = glob.glob(request_files)

    request_file = ""
    request_files = sorted(request_files, reverse=True)
    for tmp_request_file in request_files:
        with open(tmp_request_file, "r") as f:
            req_content = json.load(f)
            if req_content["status"] in ["FINISHED"]:
                request_file = tmp_request_file
                break
    return request_file


def get_raw_eval_results(results_path: str) -> List[FullEvalResult]:
    """
    Load the evaluation results from a json file
    """
    model_result_filepaths = []
    for root, dirs, files in os.walk(results_path):
        if len(files) == 0 or any([not f.endswith(".json") for f in files]):
            continue
        try:
            files.sort(key=lambda x: x.removesuffix(".json").removeprefix("results_")[:-7], reverse=True)
        except dateutil.parser._parser.ParserError:
            files = [files[-1]]

        # select the latest and finished results
        for file in files:
            model_result_filepaths.append(os.path.join(root, file))

    eval_results = {}
    for model_result_filepath in model_result_filepaths:
        # create evaluation results
        eval_result = FullEvalResult.init_from_json_file(model_result_filepath)
        model_result_date_str = model_result_filepath.split('/')[-1].removeprefix("results_").removesuffix(".json")
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
