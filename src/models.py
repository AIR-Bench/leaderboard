import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import List

import pandas as pd

from src.columns import (
    COL_NAME_IS_ANONYMOUS,
    COL_NAME_RERANKING_MODEL,
    COL_NAME_RERANKING_MODEL_LINK,
    COL_NAME_RETRIEVAL_MODEL,
    COL_NAME_RETRIEVAL_MODEL_LINK,
    COL_NAME_REVISION,
    COL_NAME_TIMESTAMP,
)


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
                is_anonymous=config.get("is_anonymous", False),
            )
            result_list.append(eval_result)
        eval_result = result_list[0]
        return cls(
            eval_name=f"{eval_result.retrieval_model}_{eval_result.reranking_model}",
            retrieval_model=eval_result.retrieval_model,
            reranking_model=eval_result.reranking_model,
            retrieval_model_link=retrieval_model_link,
            reranking_model_link=reranking_model_link,
            results=result_list,
            timestamp=eval_result.timestamp,
            revision=eval_result.revision,
            is_anonymous=eval_result.is_anonymous,
        )

    def to_dict(self, task="qa", metric="ndcg_at_3") -> List:
        """
        Convert the results in all the EvalResults over different tasks and metrics.
        The output is a list of dict compatible with the dataframe UI
        """
        results = defaultdict(dict)
        for eval_result in self.results:
            if eval_result.metric != metric:
                continue
            if eval_result.task != task:
                continue
            eval_name = eval_result.eval_name
            results[eval_name]["eval_name"] = eval_name
            results[eval_name][COL_NAME_RETRIEVAL_MODEL] = make_clickable_model(
                self.retrieval_model, self.retrieval_model_link
            )
            results[eval_name][COL_NAME_RERANKING_MODEL] = make_clickable_model(
                self.reranking_model, self.reranking_model_link
            )
            results[eval_name][COL_NAME_RETRIEVAL_MODEL_LINK] = self.retrieval_model_link
            results[eval_name][COL_NAME_RERANKING_MODEL_LINK] = self.reranking_model_link
            results[eval_name][COL_NAME_REVISION] = self.revision
            results[eval_name][COL_NAME_TIMESTAMP] = self.timestamp
            results[eval_name][COL_NAME_IS_ANONYMOUS] = self.is_anonymous

            for result in eval_result.results:
                # add result for each domain, language, and dataset
                domain = result["domain"]
                lang = result["lang"]
                dataset = result["dataset"]
                value = result["value"] * 100
                if dataset == "default":
                    benchmark_name = f"{domain}_{lang}"
                else:
                    benchmark_name = f"{domain}_{lang}_{dataset}"
                results[eval_name][get_safe_name(benchmark_name)] = value
        return [v for v in results.values()]


@dataclass
class LeaderboardDataStore:
    version: str
    slug: str
    raw_data: list = None
    qa_raw_df: pd.DataFrame = pd.DataFrame()
    doc_raw_df: pd.DataFrame = pd.DataFrame()
    qa_fmt_df: pd.DataFrame = pd.DataFrame()
    doc_fmt_df: pd.DataFrame = pd.DataFrame()
    reranking_models: list = None
    qa_types: list = None
    doc_types: list = None


# Define an enum class with the name `TaskType`. There are two types of tasks, `qa` and `long-doc`.
class TaskType(Enum):
    qa = "qa"
    long_doc = "long-doc"


def make_clickable_model(model_name: str, model_link: str):
    # link = f"https://huggingface.co/{model_name}"
    if not model_link or not model_link.startswith("https://"):
        return model_name
    return model_hyperlink(model_link, model_name)


def model_hyperlink(link, model_name):
    return f'<a target="_blank" href="{link}" style="color: var(--link-text-color); text-decoration: underline;text-decoration-style: dotted;">{model_name}</a>'


def get_safe_name(name: str):
    """Get RFC 1123 compatible safe name"""
    name = name.replace("-", "_")
    return "".join(character.lower() for character in name if (character.isalnum() or character == "_"))
