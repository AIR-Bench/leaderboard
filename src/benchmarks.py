from dataclasses import dataclass
from enum import Enum
from air_benchmark.tasks.tasks import BenchmarkTable


def get_safe_name(name: str):
    """Get RFC 1123 compatible safe name"""
    name = name.replace('-', '_')
    return ''.join(
        character.lower()
        for character in name
        if (character.isalnum() or character == '_'))


METRIC_LIST = [
    "ndcg_at_1",
    "ndcg_at_3",
    "ndcg_at_5",
    "ndcg_at_10",
    "ndcg_at_100",
    "ndcg_at_1000",
    "map_at_1",
    "map_at_3",
    "map_at_5",
    "map_at_10",
    "map_at_100",
    "map_at_1000",
    "recall_at_1",
    "recall_at_3",
    "recall_at_5",
    "recall_at_10",
    "recall_at_100",
    "recall_at_1000",
    "precision_at_1",
    "precision_at_3",
    "precision_at_5",
    "precision_at_10",
    "precision_at_100",
    "precision_at_1000",
    "mrr_at_1",
    "mrr_at_3",
    "mrr_at_5",
    "mrr_at_10",
    "mrr_at_100",
    "mrr_at_1000"
]


@dataclass
class Benchmark:
    name: str  # [domain]_[language]_[metric], task_key in the json file,
    metric: str  # ndcg_at_1 ,metric_key in the json file
    col_name: str  # [domain]_[language], name to display in the leaderboard
    domain: str
    lang: str
    task: str


qa_benchmark_dict = {}
long_doc_benchmark_dict = {}
for task, domain_dict in BenchmarkTable['AIR-Bench_24.04'].items():
    for domain, lang_dict in domain_dict.items():
        for lang, dataset_list in lang_dict.items():
            if task == "qa":
                benchmark_name = f"{domain}_{lang}"
                benchmark_name = get_safe_name(benchmark_name)
                col_name = benchmark_name
                for metric in dataset_list:
                    qa_benchmark_dict[benchmark_name] = Benchmark(benchmark_name, metric, col_name, domain, lang, task)
            elif task == "long-doc":
                for dataset in dataset_list:
                    benchmark_name = f"{domain}_{lang}_{dataset}"
                    benchmark_name = get_safe_name(benchmark_name)
                    col_name = benchmark_name
                    for metric in METRIC_LIST:
                        long_doc_benchmark_dict[benchmark_name] = Benchmark(benchmark_name, metric, col_name, domain,
                                                                            lang, task)

BenchmarksQA = Enum('BenchmarksQA', qa_benchmark_dict)
BenchmarksLongDoc = Enum('BenchmarksLongDoc', long_doc_benchmark_dict)

BENCHMARK_COLS_QA = [c.col_name for c in qa_benchmark_dict.values()]
BENCHMARK_COLS_LONG_DOC = [c.col_name for c in long_doc_benchmark_dict.values()]

DOMAIN_COLS_QA = list(frozenset([c.domain for c in qa_benchmark_dict.values()]))
LANG_COLS_QA = list(frozenset([c.lang for c in qa_benchmark_dict.values()]))

DOMAIN_COLS_LONG_DOC = list(frozenset([c.domain for c in long_doc_benchmark_dict.values()]))
LANG_COLS_LONG_DOC = list(frozenset([c.lang for c in long_doc_benchmark_dict.values()]))

DEFAULT_METRIC_QA = "ndcg_at_10"
DEFAULT_METRIC_LONG_DOC = "recall_at_10"
