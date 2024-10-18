from dataclasses import dataclass
from enum import Enum

from air_benchmark.tasks.tasks import BenchmarkTable

from src.envs import BENCHMARK_VERSION_LIST, METRIC_LIST
from src.models import TaskType, get_safe_name


@dataclass
class Benchmark:
    name: str  # [domain]_[language]_[metric], task_key in the json file,
    metric: str  # metric_key in the json file
    col_name: str  # [domain]_[language], name to display in the leaderboard
    domain: str
    lang: str
    task: str


# create a function return an enum class containing all the benchmarks
def get_qa_benchmarks_dict(version: str):
    benchmark_dict = {}
    for task, domain_dict in BenchmarkTable[version].items():
        if task != TaskType.qa.value:
            continue
        for domain, lang_dict in domain_dict.items():
            for lang, dataset_list in lang_dict.items():
                benchmark_name = get_safe_name(f"{domain}_{lang}")
                col_name = benchmark_name
                for metric in dataset_list:
                    if "test" not in dataset_list[metric]["splits"]:
                        continue
                    benchmark_dict[benchmark_name] = Benchmark(benchmark_name, metric, col_name, domain, lang, task)
    return benchmark_dict


def get_doc_benchmarks_dict(version: str):
    benchmark_dict = {}
    for task, domain_dict in BenchmarkTable[version].items():
        if task != TaskType.long_doc.value:
            continue
        for domain, lang_dict in domain_dict.items():
            for lang, dataset_list in lang_dict.items():
                for dataset in dataset_list:
                    benchmark_name = f"{domain}_{lang}_{dataset}"
                    benchmark_name = get_safe_name(benchmark_name)
                    col_name = benchmark_name
                    if "test" not in dataset_list[dataset]["splits"]:
                        continue
                    for metric in METRIC_LIST:
                        benchmark_dict[benchmark_name] = Benchmark(
                            benchmark_name, metric, col_name, domain, lang, task
                        )
    return benchmark_dict


_qa_benchmark_dict = {}
for version in BENCHMARK_VERSION_LIST:
    safe_version_name = get_safe_name(version)
    _qa_benchmark_dict[safe_version_name] = Enum(f"QABenchmarks_{safe_version_name}", get_qa_benchmarks_dict(version))

_doc_benchmark_dict = {}
for version in BENCHMARK_VERSION_LIST:
    safe_version_name = get_safe_name(version)
    _doc_benchmark_dict[safe_version_name] = Enum(
        f"LongDocBenchmarks_{safe_version_name}", get_doc_benchmarks_dict(version)
    )


QABenchmarks = Enum("QABenchmarks", _qa_benchmark_dict)
LongDocBenchmarks = Enum("LongDocBenchmarks", _doc_benchmark_dict)
