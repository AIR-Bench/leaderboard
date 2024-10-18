import pytest

from src.benchmarks import LongDocBenchmarks, QABenchmarks
from src.envs import BENCHMARK_VERSION_LIST

# Ref: https://github.com/AIR-Bench/AIR-Bench/blob/4b27b8a8f2047a963805fcf6fb9d74be51ec440c/docs/available_tasks.md
# 24.05
# | Task | dev | test |
# | ---- | --- | ---- |
# | Long-Doc | 4 | 11 |
# | QA | 54 | 53 |
#
# 24.04
# | Task | test |
# | ---- | ---- |
# | Long-Doc | 15 |
# | QA | 13 |


@pytest.mark.parametrize("num_datasets_dict", [{"air_bench_2404": 13, "air_bench_2405": 53}])
def test_qa_benchmarks(num_datasets_dict):
    assert len(QABenchmarks) == len(BENCHMARK_VERSION_LIST)
    for benchmark_list in list(QABenchmarks):
        version_slug = benchmark_list.name
        assert num_datasets_dict[version_slug] == len(benchmark_list.value)


@pytest.mark.parametrize("num_datasets_dict", [{"air_bench_2404": 15, "air_bench_2405": 11}])
def test_doc_benchmarks(num_datasets_dict):
    assert len(LongDocBenchmarks) == len(BENCHMARK_VERSION_LIST)
    for benchmark_list in list(LongDocBenchmarks):
        version_slug = benchmark_list.name
        assert num_datasets_dict[version_slug] == len(benchmark_list.value)
