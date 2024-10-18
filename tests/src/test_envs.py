from air_benchmark.tasks import BenchmarkTable

from src.envs import BENCHMARK_VERSION_LIST, DEFAULT_METRIC_LONG_DOC, DEFAULT_METRIC_QA, METRIC_LIST


def test_benchmark_version_list():
    leaderboard_versions = frozenset(BENCHMARK_VERSION_LIST)
    available_versions = frozenset([k for k in BenchmarkTable.keys()])
    assert leaderboard_versions.issubset(available_versions)


def test_default_metrics():
    assert DEFAULT_METRIC_QA in METRIC_LIST
    assert DEFAULT_METRIC_LONG_DOC in METRIC_LIST
