from src.benchmarks import BenchmarksQA, BenchmarksLongDoc


def test_qabenchmarks():
    print(list(BenchmarksQA))


def test_longdocbenchmarks():
    print(list(BenchmarksLongDoc))
