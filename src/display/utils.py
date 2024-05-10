from dataclasses import dataclass, make_dataclass

from src.benchmarks import BenchmarksQA, BenchmarksLongDoc


def fields(raw_class):
    return [v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"]


# These classes are for user facing column names,
# to avoid having to change them all around the code
# when a modification is needed
@dataclass
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False


COL_NAME_AVG = "Average ⬆️"
COL_NAME_RETRIEVAL_MODEL = "Retrieval Model"
COL_NAME_RERANKING_MODEL = "Reranking Model"

def make_autoevalcolumn(cls_name="BenchmarksQA", benchmarks=BenchmarksQA):
    ## Leaderboard columns
    auto_eval_column_dict = []
    # Init
    auto_eval_column_dict.append(
        ["retrieval_model", ColumnContent, ColumnContent(COL_NAME_RETRIEVAL_MODEL, "markdown", True, never_hidden=True)]
    )
    auto_eval_column_dict.append(
        ["reranking_model", ColumnContent, ColumnContent(COL_NAME_RERANKING_MODEL, "markdown", True, never_hidden=True)]
    )
    auto_eval_column_dict.append(
        ["average", ColumnContent, ColumnContent(COL_NAME_AVG, "number", True)]
    )
    for benchmark in benchmarks:
        auto_eval_column_dict.append(
            [benchmark.name, ColumnContent, ColumnContent(benchmark.value.col_name, "number", True)]
        )

    # We use make dataclass to dynamically fill the scores from Tasks
    return make_dataclass(cls_name, auto_eval_column_dict, frozen=True)


AutoEvalColumnQA = make_autoevalcolumn(
    "AutoEvalColumnQA", BenchmarksQA)
AutoEvalColumnLongDoc = make_autoevalcolumn(
    "AutoEvalColumnLongDoc", BenchmarksLongDoc)


## For the queue columns in the submission tab
@dataclass(frozen=True)
class EvalQueueColumn:  # Queue column
    model = ColumnContent("model", "markdown", True)
    status = ColumnContent("status", "str", True)


# Column selection
COLS_QA = [c.name for c in fields(AutoEvalColumnQA) if not c.hidden]
COLS_LONG_DOC = [c.name for c in fields(AutoEvalColumnLongDoc) if not c.hidden]
TYPES = [c.type for c in fields(AutoEvalColumnQA) if not c.hidden]
COLS_LITE = [c.name for c in fields(AutoEvalColumnQA) if c.displayed_by_default and not c.hidden]

EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]

QA_BENCHMARK_COLS = [t.value.col_name for t in BenchmarksQA]

LONG_DOC_BENCHMARK_COLS = [t.value.col_name for t in BenchmarksLongDoc]
