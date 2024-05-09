from dataclasses import dataclass, make_dataclass

from src.benchmarks import Benchmarks


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


## Leaderboard columns
auto_eval_column_dict = []
# Init
auto_eval_column_dict.append(
    ["retrieval_model", ColumnContent, ColumnContent("Retrieval Model", "markdown", True, never_hidden=True)]
)
auto_eval_column_dict.append(
    ["reranking_model", ColumnContent, ColumnContent("Reranking Model", "markdown", True, never_hidden=True)]
)
auto_eval_column_dict.append(
    ["average", ColumnContent, ColumnContent("Average ⬆️", "number", True)]
)
for benchmark in Benchmarks:
    auto_eval_column_dict.append(
        [benchmark.name, ColumnContent, ColumnContent(benchmark.value.col_name, "number", True)]
    )

# We use make dataclass to dynamically fill the scores from Tasks
AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)


## For the queue columns in the submission tab
@dataclass(frozen=True)
class EvalQueueColumn:  # Queue column
    model = ColumnContent("model", "markdown", True)
    status = ColumnContent("status", "str", True)


# Column selection
COLS = [c.name for c in fields(AutoEvalColumn) if not c.hidden]
TYPES = [c.type for c in fields(AutoEvalColumn) if not c.hidden]
COLS_LITE = [c.name for c in fields(AutoEvalColumn) if c.displayed_by_default and not c.hidden]

EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]

BENCHMARK_COLS = [t.value.col_name for t in Benchmarks]
