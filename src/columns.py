from dataclasses import dataclass, make_dataclass


def _fields(raw_class):
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


def get_default_auto_eval_column_dict():
    auto_eval_column_dict = []
    auto_eval_column_dict.append(["rank", ColumnContent, ColumnContent(COL_NAME_RANK, "number", True)])
    auto_eval_column_dict.append(
        [
            "retrieval_model",
            ColumnContent,
            ColumnContent(COL_NAME_RETRIEVAL_MODEL, "markdown", True, never_hidden=True),
        ]
    )
    auto_eval_column_dict.append(
        [
            "reranking_model",
            ColumnContent,
            ColumnContent(COL_NAME_RERANKING_MODEL, "markdown", True, never_hidden=True),
        ]
    )
    auto_eval_column_dict.append(
        ["revision", ColumnContent, ColumnContent(COL_NAME_REVISION, "markdown", True, never_hidden=True)]
    )
    auto_eval_column_dict.append(
        ["timestamp", ColumnContent, ColumnContent(COL_NAME_TIMESTAMP, "date", True, never_hidden=True)]
    )
    auto_eval_column_dict.append(["average", ColumnContent, ColumnContent(COL_NAME_AVG, "number", True)])
    auto_eval_column_dict.append(
        [
            "retrieval_model_link",
            ColumnContent,
            ColumnContent(
                COL_NAME_RETRIEVAL_MODEL_LINK,
                "markdown",
                False,
                hidden=True,
            ),
        ]
    )
    auto_eval_column_dict.append(
        [
            "reranking_model_link",
            ColumnContent,
            ColumnContent(
                COL_NAME_RERANKING_MODEL_LINK,
                "markdown",
                False,
                hidden=True,
            ),
        ]
    )
    auto_eval_column_dict.append(
        ["is_anonymous", ColumnContent, ColumnContent(COL_NAME_IS_ANONYMOUS, "bool", False, hidden=True)]
    )
    return auto_eval_column_dict


def make_autoevalcolumn(cls_name, benchmarks):
    auto_eval_column_dict = get_default_auto_eval_column_dict()
    # Leaderboard columns
    for benchmark in list(benchmarks.value):
        auto_eval_column_dict.append(
            [benchmark.name, ColumnContent, ColumnContent(benchmark.value.col_name, "number", True)]
        )

    # We use make dataclass to dynamically fill the scores from Tasks
    return make_dataclass(cls_name, auto_eval_column_dict, frozen=True)


def get_default_col_names_and_types(benchmarks):
    AutoEvalColumn = make_autoevalcolumn("AutoEvalColumn", benchmarks)
    col_names = [c.name for c in _fields(AutoEvalColumn) if not c.hidden]
    col_types = [c.type for c in _fields(AutoEvalColumn) if not c.hidden]
    return col_names, col_types


def get_fixed_col_names_and_types():
    fixed_cols = get_default_auto_eval_column_dict()[:-3]
    return [c.name for _, _, c in fixed_cols], [c.type for _, _, c in fixed_cols]


COL_NAME_AVG = "Average ⬆️"
COL_NAME_RETRIEVAL_MODEL = "Retrieval Method"
COL_NAME_RERANKING_MODEL = "Reranking Model"
COL_NAME_RETRIEVAL_MODEL_LINK = "Retrieval Model LINK"
COL_NAME_RERANKING_MODEL_LINK = "Reranking Model LINK"
COL_NAME_RANK = "Rank 🏆"
COL_NAME_REVISION = "Revision"
COL_NAME_TIMESTAMP = "Submission Date"
COL_NAME_IS_ANONYMOUS = "Anonymous Submission"
