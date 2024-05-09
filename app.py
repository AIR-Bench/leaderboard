import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import snapshot_download

from src.about import (
    INTRODUCTION_TEXT,
    LLM_BENCHMARKS_TEXT,
    TITLE,
)
from src.display.css_html_js import custom_css
from src.display.utils import (
    QA_BENCHMARK_COLS,
    COLS,
    TYPES,
    AutoEvalColumnQA,
    fields
)
from src.envs import API, EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, QUEUE_REPO, REPO_ID, RESULTS_REPO, TOKEN
from src.populate import get_leaderboard_df
from utils import update_table


def restart_space():
    API.restart_space(repo_id=REPO_ID)


try:
    print(EVAL_REQUESTS_PATH)
    snapshot_download(
        repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30,
        token=TOKEN
    )
except Exception:
    restart_space()
try:
    print(EVAL_RESULTS_PATH)
    snapshot_download(
        repo_id=RESULTS_REPO, local_dir=EVAL_RESULTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30,
        token=TOKEN
    )
except Exception:
    restart_space()

raw_data_qa, original_df_qa = get_leaderboard_df(
    EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH, COLS, QA_BENCHMARK_COLS, task='qa', metric='ndcg_at_1')
leaderboard_df = original_df_qa.copy()

# (
#     finished_eval_queue_df,
#     running_eval_queue_df,
#     pending_eval_queue_df,
# ) = get_evaluation_queue_df(EVAL_REQUESTS_PATH, EVAL_COLS)


demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("🏅 LLM Benchmark", elem_id="llm-benchmark-tab-table", id=0):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        search_bar = gr.Textbox(
                            placeholder=" 🔍 Search for your model (separate multiple queries with `;`) and press ENTER...",
                            show_label=False,
                            elem_id="search-bar",
                        )
                    with gr.Row():
                        shown_columns = gr.CheckboxGroup(
                            choices=[
                                c.name
                                for c in fields(AutoEvalColumnQA)
                                if not c.hidden and not c.never_hidden
                            ],
                            value=[
                                c.name
                                for c in fields(AutoEvalColumnQA)
                                if c.displayed_by_default and not c.hidden and not c.never_hidden
                            ],
                            label="Select columns to show",
                            elem_id="column-select",
                            interactive=True,
                        )
                    with gr.Row():
                        deleted_models_visibility = gr.Checkbox(
                            value=False, label="Show gated/private/deleted models", interactive=True
                        )
                with gr.Column(min_width=320):
                    # with gr.Box(elem_id="box-filter"):
                    filter_columns_type = gr.CheckboxGroup(
                        label="Model types",
                        choices=[t.to_str() for t in ModelType],
                        value=[t.to_str() for t in ModelType],
                        interactive=True,
                        elem_id="filter-columns-type",
                    )
                    filter_columns_precision = gr.CheckboxGroup(
                        label="Precision",
                        choices=[i.value.name for i in Precision],
                        value=[i.value.name for i in Precision],
                        interactive=True,
                        elem_id="filter-columns-precision",
                    )
                    filter_columns_size = gr.CheckboxGroup(
                        label="Model sizes (in billions of parameters)",
                        choices=list(NUMERIC_INTERVALS.keys()),
                        value=list(NUMERIC_INTERVALS.keys()),
                        interactive=True,
                        elem_id="filter-columns-size",
                    )

            leaderboard_table = gr.components.Dataframe(
                value=leaderboard_df[
                    [c.name for c in fields(AutoEvalColumnQA) if c.never_hidden]
                    + shown_columns.value
                    ],
                headers=[c.name for c in fields(AutoEvalColumnQA) if c.never_hidden] + shown_columns.value,
                datatype=TYPES,
                elem_id="leaderboard-table",
                interactive=False,
                visible=True,
            )

            # Dummy leaderboard for handling the case when the user uses backspace key
            hidden_leaderboard_table_for_search = gr.components.Dataframe(
                value=original_df_qa[COLS],
                headers=COLS,
                datatype=TYPES,
                visible=False,
            )
            search_bar.submit(
                update_table,
                [
                    hidden_leaderboard_table_for_search,
                    shown_columns,
                    filter_columns_type,
                    filter_columns_precision,
                    filter_columns_size,
                    deleted_models_visibility,
                    search_bar,
                ],
                leaderboard_table,
            )
            for selector in [shown_columns, filter_columns_type, filter_columns_precision, filter_columns_size,
                             deleted_models_visibility]:
                selector.change(
                    update_table,
                    [
                        hidden_leaderboard_table_for_search,
                        shown_columns,
                        filter_columns_type,
                        filter_columns_precision,
                        filter_columns_size,
                        deleted_models_visibility,
                        search_bar,
                    ],
                    leaderboard_table,
                    queue=True,
                )

        with gr.TabItem("📝 About", elem_id="llm-benchmark-tab-table", id=2):
            gr.Markdown(LLM_BENCHMARKS_TEXT, elem_classes="markdown-text")

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", seconds=1800)
scheduler.start()
demo.queue(default_concurrency_limit=40).launch()
