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
from utils import update_table, update_metric
from src.benchmarks import DOMAIN_COLS_QA, LANG_COLS_QA, metric_list


from functools import partial

def restart_space():
    API.restart_space(repo_id=REPO_ID)

# try:
#     print(EVAL_REQUESTS_PATH)
#     snapshot_download(
#         repo_id=QUEUE_REPO, local_dir=EVAL_REQUESTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30,
#         token=TOKEN
#     )
# except Exception:
#     restart_space()
# try:
#     print(EVAL_RESULTS_PATH)
#     snapshot_download(
#         repo_id=RESULTS_REPO, local_dir=EVAL_RESULTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30,
#         token=TOKEN
#     )
# except Exception:
#     restart_space()

from src.leaderboard.read_evals import get_raw_eval_results
raw_data_qa = get_raw_eval_results(EVAL_RESULTS_PATH, EVAL_REQUESTS_PATH)
original_df_qa = get_leaderboard_df(raw_data_qa, COLS, QA_BENCHMARK_COLS, task='qa', metric='ndcg_at_3')
print(f'data loaded: {len(raw_data_qa)}, {original_df_qa.shape}')
leaderboard_df = original_df_qa.copy()


def update_metric_qa(
        metric: str,
        domains: list,
        langs: list,
        reranking_model: list,
        query: str,
):
    return update_metric(raw_data_qa, metric, domains, langs, reranking_model, query)
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
        with gr.TabItem("QA", elem_id="llm-benchmark-tab-table", id=0):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        search_bar = gr.Textbox(
                            placeholder=" 🔍 Search for your model (separate multiple queries with `;`) and press ENTER...",
                            show_label=False,
                            elem_id="search-bar",
                        )
                    # select domain
                    with gr.Row():
                        selected_domains = gr.CheckboxGroup(
                            choices=DOMAIN_COLS_QA,
                            value=DOMAIN_COLS_QA,
                            label="Select the domains",
                            elem_id="domain-column-select",
                            interactive=True,
                        )
                    # select language
                    with gr.Row():
                        selected_langs = gr.CheckboxGroup(
                            choices=LANG_COLS_QA,
                            value=LANG_COLS_QA,
                            label="Select the languages",
                            elem_id="language-column-select",
                            interactive=True
                        )
                    # select reranking models
                    reranking_models = list(frozenset([eval_result.reranking_model for eval_result in raw_data_qa]))
                    with gr.Row():
                        selected_rerankings = gr.CheckboxGroup(
                            choices=reranking_models,
                            value=reranking_models,
                            label="Select the reranking models",
                            elem_id="reranking-select",
                            interactive=True
                        )
                with gr.Column(min_width=320):
                    selected_metric = gr.Dropdown(
                        choices=metric_list,
                        value=metric_list[1],
                        label="Select the metric",
                        interactive=True,
                        elem_id="metric-select",
                    )

            # reload the leaderboard_df and raw_data when selected_metric is changed
            leaderboard_table = gr.components.Dataframe(
                value=leaderboard_df,
                # headers=shown_columns,
                # datatype=TYPES,
                elem_id="leaderboard-table",
                interactive=False,
                visible=True,
            )

            # Dummy leaderboard for handling the case when the user uses backspace key
            hidden_leaderboard_table_for_search = gr.components.Dataframe(
                value=leaderboard_df,
                # headers=COLS,
                # datatype=TYPES,
                visible=False,
            )

            # Set search_bar listener
            search_bar.submit(
                update_table,
                [
                    hidden_leaderboard_table_for_search,
                    selected_domains,
                    selected_langs,
                    selected_rerankings,
                    search_bar,
                ],
                leaderboard_table,
            )

            # Set column-wise listener
            for selector in [
                selected_domains, selected_langs, selected_rerankings
            ]:
                selector.change(
                    update_table,
                    [
                        hidden_leaderboard_table_for_search,
                        selected_domains,
                        selected_langs,
                        selected_rerankings,
                        search_bar,
                    ],
                    leaderboard_table,
                    queue=True,
                )

            # set metric listener
            selected_metric.change(
                update_metric_qa,
                [
                    selected_metric,
                    selected_domains,
                    selected_langs,
                    selected_rerankings,
                    search_bar,
                ],
                leaderboard_table,
                queue=True
            )

        with gr.TabItem("📝 About", elem_id="llm-benchmark-tab-table", id=2):
            gr.Markdown(LLM_BENCHMARKS_TEXT, elem_classes="markdown-text")

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", seconds=1800)
scheduler.start()
demo.queue(default_concurrency_limit=40).launch()
