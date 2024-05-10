import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import snapshot_download

from src.about import (
    INTRODUCTION_TEXT,
    LLM_BENCHMARKS_TEXT,
    TITLE,
    EVALUATION_QUEUE_TEXT
)
from src.display.css_html_js import custom_css
from src.display.utils import (
    QA_BENCHMARK_COLS,
    LONG_DOC_BENCHMARK_COLS,
    COLS_QA,
    COLS_LONG_DOC,
    EVAL_COLS,
    TYPES,
    AutoEvalColumnQA,
    fields
)
from src.envs import API, EVAL_REQUESTS_PATH, EVAL_RESULTS_PATH, QUEUE_REPO, REPO_ID, RESULTS_REPO, TOKEN
from src.populate import get_leaderboard_df, get_evaluation_queue_df
from utils import update_table, update_metric, update_table_long_doc, upload_file
from src.benchmarks import DOMAIN_COLS_QA, LANG_COLS_QA, DOMAIN_COLS_LONG_DOC, LANG_COLS_LONG_DOC, metric_list


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
original_df_qa = get_leaderboard_df(raw_data_qa, COLS_QA, QA_BENCHMARK_COLS, task='qa', metric='ndcg_at_3')
original_df_long_doc = get_leaderboard_df(raw_data_qa, COLS_LONG_DOC, LONG_DOC_BENCHMARK_COLS, task='long_doc', metric='ndcg_at_3')
print(f'raw data: {len(raw_data_qa)}')
print(f'QA data loaded: {original_df_qa.shape}')
print(f'Long-Doc data loaded: {len(original_df_long_doc)}')

leaderboard_df = original_df_qa.copy()
leaderboard_df_long_doc = original_df_long_doc.copy()
print(leaderboard_df_long_doc.head())


def update_metric_qa(
        metric: str,
        domains: list,
        langs: list,
        reranking_model: list,
        query: str,
):
    return update_metric(raw_data_qa, 'qa', metric, domains, langs, reranking_model, query)

def update_metric_long_doc(
        metric: str,
        domains: list,
        langs: list,
        reranking_model: list,
        query: str,
):
    return update_metric(raw_data_qa, 'long_doc', metric, domains, langs, reranking_model, query)


(
    finished_eval_queue_df,
    running_eval_queue_df,
    pending_eval_queue_df,
) = get_evaluation_queue_df(EVAL_REQUESTS_PATH, EVAL_COLS)


demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("QA", elem_id="qa-benchmark-tab-table", id=0):
            with gr.Row():
                with gr.Column():
                    # search bar for model name
                    with gr.Row():
                        search_bar = gr.Textbox(
                            placeholder=" 🔍 Search for your model (separate multiple queries with `;`) and press ENTER...",
                            show_label=False,
                            elem_id="search-bar",
                        )
                    # select the metric
                    selected_metric = gr.Dropdown(
                        choices=metric_list,
                        value=metric_list[1],
                        label="Select the metric",
                        interactive=True,
                        elem_id="metric-select",
                    )
                with gr.Column(min_width=320):
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
                    # select reranking model
                    reranking_models = list(frozenset([eval_result.reranking_model for eval_result in raw_data_qa]))
                    with gr.Row():
                        selected_rerankings = gr.CheckboxGroup(
                            choices=reranking_models,
                            value=reranking_models,
                            label="Select the reranking models",
                            elem_id="reranking-select",
                            interactive=True
                        )

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

        with gr.TabItem("Long Doc", elem_id="long-doc-benchmark-tab-table", id=1):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        search_bar = gr.Textbox(
                            placeholder=" 🔍 Search for your model (separate multiple queries with `;`) and press ENTER...",
                            show_label=False,
                            elem_id="search-bar-long-doc",
                        )
                        # select the metric
                    selected_metric = gr.Dropdown(
                        choices=metric_list,
                        value=metric_list[1],
                        label="Select the metric",
                        interactive=True,
                        elem_id="metric-select-long-doc",
                    )
                with gr.Column(min_width=320):
                    # select domain
                    with gr.Row():
                        selected_domains = gr.CheckboxGroup(
                            choices=DOMAIN_COLS_LONG_DOC,
                            value=DOMAIN_COLS_LONG_DOC,
                            label="Select the domains",
                            elem_id="domain-column-select-long-doc",
                            interactive=True,
                        )
                    # select language
                    with gr.Row():
                        selected_langs = gr.CheckboxGroup(
                            choices=LANG_COLS_LONG_DOC,
                            value=LANG_COLS_LONG_DOC,
                            label="Select the languages",
                            elem_id="language-column-select-long-doc",
                            interactive=True
                        )
                    # select reranking model
                    reranking_models = list(frozenset([eval_result.reranking_model for eval_result in raw_data_qa]))
                    with gr.Row():
                        selected_rerankings = gr.CheckboxGroup(
                            choices=reranking_models,
                            value=reranking_models,
                            label="Select the reranking models",
                            elem_id="reranking-select-long-doc",
                            interactive=True
                        )

            leaderboard_table_long_doc = gr.components.Dataframe(
                value=leaderboard_df_long_doc,
                # headers=shown_columns,
                # datatype=TYPES,
                elem_id="leaderboard-table-long-doc",
                interactive=False,
                visible=True,
            )

            # Dummy leaderboard for handling the case when the user uses backspace key
            hidden_leaderboard_table_for_search = gr.components.Dataframe(
                value=leaderboard_df_long_doc,
                # headers=COLS,
                # datatype=TYPES,
                visible=False,
            )

            # Set search_bar listener
            search_bar.submit(
                update_table_long_doc,
                [
                    hidden_leaderboard_table_for_search,
                    selected_domains,
                    selected_langs,
                    selected_rerankings,
                    search_bar,
                ],
                leaderboard_table_long_doc,
            )

            # Set column-wise listener
            for selector in [
                selected_domains, selected_langs, selected_rerankings
            ]:
                selector.change(
                    update_table_long_doc,
                    [
                        hidden_leaderboard_table_for_search,
                        selected_domains,
                        selected_langs,
                        selected_rerankings,
                        search_bar,
                    ],
                    leaderboard_table_long_doc,
                    queue=True,
                )

            # set metric listener
            selected_metric.change(
                update_metric_long_doc,
                [
                    selected_metric,
                    selected_domains,
                    selected_langs,
                    selected_rerankings,
                    search_bar,
                ],
                leaderboard_table_long_doc,
                queue=True
            )

        with gr.TabItem("🚀Submit here!", elem_id="submit-tab-table", id=2):
            with gr.Column():
                with gr.Row():
                    gr.Markdown(EVALUATION_QUEUE_TEXT, elem_classes="markdown-text")
                with gr.Row():
                    with gr.Accordion(f"✅ Finished Evaluations ({len(finished_eval_queue_df)})", open=False):
                        with gr.Row():
                            finished_eval_table = gr.components.Dataframe(
                                value=finished_eval_queue_df,
                                row_count=5,
                            )
                with gr.Row():
                    with gr.Accordion(
                            f"🔄 Running Evaluation Queue ({len(running_eval_queue_df)})",
                            open=False,
                    ):
                        with gr.Row():
                            running_eval_table = gr.components.Dataframe(
                                value=running_eval_queue_df,
                                row_count=5,
                            )
                with gr.Row():
                    with gr.Accordion(
                            f"⏳ Pending Evaluation Queue ({len(pending_eval_queue_df)})",
                            open=False,
                    ):
                        with gr.Row():
                            pending_eval_table = gr.components.Dataframe(
                                value=pending_eval_queue_df,
                                row_count=5,
                            )
                with gr.Row():
                    gr.Markdown("## ✉️Submit your model here!", elem_classes="markdown-text")
                # with gr.Row():
                #     with gr.Column():
                #         model_name_textbox = gr.Textbox(label="Model name")
                #     with gr.Column():
                #         model_url = gr.Textbox(label="Model URL")
                with gr.Row():
                    file_output = gr.File()
                with gr.Row():
                    upload_button = gr.UploadButton("Click to submit evaluation", file_count="multiple")
                upload_button.upload(upload_file, upload_button, file_output)

        # with gr.TabItem("📝 About", elem_id="llm-benchmark-tab-table", id=3):
        #     gr.Markdown(LLM_BENCHMARKS_TEXT, elem_classes="markdown-text")

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", seconds=1800)
scheduler.start()
demo.queue(default_concurrency_limit=40).launch()
