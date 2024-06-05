import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import snapshot_download

from src.about import (
    INTRODUCTION_TEXT,
    BENCHMARKS_TEXT,
    TITLE,
    EVALUATION_QUEUE_TEXT
)
from src.benchmarks import DOMAIN_COLS_QA, LANG_COLS_QA, DOMAIN_COLS_LONG_DOC, LANG_COLS_LONG_DOC, METRIC_LIST, \
    DEFAULT_METRIC
from src.display.css_html_js import custom_css
from src.display.utils import COL_NAME_IS_ANONYMOUS, COL_NAME_REVISION, COL_NAME_TIMESTAMP
from src.envs import API, EVAL_RESULTS_PATH, REPO_ID, RESULTS_REPO, TOKEN
from src.read_evals import get_raw_eval_results, get_leaderboard_df
from src.utils import update_metric, upload_file, get_default_cols, submit_results
from src.display.gradio_formatting import get_version_dropdown, get_search_bar, get_reranking_dropdown, get_noreranker_button, get_metric_dropdown, get_domain_dropdown, get_language_dropdown, get_anonymous_checkbox, get_revision_and_ts_checkbox, get_leaderboard_table
from src.display.gradio_listener import set_listeners

def restart_space():
    API.restart_space(repo_id=REPO_ID)


try:
    snapshot_download(
        repo_id=RESULTS_REPO, local_dir=EVAL_RESULTS_PATH, repo_type="dataset", tqdm_class=None, etag_timeout=30,
        token=TOKEN
    )
except Exception as e:
    print(f'failed to download')
    restart_space()

raw_data = get_raw_eval_results(f"{EVAL_RESULTS_PATH}/AIR-Bench_24.04")

original_df_qa = get_leaderboard_df(
    raw_data, task='qa', metric=DEFAULT_METRIC)
original_df_long_doc = get_leaderboard_df(
    raw_data, task='long-doc', metric=DEFAULT_METRIC)
print(f'raw data: {len(raw_data)}')
print(f'QA data loaded: {original_df_qa.shape}')
print(f'Long-Doc data loaded: {len(original_df_long_doc)}')

leaderboard_df_qa = original_df_qa.copy()
# leaderboard_df_qa = leaderboard_df_qa[has_no_nan_values(df, _benchmark_cols)]
shown_columns_qa, types_qa = get_default_cols(
    'qa', leaderboard_df_qa.columns, add_fix_cols=True)
leaderboard_df_qa = leaderboard_df_qa[~leaderboard_df_qa[COL_NAME_IS_ANONYMOUS]][shown_columns_qa]
leaderboard_df_qa.drop([COL_NAME_REVISION, COL_NAME_TIMESTAMP], axis=1, inplace=True)

leaderboard_df_long_doc = original_df_long_doc.copy()
shown_columns_long_doc, types_long_doc = get_default_cols(
    'long-doc', leaderboard_df_long_doc.columns, add_fix_cols=True)
leaderboard_df_long_doc = leaderboard_df_long_doc[~leaderboard_df_long_doc[COL_NAME_IS_ANONYMOUS]][shown_columns_long_doc]
leaderboard_df_long_doc.drop([COL_NAME_REVISION, COL_NAME_TIMESTAMP], axis=1, inplace=True)

# select reranking model
reranking_models = sorted(list(frozenset([eval_result.reranking_model for eval_result in raw_data])))


def update_metric_qa(
        metric: str,
        domains: list,
        langs: list,
        reranking_model: list,
        query: str,
        show_anonymous: bool,
        show_revision_and_timestamp,
):
    return update_metric(raw_data, 'qa', metric, domains, langs, reranking_model, query, show_anonymous, show_revision_and_timestamp)

def update_metric_long_doc(
        metric: str,
        domains: list,
        langs: list,
        reranking_model: list,
        query: str,
        show_anonymous: bool,
        show_revision_and_timestamp,
):
    return update_metric(raw_data, "long-doc", metric, domains, langs, reranking_model, query, show_anonymous, show_revision_and_timestamp)


demo = gr.Blocks(css=custom_css)
with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("QA", elem_id="qa-benchmark-tab-table", id=0):
            with gr.Row():
                with gr.Column():
                    # search retrieval models
                    with gr.Row():
                        selected_version = get_version_dropdown()
                    with gr.Row():
                        search_bar = get_search_bar()
                    with gr.Row():
                        selected_rerankings = get_reranking_dropdown(reranking_models)
                    with gr.Row():
                        select_noreranker_only_btn = get_noreranker_button()

                with gr.Column(min_width=320):
                    # select the metric
                    selected_metric = get_metric_dropdown(METRIC_LIST, DEFAULT_METRIC)
                    # select domain
                    with gr.Row():
                        selected_domains = get_domain_dropdown(DOMAIN_COLS_QA, DOMAIN_COLS_QA)
                    # select language
                    with gr.Row():
                        selected_langs = get_language_dropdown(LANG_COLS_QA, LANG_COLS_QA)
                    with gr.Row():
                        show_anonymous = get_anonymous_checkbox()
                    with gr.Row():
                        show_revision_and_timestamp = get_revision_and_ts_checkbox()


            leaderboard_table = get_leaderboard_table(leaderboard_df_qa, types_qa)

            # Dummy leaderboard for handling the case when the user uses backspace key
            hidden_leaderboard_table_for_search = get_leaderboard_table(original_df_qa, types_qa, visible=False)

            set_listeners(
                "qa",
                leaderboard_table,
                hidden_leaderboard_table_for_search,
                search_bar,
                select_noreranker_only_btn,
                selected_domains,
                selected_langs,
                selected_rerankings,
                show_anonymous,
                show_revision_and_timestamp,
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
                    show_anonymous,
                ],
                leaderboard_table,
                queue=True
            )

        with gr.TabItem("Long Doc", elem_id="long-doc-benchmark-tab-table", id=1):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        selected_version = get_version_dropdown()
                    with gr.Row():
                        search_bar = get_search_bar()
                    # select reranking model
                    with gr.Row():
                        selected_rerankings = get_reranking_dropdown(reranking_models)
                    with gr.Row():
                        select_noreranker_only_btn = get_noreranker_button()
                with gr.Column(min_width=320):
                    # select the metric
                    with gr.Row():
                        selected_metric = get_metric_dropdown(METRIC_LIST, DEFAULT_METRIC)
                    # select domain
                    with gr.Row():
                        selected_domains = get_domain_dropdown(DOMAIN_COLS_LONG_DOC, DOMAIN_COLS_LONG_DOC)
                    # select language
                    with gr.Row():
                        selected_langs = get_language_dropdown(
                            LANG_COLS_LONG_DOC, LANG_COLS_LONG_DOC
                        )
                    with gr.Row():
                        show_anonymous = get_anonymous_checkbox()
                    with gr.Row():
                        show_revision_and_timestamp = get_revision_and_ts_checkbox()

            leaderboard_table = get_leaderboard_table(
                leaderboard_df_long_doc, types_long_doc
            )

            # Dummy leaderboard for handling the case when the user uses backspace key
            hidden_leaderboard_table_for_search =get_leaderboard_table(
                original_df_long_doc, types_long_doc, visible=False
            )

            set_listeners(
                "long-doc",
                leaderboard_table,
                hidden_leaderboard_table_for_search,
                search_bar,
                select_noreranker_only_btn,
                selected_domains,
                selected_langs,
                selected_rerankings,
                show_anonymous,
                show_revision_and_timestamp,
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
                    show_anonymous,
                    show_revision_and_timestamp
                ],
                leaderboard_table,
                queue=True
            )

        with gr.TabItem("🚀Submit here!", elem_id="submit-tab-table", id=2):
            with gr.Column():
                with gr.Row():
                    gr.Markdown(EVALUATION_QUEUE_TEXT, elem_classes="markdown-text")
                with gr.Row():
                    gr.Markdown("## ✉️Submit your model here!", elem_classes="markdown-text")
                with gr.Row():
                    with gr.Column():
                        model_name = gr.Textbox(label="Retrieval Method name")
                    with gr.Column():
                        model_url = gr.Textbox(label="Retrieval Method URL")
                with gr.Row():
                    with gr.Column():
                        reranking_model_name = gr.Textbox(
                            label="Reranking Model name",
                            info="Optional",
                            value="NoReranker"
                        )
                    with gr.Column():
                        reranking_model_url = gr.Textbox(
                            label="Reranking Model URL",
                            info="Optional",
                            value=""
                        )
                with gr.Row():
                    with gr.Column():
                        benchmark_version = gr.Dropdown(
                            ["AIR-Bench_24.04", ],
                            value="AIR-Bench_24.04",
                            interactive=True,
                            label="AIR-Bench Version")
                with gr.Row():
                    upload_button = gr.UploadButton("Click to upload search results", file_count="single")
                with gr.Row():
                    file_output = gr.File()
                with gr.Row():
                    is_anonymous = gr.Checkbox(
                        label="Nope. I want to submit anonymously 🥷",
                        value=False,
                        info="Do you want to shown on the leaderboard by default?")
                with gr.Row():
                    submit_button = gr.Button("Submit")
                with gr.Row():
                    submission_result = gr.Markdown()
                upload_button.upload(
                    upload_file,
                    [
                        upload_button,
                    ],
                    file_output)
                submit_button.click(
                    submit_results,
                    [
                        file_output,
                        model_name,
                        model_url,
                        reranking_model_name,
                        reranking_model_url,
                        benchmark_version,
                        is_anonymous
                    ],
                    submission_result,
                    show_progress="hidden"
                )

        with gr.TabItem("📝 About", elem_id="llm-benchmark-tab-table", id=3):
            gr.Markdown(BENCHMARKS_TEXT, elem_classes="markdown-text")

scheduler = BackgroundScheduler()
scheduler.add_job(restart_space, "interval", seconds=1800)
scheduler.start()
demo.queue(default_concurrency_limit=40).launch()
