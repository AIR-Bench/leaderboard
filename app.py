import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import snapshot_download

from src.about import (
    INTRODUCTION_TEXT,
    BENCHMARKS_TEXT,
    TITLE,
    EVALUATION_QUEUE_TEXT
)
from src.benchmarks import (
    DOMAIN_COLS_QA,
    LANG_COLS_QA,
    DOMAIN_COLS_LONG_DOC,
    LANG_COLS_LONG_DOC,
    METRIC_LIST,
    DEFAULT_METRIC_QA,
    DEFAULT_METRIC_LONG_DOC
)
from src.display.css_html_js import custom_css
from src.display.utils import (
    COL_NAME_IS_ANONYMOUS,
    COL_NAME_REVISION,
    COL_NAME_TIMESTAMP,
    COL_NAME_RERANKING_MODEL,
    COL_NAME_RETRIEVAL_MODEL
)
from src.envs import (
    API,
    EVAL_RESULTS_PATH,
    REPO_ID,
    RESULTS_REPO,
    TOKEN,
    BM25_LINK,
    BENCHMARK_VERSION_LIST,
    LATEST_BENCHMARK_VERSION
)
from src.read_evals import (
    get_raw_eval_results,
    get_leaderboard_df
)
from src.utils import (
    update_metric,
    upload_file,
    get_default_cols,
    submit_results,
    reset_rank,
    remove_html
)
from src.display.gradio_formatting import (
    get_version_dropdown,
    get_search_bar,
    get_reranking_dropdown,
    get_metric_dropdown,
    get_domain_dropdown,
    get_language_dropdown,
    get_anonymous_checkbox,
    get_revision_and_ts_checkbox,
    get_leaderboard_table,
    get_noreranking_dropdown
)
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

raw_data = get_raw_eval_results(f"{EVAL_RESULTS_PATH}/{LATEST_BENCHMARK_VERSION}")

original_df_qa = get_leaderboard_df(
    raw_data, task='qa', metric=DEFAULT_METRIC_QA)
original_df_long_doc = get_leaderboard_df(
    raw_data, task='long-doc', metric=DEFAULT_METRIC_LONG_DOC)
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
        with gr.TabItem("Results", elem_id="results-tab-table"):
            with gr.Row():
                selected_version = get_version_dropdown()

            with gr.TabItem("QA", elem_id="qa-benchmark-tab-table", id=0):
                with gr.Row():
                    with gr.Column(min_width=320):
                        # select domain
                        with gr.Row():
                            selected_domains = get_domain_dropdown(DOMAIN_COLS_QA, DOMAIN_COLS_QA)
                        # select language
                        with gr.Row():
                            selected_langs = get_language_dropdown(LANG_COLS_QA, LANG_COLS_QA)

                    with gr.Column():
                        # select the metric
                        selected_metric = get_metric_dropdown(METRIC_LIST, DEFAULT_METRIC_QA)
                        with gr.Row():
                            show_anonymous = get_anonymous_checkbox()
                        with gr.Row():
                            show_revision_and_timestamp = get_revision_and_ts_checkbox()
                with gr.Tabs(elem_classes="tab-buttons") as sub_tabs:
                    with gr.TabItem("Retrieval + Reranking", id=10):
                        with gr.Row():
                            # search retrieval models
                            with gr.Column():
                                search_bar = get_search_bar()
                            # select reranking models
                            with gr.Column():
                                selected_rerankings = get_reranking_dropdown(reranking_models)
                        leaderboard_table = get_leaderboard_table(leaderboard_df_qa, types_qa)
                        # Dummy leaderboard for handling the case when the user uses backspace key
                        hidden_leaderboard_table_for_search = get_leaderboard_table(original_df_qa, types_qa, visible=False)

                        set_listeners(
                            "qa",
                            leaderboard_table,
                            hidden_leaderboard_table_for_search,
                            search_bar,
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
                                show_revision_and_timestamp,
                            ],
                            leaderboard_table,
                            queue=True
                        )
                    with gr.TabItem("Retrieval Only", id=11):
                        with gr.Row():
                            with gr.Column(scale=1):
                                search_bar_retriever = get_search_bar()
                            with gr.Column(scale=1):
                                selected_noreranker = get_noreranking_dropdown()
                        lb_df_retriever = leaderboard_df_qa[leaderboard_df_qa[COL_NAME_RERANKING_MODEL] == "NoReranker"]
                        lb_df_retriever = reset_rank(lb_df_retriever)
                        lb_table_retriever = get_leaderboard_table(lb_df_retriever, types_qa)
                        # Dummy leaderboard for handling the case when the user uses backspace key
                        hidden_lb_df_retriever = original_df_qa[original_df_qa[COL_NAME_RERANKING_MODEL] == "NoReranker"]
                        hidden_lb_df_retriever = reset_rank(hidden_lb_df_retriever)
                        hidden_lb_table_retriever = get_leaderboard_table(hidden_lb_df_retriever, types_qa, visible=False)

                        set_listeners(
                            "qa",
                            lb_table_retriever,
                            hidden_lb_table_retriever,
                            search_bar_retriever,
                            selected_domains,
                            selected_langs,
                            selected_noreranker,
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
                                selected_noreranker,
                                search_bar_retriever,
                                show_anonymous,
                                show_revision_and_timestamp,
                            ],
                            lb_table_retriever,
                            queue=True
                        )
                    with gr.TabItem("Reranking Only", id=12):
                        lb_df_reranker = leaderboard_df_qa[leaderboard_df_qa[COL_NAME_RETRIEVAL_MODEL] == BM25_LINK]
                        lb_df_reranker = reset_rank(lb_df_reranker)
                        reranking_models_reranker = lb_df_reranker[COL_NAME_RERANKING_MODEL].apply(remove_html).unique().tolist()
                        with gr.Row():
                            with gr.Column(scale=1):
                                selected_rerankings_reranker = get_reranking_dropdown(reranking_models_reranker)
                            with gr.Column(scale=1):
                                search_bar_reranker = gr.Textbox(show_label=False, visible=False)
                        lb_table_reranker = get_leaderboard_table(lb_df_reranker, types_qa)
                        hidden_lb_df_reranker = original_df_qa[original_df_qa[COL_NAME_RETRIEVAL_MODEL] == BM25_LINK]
                        hidden_lb_df_reranker = reset_rank(hidden_lb_df_reranker)
                        hidden_lb_table_reranker = get_leaderboard_table(
                            hidden_lb_df_reranker, types_qa, visible=False
                        )

                        set_listeners(
                            "qa",
                            lb_table_reranker,
                            hidden_lb_table_reranker,
                            search_bar_reranker,
                            selected_domains,
                            selected_langs,
                            selected_rerankings_reranker,
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
                                selected_rerankings_reranker,
                                search_bar_reranker,
                                show_anonymous,
                                show_revision_and_timestamp,
                            ],
                            lb_table_reranker,
                            queue=True
                        )
            with gr.TabItem("Long Doc", elem_id="long-doc-benchmark-tab-table", id=1):
                with gr.Row():
                    with gr.Column(min_width=320):
                        # select domain
                        with gr.Row():
                            selected_domains = get_domain_dropdown(DOMAIN_COLS_LONG_DOC, DOMAIN_COLS_LONG_DOC)
                        # select language
                        with gr.Row():
                            selected_langs = get_language_dropdown(
                                LANG_COLS_LONG_DOC, LANG_COLS_LONG_DOC
                            )
                    with gr.Column():
                        # select the metric
                        with gr.Row():
                            selected_metric = get_metric_dropdown(METRIC_LIST, DEFAULT_METRIC_LONG_DOC)
                        with gr.Row():
                            show_anonymous = get_anonymous_checkbox()
                        with gr.Row():
                            show_revision_and_timestamp = get_revision_and_ts_checkbox()
                with gr.Tabs(elem_classes="tab-buttons") as sub_tabs:
                    with gr.TabItem("Retrieval + Reranking", id=20):
                        with gr.Row():
                            with gr.Column():
                                search_bar = get_search_bar()
                            # select reranking model
                            with gr.Column():
                                selected_rerankings = get_reranking_dropdown(reranking_models)

                        lb_table = get_leaderboard_table(
                            leaderboard_df_long_doc, types_long_doc
                        )

                        # Dummy leaderboard for handling the case when the user uses backspace key
                        hidden_lb_table_for_search = get_leaderboard_table(
                            original_df_long_doc, types_long_doc, visible=False
                        )

                        set_listeners(
                            "long-doc",
                            lb_table,
                            hidden_lb_table_for_search,
                            search_bar,
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
                            lb_table,
                            queue=True
                        )
                    with gr.TabItem("Retrieval Only", id=21):
                        with gr.Row():
                            with gr.Column(scale=1):
                                search_bar_retriever = get_search_bar()
                            with gr.Column(scale=1):
                                selected_noreranker = get_noreranking_dropdown()
                        lb_df_retriever_long_doc = leaderboard_df_long_doc[
                            leaderboard_df_long_doc[COL_NAME_RERANKING_MODEL] == "NoReranker"
                        ]
                        lb_df_retriever_long_doc = reset_rank(lb_df_retriever_long_doc)
                        hidden_lb_db_retriever_long_doc = original_df_long_doc[
                            original_df_long_doc[COL_NAME_RERANKING_MODEL] == "NoReranker"
                        ]
                        hidden_lb_db_retriever_long_doc = reset_rank(hidden_lb_db_retriever_long_doc)
                        lb_table_retriever_long_doc = get_leaderboard_table(
                            lb_df_retriever_long_doc, types_long_doc)
                        hidden_lb_table_retriever_long_doc = get_leaderboard_table(
                            hidden_lb_db_retriever_long_doc, types_long_doc, visible=False
                        )

                        set_listeners(
                            "long-doc",
                            lb_table_retriever_long_doc,
                            hidden_lb_table_retriever_long_doc,
                            search_bar_retriever,
                            selected_domains,
                            selected_langs,
                            selected_noreranker,
                            show_anonymous,
                            show_revision_and_timestamp,
                        )

                        selected_metric.change(
                            update_metric_long_doc,
                            [
                                selected_metric,
                                selected_domains,
                                selected_langs,
                                selected_noreranker,
                                search_bar_retriever,
                                show_anonymous,
                                show_revision_and_timestamp,
                            ],
                            lb_table_retriever_long_doc,
                            queue=True
                        )
                    with gr.TabItem("Reranking Only", id=22):
                        lb_df_reranker_ldoc = leaderboard_df_long_doc[
                            leaderboard_df_long_doc[COL_NAME_RETRIEVAL_MODEL] == BM25_LINK
                            ]
                        lb_df_reranker_ldoc = reset_rank(lb_df_reranker_ldoc)
                        reranking_models_reranker_ldoc = lb_df_reranker_ldoc[COL_NAME_RERANKING_MODEL].apply(remove_html).unique().tolist()
                        with gr.Row():
                            with gr.Column(scale=1):
                                selected_rerankings_reranker_ldoc = get_reranking_dropdown(reranking_models_reranker_ldoc)
                            with gr.Column(scale=1):
                                search_bar_reranker_ldoc = gr.Textbox(show_label=False, visible=False)
                        lb_table_reranker_ldoc = get_leaderboard_table(lb_df_reranker_ldoc, types_long_doc)
                        hidden_lb_df_reranker_ldoc = original_df_long_doc[original_df_long_doc[COL_NAME_RETRIEVAL_MODEL] == BM25_LINK]
                        hidden_lb_df_reranker_ldoc = reset_rank(hidden_lb_df_reranker_ldoc)
                        hidden_lb_table_reranker_ldoc = get_leaderboard_table(
                            hidden_lb_df_reranker_ldoc, types_long_doc, visible=False
                        )

                        set_listeners(
                            "long-doc",
                            lb_table_reranker_ldoc,
                            hidden_lb_table_reranker_ldoc,
                            search_bar_reranker_ldoc,
                            selected_domains,
                            selected_langs,
                            selected_rerankings_reranker_ldoc,
                            show_anonymous,
                            show_revision_and_timestamp,
                        )
                        selected_metric.change(
                            update_metric_long_doc,
                            [
                                selected_metric,
                                selected_domains,
                                selected_langs,
                                selected_rerankings_reranker_ldoc,
                                search_bar_reranker_ldoc,
                                show_anonymous,
                                show_revision_and_timestamp,
                            ],
                            lb_table_reranker_ldoc,
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
                            BENCHMARK_VERSION_LIST,
                            value=LATEST_BENCHMARK_VERSION,
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

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(restart_space, "interval", seconds=1800)
    scheduler.start()
    demo.queue(default_concurrency_limit=40)
    demo.launch()
