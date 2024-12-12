import os

import gradio as gr
from apscheduler.schedulers.background import BackgroundScheduler
from huggingface_hub import snapshot_download

from src.about import BENCHMARKS_TEXT, EVALUATION_QUEUE_TEXT, INTRODUCTION_TEXT, TITLE
from src.benchmarks import LongDocBenchmarks, QABenchmarks
from src.columns import COL_NAME_RERANKING_MODEL, COL_NAME_RETRIEVAL_MODEL
from src.components import (
    get_anonymous_checkbox,
    get_domain_dropdown,
    get_language_dropdown,
    get_leaderboard_table,
    get_metric_dropdown,
    get_noreranking_dropdown,
    get_reranking_dropdown,
    get_revision_and_ts_checkbox,
    get_search_bar,
    get_version_dropdown,
)
from src.css_html_js import custom_css
from src.envs import (
    API,
    BENCHMARK_VERSION_LIST,
    DEFAULT_METRIC_LONG_DOC,
    DEFAULT_METRIC_QA,
    EVAL_RESULTS_PATH,
    LATEST_BENCHMARK_VERSION,
    METRIC_LIST,
    REPO_ID,
    RESULTS_REPO,
    TOKEN,
)
from src.loaders import load_eval_results
from src.models import TaskType, model_hyperlink
from src.utils import remove_html, reset_rank, set_listeners, submit_results, update_metric, upload_file


def restart_space():
    API.restart_space(repo_id=REPO_ID)


try:
    if os.environ.get("LOCAL_MODE", False):
        print("Running in local mode")
        snapshot_download(
            repo_id=RESULTS_REPO,
            local_dir=EVAL_RESULTS_PATH,
            repo_type="dataset",
            tqdm_class=None,
            etag_timeout=30,
            token=TOKEN,
        )
except Exception:
    print("failed to download")
    restart_space()

global ds_dict
ds_dict = load_eval_results(EVAL_RESULTS_PATH)
global datastore
datastore = ds_dict[LATEST_BENCHMARK_VERSION]


def update_qa_metric(
    metric: str,
    domains: list,
    langs: list,
    reranking_model: list,
    query: str,
    show_anonymous: bool,
    show_revision_and_timestamp: bool,
):
    global datastore
    return update_metric(
        datastore,
        TaskType.qa,
        metric,
        domains,
        langs,
        reranking_model,
        query,
        show_anonymous,
        show_revision_and_timestamp,
    )


def update_doc_metric(
    metric: str,
    domains: list,
    langs: list,
    reranking_model: list,
    query: str,
    show_anonymous: bool,
    show_revision_and_timestamp,
):
    global datastore
    return update_metric(
        datastore,
        TaskType.long_doc,
        metric,
        domains,
        langs,
        reranking_model,
        query,
        show_anonymous,
        show_revision_and_timestamp,
    )


def update_datastore(version):
    global datastore
    global ds_dict
    if datastore.version != version:
        print(f"updated data version: {datastore.version} -> {version}")
        datastore = ds_dict[version]
    else:
        print(f"current data version: {datastore.version}")
    return datastore


def update_qa_domains(version):
    datastore = update_datastore(version)
    domain_elem = get_domain_dropdown(QABenchmarks[datastore.slug])
    return domain_elem


def update_doc_domains(version):
    datastore = update_datastore(version)
    domain_elem = get_domain_dropdown(LongDocBenchmarks[datastore.slug])
    return domain_elem


def update_qa_langs(version):
    datastore = update_datastore(version)
    lang_elem = get_language_dropdown(QABenchmarks[datastore.slug])
    return lang_elem


def update_doc_langs(version):
    datastore = update_datastore(version)
    lang_elem = get_language_dropdown(LongDocBenchmarks[datastore.slug])
    return lang_elem


def update_qa_models(version):
    datastore = update_datastore(version)
    model_elem = get_reranking_dropdown(datastore.reranking_models)
    return model_elem


def update_qa_df_ret_rerank(version):
    datastore = update_datastore(version)
    return get_leaderboard_table(datastore.qa_fmt_df, datastore.qa_types)


def update_qa_hidden_df_ret_rerank(version):
    datastore = update_datastore(version)
    return get_leaderboard_table(datastore.qa_raw_df, datastore.qa_types, visible=False)


def update_doc_df_ret_rerank(version):
    datastore = update_datastore(version)
    return get_leaderboard_table(datastore.doc_fmt_df, datastore.doc_types)


def update_doc_hidden_df_ret_rerank(version):
    datastore = update_datastore(version)
    return get_leaderboard_table(datastore.doc_raw_df, datastore.doc_types, visible=False)


def filter_df_ret(df):
    df_ret = df[df[COL_NAME_RERANKING_MODEL] == "NoReranker"]
    df_ret = reset_rank(df_ret)
    return df_ret


def update_qa_df_ret(version):
    datastore = update_datastore(version)
    df_ret = filter_df_ret(datastore.qa_fmt_df)
    return get_leaderboard_table(df_ret, datastore.qa_types)


def update_qa_hidden_df_ret(version):
    datastore = update_datastore(version)
    df_ret_hidden = filter_df_ret(datastore.qa_raw_df)
    return get_leaderboard_table(df_ret_hidden, datastore.qa_types, visible=False)


def update_doc_df_ret(version):
    datastore = update_datastore(version)
    df_ret = filter_df_ret(datastore.doc_fmt_df)
    return get_leaderboard_table(df_ret, datastore.doc_types)


def update_doc_hidden_df_ret(version):
    datastore = update_datastore(version)
    df_ret_hidden = filter_df_ret(datastore.doc_raw_df)
    return get_leaderboard_table(df_ret_hidden, datastore.doc_types, visible=False)


def filter_df_rerank(df):
    df_rerank = df[df[COL_NAME_RETRIEVAL_MODEL] == BM25_LINK]
    df_rerank = reset_rank(df_rerank)
    return df_rerank


def update_qa_df_rerank(version):
    datastore = update_datastore(version)
    df_rerank = filter_df_rerank(datastore.qa_fmt_df)
    return get_leaderboard_table(df_rerank, datastore.qa_types)


def update_qa_hidden_df_rerank(version):
    datastore = update_datastore(version)
    df_rerank_hidden = filter_df_rerank(datastore.qa_raw_df)
    return get_leaderboard_table(df_rerank_hidden, datastore.qa_types, visible=False)


def update_doc_df_rerank(version):
    datastore = update_datastore(version)
    df_rerank = filter_df_rerank(datastore.doc_fmt_df)
    return get_leaderboard_table(df_rerank, datastore.doc_types)


def update_doc_hidden_df_rerank(version):
    datastore = update_datastore(version)
    df_rerank_hidden = filter_df_rerank(datastore.doc_raw_df)
    return get_leaderboard_table(df_rerank_hidden, datastore.doc_types, visible=False)


demo = gr.Blocks(css=custom_css)

BM25_LINK = model_hyperlink("https://github.com/castorini/pyserini", "BM25")

with demo:
    gr.HTML(TITLE)
    gr.Markdown(INTRODUCTION_TEXT, elem_classes="markdown-text")

    with gr.Tabs(elem_classes="tab-buttons") as tabs:
        with gr.TabItem("Results", elem_id="results-tab-table"):
            with gr.Row():
                version = get_version_dropdown()

            with gr.TabItem("QA", elem_id="qa-benchmark-tab-table", id=0):
                with gr.Row():
                    with gr.Column(min_width=320):
                        # select domain
                        with gr.Row():
                            domains = get_domain_dropdown(QABenchmarks[datastore.slug])
                            version.change(update_qa_domains, version, domains)
                        # select language
                        with gr.Row():
                            langs = get_language_dropdown(QABenchmarks[datastore.slug])
                            version.change(update_qa_langs, version, langs)
                    with gr.Column():
                        # select the metric
                        metric = get_metric_dropdown(METRIC_LIST, DEFAULT_METRIC_QA)
                        with gr.Row():
                            show_anonymous = get_anonymous_checkbox()
                        with gr.Row():
                            show_rev_ts = get_revision_and_ts_checkbox()
                with gr.Tabs(elem_classes="tab-buttons") as sub_tabs:
                    with gr.TabItem("Retrieval + Reranking", id=10):
                        with gr.Row():
                            # search retrieval models
                            with gr.Column():
                                search_bar = get_search_bar()
                            # select reranking models
                            with gr.Column():
                                models = get_reranking_dropdown(datastore.reranking_models)
                                version.change(update_qa_models, version, models)
                        #  shown_table
                        qa_df_elem_ret_rerank = get_leaderboard_table(datastore.qa_fmt_df, datastore.qa_types)
                        version.change(update_qa_df_ret_rerank, version, qa_df_elem_ret_rerank)
                        # Dummy leaderboard for handling the case when the user uses backspace key
                        qa_df_elem_ret_rerank_hidden = get_leaderboard_table(
                            datastore.qa_raw_df, datastore.qa_types, visible=False
                        )
                        version.change(update_qa_hidden_df_ret_rerank, version, qa_df_elem_ret_rerank_hidden)

                        set_listeners(
                            TaskType.qa,
                            qa_df_elem_ret_rerank,
                            qa_df_elem_ret_rerank_hidden,
                            search_bar,
                            version,
                            domains,
                            langs,
                            models,
                            show_anonymous,
                            show_rev_ts,
                        )

                        # set metric listener
                        metric.change(
                            update_qa_metric,
                            [metric, domains, langs, models, search_bar, show_anonymous, show_rev_ts],
                            qa_df_elem_ret_rerank,
                            queue=True,
                        )

                    with gr.TabItem("Retrieval Only", id=11):
                        with gr.Row():
                            with gr.Column(scale=1):
                                search_bar_ret = get_search_bar()
                            with gr.Column(scale=1):
                                models_ret = get_noreranking_dropdown()
                                version.change(update_qa_models, version, models_ret)
                        _qa_df_ret = filter_df_ret(datastore.qa_fmt_df)
                        qa_df_elem_ret = get_leaderboard_table(_qa_df_ret, datastore.qa_types)
                        version.change(update_qa_df_ret, version, qa_df_elem_ret)

                        # Dummy leaderboard for handling the case when the user uses backspace key
                        _qa_df_ret_hidden = filter_df_ret(datastore.qa_raw_df)
                        qa_df_elem_ret_hidden = get_leaderboard_table(
                            _qa_df_ret_hidden, datastore.qa_types, visible=False
                        )
                        version.change(update_qa_hidden_df_ret, version, qa_df_elem_ret_hidden)

                        set_listeners(
                            TaskType.qa,
                            qa_df_elem_ret,
                            qa_df_elem_ret_hidden,
                            search_bar_ret,
                            version,
                            domains,
                            langs,
                            models_ret,
                            show_anonymous,
                            show_rev_ts,
                        )

                        metric.change(
                            update_qa_metric,
                            [
                                metric,
                                domains,
                                langs,
                                models_ret,
                                search_bar_ret,
                                show_anonymous,
                                show_rev_ts,
                            ],
                            qa_df_elem_ret,
                            queue=True,
                        )

                    with gr.TabItem("Reranking Only", id=12):
                        _qa_df_rerank = filter_df_rerank(datastore.qa_fmt_df)
                        qa_rerank_models = _qa_df_rerank[COL_NAME_RERANKING_MODEL].apply(remove_html).unique().tolist()
                        with gr.Row():
                            with gr.Column(scale=1):
                                qa_models_rerank = get_reranking_dropdown(qa_rerank_models)
                                version.change(update_qa_models, version, qa_models_rerank)
                            with gr.Column(scale=1):
                                qa_search_bar_rerank = gr.Textbox(show_label=False, visible=False)
                        qa_df_elem_rerank = get_leaderboard_table(_qa_df_rerank, datastore.qa_types)
                        version.change(update_qa_df_rerank, version, qa_df_elem_rerank)

                        _qa_df_rerank_hidden = filter_df_rerank(datastore.qa_raw_df)
                        qa_df_elem_rerank_hidden = get_leaderboard_table(
                            _qa_df_rerank_hidden, datastore.qa_types, visible=False
                        )
                        version.change(update_qa_hidden_df_rerank, version, qa_df_elem_rerank_hidden)

                        set_listeners(
                            TaskType.qa,
                            qa_df_elem_rerank,
                            qa_df_elem_rerank_hidden,
                            qa_search_bar_rerank,
                            version,
                            domains,
                            langs,
                            qa_models_rerank,
                            show_anonymous,
                            show_rev_ts,
                        )

                        metric.change(
                            update_qa_metric,
                            [
                                metric,
                                domains,
                                langs,
                                qa_models_rerank,
                                qa_search_bar_rerank,
                                show_anonymous,
                                show_rev_ts,
                            ],
                            qa_df_elem_rerank,
                            queue=True,
                        )
            with gr.TabItem("Long Doc", elem_id="long-doc-benchmark-tab-table", id=1):
                with gr.Row():
                    with gr.Column(min_width=320):
                        # select domain
                        with gr.Row():
                            domains = get_domain_dropdown(LongDocBenchmarks[datastore.slug])
                            version.change(update_doc_domains, version, domains)
                        # select language
                        with gr.Row():
                            langs = get_language_dropdown(LongDocBenchmarks[datastore.slug])
                            version.change(update_doc_langs, version, langs)
                    with gr.Column():
                        # select the metric
                        with gr.Row():
                            metric = get_metric_dropdown(METRIC_LIST, DEFAULT_METRIC_LONG_DOC)
                        with gr.Row():
                            show_anonymous = get_anonymous_checkbox()
                        with gr.Row():
                            show_rev_ts = get_revision_and_ts_checkbox()
                with gr.Tabs(elem_classes="tab-buttons"):
                    with gr.TabItem("Retrieval + Reranking", id=20):
                        with gr.Row():
                            with gr.Column():
                                search_bar = get_search_bar()
                            with gr.Column():
                                models = get_reranking_dropdown(datastore.reranking_models)
                                version.change(update_qa_models, version, models)

                        doc_df_elem_ret_rerank = get_leaderboard_table(datastore.doc_fmt_df, datastore.doc_types)

                        version.change(update_doc_df_ret_rerank, version, doc_df_elem_ret_rerank)

                        doc_df_elem_ret_rerank_hidden = get_leaderboard_table(
                            datastore.doc_raw_df, datastore.doc_types, visible=False
                        )

                        version.change(update_doc_hidden_df_ret_rerank, version, doc_df_elem_ret_rerank_hidden)

                        set_listeners(
                            TaskType.long_doc,
                            doc_df_elem_ret_rerank,
                            doc_df_elem_ret_rerank_hidden,
                            search_bar,
                            version,
                            domains,
                            langs,
                            models,
                            show_anonymous,
                            show_rev_ts,
                        )

                        # set metric listener
                        metric.change(
                            update_doc_metric,
                            [
                                metric,
                                domains,
                                langs,
                                models,
                                search_bar,
                                show_anonymous,
                                show_rev_ts,
                            ],
                            doc_df_elem_ret_rerank,
                            queue=True,
                        )
                    with gr.TabItem("Retrieval Only", id=21):
                        with gr.Row():
                            with gr.Column(scale=1):
                                search_bar_ret = get_search_bar()
                            with gr.Column(scale=1):
                                models_ret = get_noreranking_dropdown()
                        _doc_df_ret = filter_df_ret(datastore.doc_fmt_df)
                        doc_df_elem_ret = get_leaderboard_table(_doc_df_ret, datastore.doc_types)
                        version.change(update_doc_df_ret, version, doc_df_elem_ret)

                        _doc_df_ret_hidden = filter_df_ret(datastore.doc_raw_df)
                        doc_df_elem_ret_hidden = get_leaderboard_table(
                            _doc_df_ret_hidden, datastore.doc_types, visible=False
                        )
                        version.change(update_doc_hidden_df_ret, version, doc_df_elem_ret_hidden)

                        set_listeners(
                            TaskType.long_doc,
                            doc_df_elem_ret,
                            doc_df_elem_ret_hidden,
                            search_bar_ret,
                            version,
                            domains,
                            langs,
                            models_ret,
                            show_anonymous,
                            show_rev_ts,
                        )

                        metric.change(
                            update_doc_metric,
                            [
                                metric,
                                domains,
                                langs,
                                models_ret,
                                search_bar_ret,
                                show_anonymous,
                                show_rev_ts,
                            ],
                            doc_df_elem_ret,
                            queue=True,
                        )
                    with gr.TabItem("Reranking Only", id=22):
                        _doc_df_rerank = filter_df_rerank(datastore.doc_fmt_df)
                        doc_rerank_models = (
                            _doc_df_rerank[COL_NAME_RERANKING_MODEL].apply(remove_html).unique().tolist()
                        )
                        with gr.Row():
                            with gr.Column(scale=1):
                                doc_models_rerank = get_reranking_dropdown(doc_rerank_models)
                            with gr.Column(scale=1):
                                doc_search_bar_rerank = gr.Textbox(show_label=False, visible=False)
                        doc_df_elem_rerank = get_leaderboard_table(_doc_df_rerank, datastore.doc_types)
                        version.change(update_doc_df_rerank, version, doc_df_elem_rerank)

                        _doc_df_rerank_hidden = filter_df_rerank(datastore.doc_raw_df)
                        doc_df_elem_rerank_hidden = get_leaderboard_table(
                            _doc_df_rerank_hidden, datastore.doc_types, visible=False
                        )

                        version.change(update_doc_hidden_df_rerank, version, doc_df_elem_rerank_hidden)

                        set_listeners(
                            TaskType.long_doc,
                            doc_df_elem_rerank,
                            doc_df_elem_rerank_hidden,
                            doc_search_bar_rerank,
                            version,
                            domains,
                            langs,
                            doc_models_rerank,
                            show_anonymous,
                            show_rev_ts,
                        )

                        metric.change(
                            update_doc_metric,
                            [
                                metric,
                                domains,
                                langs,
                                doc_models_rerank,
                                doc_search_bar_rerank,
                                show_anonymous,
                                show_rev_ts,
                            ],
                            doc_df_elem_rerank,
                            queue=True,
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
                            label="Reranking Model name", info="Optional", value="NoReranker"
                        )
                    with gr.Column():
                        reranking_model_url = gr.Textbox(label="Reranking Model URL", info="Optional", value="")
                with gr.Row():
                    with gr.Column():
                        benchmark_version = gr.Dropdown(
                            BENCHMARK_VERSION_LIST,
                            value=LATEST_BENCHMARK_VERSION,
                            interactive=True,
                            label="AIR-Bench Version",
                        )
                with gr.Row():
                    upload_button = gr.UploadButton("Click to upload search results", file_count="single")
                with gr.Row():
                    file_output = gr.File()
                with gr.Row():
                    is_anonymous = gr.Checkbox(
                        label="Nope. I want to submit anonymously 🥷",
                        value=False,
                        info="Do you want to shown on the leaderboard by default?",
                    )
                with gr.Row():
                    submit_button = gr.Button("Submit")
                with gr.Row():
                    submission_result = gr.Markdown()
                upload_button.upload(
                    upload_file,
                    [
                        upload_button,
                    ],
                    file_output,
                )
                submit_button.click(
                    submit_results,
                    [
                        file_output,
                        model_name,
                        model_url,
                        reranking_model_name,
                        reranking_model_url,
                        benchmark_version,
                        is_anonymous,
                    ],
                    submission_result,
                    show_progress="hidden",
                )

        with gr.TabItem("📝 About", elem_id="llm-benchmark-tab-table", id=3):
            gr.Markdown(BENCHMARKS_TEXT, elem_classes="markdown-text")

if __name__ == "__main__":
    scheduler = BackgroundScheduler()
    scheduler.add_job(restart_space, "interval", seconds=1800)
    scheduler.start()
    demo.queue(default_concurrency_limit=40)
    demo.launch()
