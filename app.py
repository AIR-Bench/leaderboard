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
from src.utils import update_table, update_metric, update_table_long_doc, upload_file, get_default_cols, submit_results, clear_reranking_selections


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


def update_table_without_ranking(
        hidden_df,
        domains,
        langs,
        reranking_query,
        query,
        show_anonymous,
        show_revision_and_timestamp,
):
    return update_table(hidden_df, domains, langs, reranking_query, query, show_anonymous, reset_ranking=False, show_revision_and_timestamp=show_revision_and_timestamp)


def update_table_without_ranking_long_doc(
        hidden_df,
        domains,
        langs,
        reranking_query,
        query,
        show_anonymous,
        show_revision_and_timestamp,
):
    return update_table_long_doc(hidden_df, domains, langs, reranking_query, query, show_anonymous, reset_ranking=False, show_revision_and_timestamp=show_revision_and_timestamp)


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
                        selected_version = gr.Dropdown(
                            choices=["AIR-Bench_24.04",],
                            value="AIR-Bench_24.04",
                            label="Select the version of AIR-Bench",
                            interactive = True
                        )
                    with gr.Row():
                        search_bar = gr.Textbox(
                            placeholder=" 🔍 Search for retrieval methods (separate multiple queries with `;`) and press ENTER...",
                            show_label=False,
                            elem_id="search-bar",
                            info="Search the retrieval methods"
                        )
                    # select reranking model
                    reranking_models = sorted(list(frozenset([eval_result.reranking_model for eval_result in raw_data])))
                    with gr.Row():
                        selected_rerankings = gr.Dropdown(
                            choices=reranking_models,
                            # value=reranking_models,
                            label="Select the reranking models",
                            elem_id="reranking-select",
                            interactive=True,
                            multiselect=True
                        )
                    with gr.Row():
                        select_noreranker_only_btn = gr.ClearButton(
                            selected_rerankings,
                            value="Only show results without ranking models",
                        )

                with gr.Column(min_width=320):
                    # select the metric
                    selected_metric = gr.Dropdown(
                        choices=METRIC_LIST,
                        value=DEFAULT_METRIC,
                        label="Select the metric",
                        interactive=True,
                        elem_id="metric-select",
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
                        selected_langs = gr.Dropdown(
                            choices=LANG_COLS_QA,
                            value=LANG_COLS_QA,
                            label="Select the languages",
                            elem_id="language-column-select",
                            multiselect=True,
                            interactive=True
                        )
                    with gr.Row():
                        show_anonymous = gr.Checkbox(
                            label="Show anonymous submissions",
                            value=False,
                            info="The anonymous submissions might have invalid model information."
                        )
                    with gr.Row():
                        show_revision_and_timestamp = gr.Checkbox(
                            label="Show submission details",
                            value=False,
                            info="Show the revision and timestamp information of submissions"
                        )

            leaderboard_table = gr.components.Dataframe(
                value=leaderboard_df_qa,
                datatype=types_qa,
                elem_id="leaderboard-table",
                interactive=False,
                visible=True,
            )

            # Dummy leaderboard for handling the case when the user uses backspace key
            hidden_leaderboard_table_for_search = gr.components.Dataframe(
                value=original_df_qa,
                datatype=types_qa,
                visible=False,
            )

            # Set search_bar listener
            search_bar.submit(
                update_table_without_ranking,
                [
                    hidden_leaderboard_table_for_search,
                    selected_domains,
                    selected_langs,
                    selected_rerankings,
                    search_bar,
                    show_anonymous,
                ],
                leaderboard_table,
            )

            for selector in [show_revision_and_timestamp, selected_rerankings]:
                selector.change(
                    update_table_without_ranking,
                    [
                        hidden_leaderboard_table_for_search,
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

            # Set column-wise listener
            for selector in [
                selected_domains, selected_langs, show_anonymous
            ]:
                selector.change(
                    update_table,
                    [
                        hidden_leaderboard_table_for_search,
                        selected_domains,
                        selected_langs,
                        selected_rerankings,
                        search_bar,
                        show_anonymous,
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
                    show_anonymous,
                ],
                leaderboard_table,
                queue=True
            )

            select_noreranker_only_btn.click(
                clear_reranking_selections,
                outputs=selected_rerankings
            )

        with gr.TabItem("Long Doc", elem_id="long-doc-benchmark-tab-table", id=1):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        selected_version = gr.Dropdown(
                            choices=["AIR-Bench_24.04",],
                            value="AIR-Bench_24.04",
                            label="Select the version of AIR-Bench",
                            interactive=True
                        )
                    with gr.Row():
                        search_bar = gr.Textbox(
                            info="Search the retrieval methods",
                            placeholder=" 🔍 Search for retrieval methods (separate multiple queries with `;`) and press ENTER...",
                            show_label=False,
                            elem_id="search-bar-long-doc",
                        )
                    # select reranking model
                    reranking_models = list(frozenset([eval_result.reranking_model for eval_result in raw_data]))
                    with gr.Row():
                        selected_rerankings = gr.Dropdown(
                            choices=reranking_models,
                            # value=reranking_models,
                            label="Select the reranking models",
                            elem_id="reranking-select-long-doc",
                            interactive=True,
                            multiselect=True,
                        )
                    with gr.Row():
                        select_noreranker_only_btn = gr.ClearButton(
                            selected_rerankings,
                            value="Only show results without ranking models",
                        )
                with gr.Column(min_width=320):
                    # select the metric
                    with gr.Row():
                        selected_metric = gr.Dropdown(
                            choices=METRIC_LIST,
                            value=DEFAULT_METRIC,
                            label="Select the metric",
                            interactive=True,
                            elem_id="metric-select-long-doc",
                        )
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
                        selected_langs = gr.Dropdown(
                            choices=LANG_COLS_LONG_DOC,
                            value=LANG_COLS_LONG_DOC,
                            label="Select the languages",
                            elem_id="language-column-select-long-doc",
                            multiselect=True,
                            interactive=True
                        )
                    with gr.Row():
                        show_anonymous = gr.Checkbox(
                            label="Show anonymous submissions",
                            value=False,
                            info="The anonymous submissions might have invalid model information."
                        )
                    with gr.Row():
                        show_revision_and_timestamp = gr.Checkbox(
                            label="Show submission details",
                            value=False,
                            info="Show the revision and timestamp information of submissions"
                        )

            leaderboard_table_long_doc = gr.components.Dataframe(
                value=leaderboard_df_long_doc,
                datatype=types_long_doc,
                elem_id="leaderboard-table-long-doc",
                interactive=False,
                visible=True,
            )

            # Dummy leaderboard for handling the case when the user uses backspace key
            hidden_leaderboard_table_for_search = gr.components.Dataframe(
                value=original_df_long_doc,
                datatype=types_long_doc,
                visible=False,
            )

            # Set search_bar listener
            search_bar.submit(
                update_table_without_ranking_long_doc,
                [
                    hidden_leaderboard_table_for_search,
                    selected_domains,
                    selected_langs,
                    selected_rerankings,
                    search_bar,
                    show_anonymous,
                    show_revision_and_timestamp
                ],
                leaderboard_table_long_doc,
            )

            for selector in [show_revision_and_timestamp, selected_rerankings]:
                selector.change(
                    update_table_without_ranking_long_doc,
                    [
                        hidden_leaderboard_table_for_search,
                        selected_domains,
                        selected_langs,
                        selected_rerankings,
                        search_bar,
                        show_anonymous,
                        show_revision_and_timestamp
                    ],
                    leaderboard_table_long_doc,
                    queue=True,
                )

            # Set column-wise listener
            for selector in [
                selected_domains, selected_langs, show_anonymous
            ]:
                selector.change(
                    update_table_long_doc,
                    [
                        hidden_leaderboard_table_for_search,
                        selected_domains,
                        selected_langs,
                        selected_rerankings,
                        search_bar,
                        show_anonymous,
                        show_revision_and_timestamp
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
                    show_anonymous,
                    show_revision_and_timestamp
                ],
                leaderboard_table_long_doc,
                queue=True
            )

            select_noreranker_only_btn.click(
                clear_reranking_selections,
                outputs=selected_rerankings
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
