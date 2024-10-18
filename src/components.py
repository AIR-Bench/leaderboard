import gradio as gr

from src.envs import BENCHMARK_VERSION_LIST, LATEST_BENCHMARK_VERSION


def get_version_dropdown():
    return gr.Dropdown(
        choices=BENCHMARK_VERSION_LIST,
        value=LATEST_BENCHMARK_VERSION,
        label="Select the version of AIR-Bench",
        interactive=True,
    )


def get_search_bar():
    return gr.Textbox(
        placeholder=" 🔍 Search for retrieval methods (separate multiple queries with `;`) and press ENTER...",
        show_label=False,
        info="Search the retrieval methods",
    )


def get_reranking_dropdown(model_list):
    return gr.Dropdown(choices=model_list, label="Select the reranking models", interactive=True, multiselect=True)


def get_noreranking_dropdown():
    return gr.Dropdown(
        choices=[
            "NoReranker",
        ],
        value=[
            "NoReranker",
        ],
        interactive=False,
        multiselect=True,
        visible=False,
    )


def get_noreranker_button():
    return gr.Button(
        value="Only show results without ranking models",
    )


def get_metric_dropdown(metric_list, default_metrics):
    return gr.Dropdown(
        choices=metric_list,
        value=default_metrics,
        label="Select the metric",
        interactive=True,
    )


def get_domain_dropdown(benchmarks, default_domains=None):
    domain_list = list(frozenset([c.value.domain for c in list(benchmarks.value)]))
    if default_domains is None:
        default_domains = domain_list
    return gr.CheckboxGroup(
        choices=domain_list,
        value=default_domains,
        label="Select the domains",
        interactive=True,
    )


def get_language_dropdown(benchmarks, default_languages=None):
    language_list = list(frozenset([c.value.lang for c in list(benchmarks.value)]))
    if default_languages is None:
        default_languages = language_list
    return gr.Dropdown(
        choices=language_list,
        value=default_languages,
        label="Select the languages",
        multiselect=True,
        interactive=True,
    )


def get_anonymous_checkbox():
    return gr.Checkbox(
        label="Show anonymous submissions",
        value=False,
        info="The anonymous submissions might have invalid model information.",
    )


def get_revision_and_ts_checkbox():
    return gr.Checkbox(
        label="Show submission details", value=False, info="Show the revision and timestamp information of submissions"
    )


def get_leaderboard_table(df, datatype, visible=True):
    return gr.components.Dataframe(
        value=df,
        datatype=datatype,
        elem_id="leaderboard-table",
        interactive=False,
        visible=visible,
    )
