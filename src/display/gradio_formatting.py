import gradio as gr


def get_version_dropdown():
    return gr.Dropdown(
        choices=["AIR-Bench_24.04", ],
        value="AIR-Bench_24.04",
        label="Select the version of AIR-Bench",
        interactive=True
    )


def get_search_bar():
    return gr.Textbox(
        placeholder=" 🔍 Search for retrieval methods (separate multiple queries with `;`) and press ENTER...",
        show_label=False,
        # elem_id="search-bar",
        info="Search the retrieval methods"
    )


def get_reranking_dropdown(model_list):
    return gr.Dropdown(
        choices=model_list,
        label="Select the reranking models",
        # elem_id="reranking-select",
        interactive=True,
        multiselect=True
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
        # elem_id="metric-select-long-doc",
    )


def get_domain_dropdown(domain_list, default_domains):
    return gr.CheckboxGroup(
        choices=domain_list,
        value=default_domains,
        label="Select the domains",
        # elem_id="domain-column-select",
        interactive=True,
    )


def get_language_dropdown(language_list, default_languages):
    return gr.Dropdown(
        choices=language_list,
        value=language_list,
        label="Select the languages",
        # elem_id="language-column-select",
        multiselect=True,
        interactive=True
    )


def get_anonymous_checkbox():
    return gr.Checkbox(
        label="Show anonymous submissions",
        value=False,
        info="The anonymous submissions might have invalid model information."
    )


def get_revision_and_ts_checkbox():
    return gr.Checkbox(
        label="Show submission details",
        value=False,
        info="Show the revision and timestamp information of submissions"
    )


def get_leaderboard_table(df, datatype, visible=True):
    return gr.components.Dataframe(
                value=df,
                datatype=datatype,
                elem_id="leaderboard-table",
                interactive=False,
                visible=visible,
            )
