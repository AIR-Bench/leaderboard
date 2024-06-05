from src.utils import update_table, update_table_long_doc


def set_listeners(
        task,
        displayed_leaderboard,
        hidden_leaderboard,
        search_bar,
        selected_domains,
        selected_langs,
        selected_rerankings,
        show_anonymous,
        show_revision_and_timestamp,

):
    if task == "qa":
        update_table_func = update_table
    elif task == "long-doc":
        update_table_func = update_table_long_doc
    else:
        raise NotImplementedError
    # Set search_bar listener
    search_bar.submit(
        update_table_func,
        [
            hidden_leaderboard,  #  hidden_leaderboard_table_for_search,
            selected_domains,
            selected_langs,
            selected_rerankings,
            search_bar,
            show_anonymous,
        ],
        displayed_leaderboard
    )

    # Set column-wise listener
    for selector in [
        selected_domains, selected_langs, show_anonymous, show_revision_and_timestamp, selected_rerankings
    ]:
        selector.change(
            update_table_func,
            [
                hidden_leaderboard,
                selected_domains,
                selected_langs,
                selected_rerankings,
                search_bar,
                show_anonymous,
                show_revision_and_timestamp
            ],
            displayed_leaderboard,
            queue=True,
        )
