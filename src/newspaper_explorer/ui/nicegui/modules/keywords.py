"""
Keywords tab for the NiceGUI interface - Keyword extraction and analysis
"""

import json
from pathlib import Path
from nicegui import ui
import polars as pl
from base64 import b64encode

from newspaper_explorer.ui.nicegui.visualizations.keywords import (
    create_keyword_frequency_chart,
    create_keyword_score_distribution,
    create_keywords_per_document_chart,
    create_keyword_wordcloud,
    create_keyword_timeline,
    get_keyword_statistics,
    get_top_keywords,
    get_documents_for_keyword,
    get_keyword_cooccurrence,
)


def load_result_metadata(result_file_path: str) -> dict:
    """
    Load metadata.json from the same directory as the result file.

    Args:
        result_file_path: Path to the result file (parquet or csv)

    Returns:
        Dictionary with metadata, or empty dict if not found
    """
    result_path = Path(result_file_path)
    metadata_path = result_path.parent / "metadata.json"

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading metadata from {metadata_path}: {e}")
            return {}
    return {}


async def create_keywords_tab(state):
    """
    Create the keywords tab for keyword extraction and analysis

    Args:
        state: AppState instance
    """
    ui.label("Keyword Analysis").classes("text-h4 q-mb-md")

    # Get available keyword result files
    available_files = state.get_available_keyword_files()

    # File selector at the top if multiple files available
    if len(available_files) > 0:
        with ui.card().classes("w-full q-mb-md"):
            with ui.row().classes("w-full items-center gap-4 q-pa-md"):
                ui.label("Keyword Dataset:").classes("text-subtitle1")

                # Create options dict for select
                file_options = {
                    file_path: display_name for display_name, file_path in available_files
                }

                # Get currently loaded file path or default to first
                current_file = None
                if state.keywords_df is not None and available_files:
                    current_file = available_files[0][1]  # Default to first file

                file_select = ui.select(
                    options=file_options,
                    value=current_file,
                    label="Select keyword extraction results",
                ).classes("col-grow")

                # Reload button
                async def reload_keywords():
                    """Reload keywords with selected file"""
                    if file_select.value:
                        state.load_keywords(file_path=file_select.value)
                        ui.notify(f"Loaded keyword data", type="positive")
                        # Update metadata display
                        metadata = load_result_metadata(file_select.value)
                        metadata_container.clear()
                        with metadata_container:
                            display_metadata(metadata)

                ui.button("Load", icon="refresh", on_click=reload_keywords).props("outline")

                # Show info about current dataset
                if state.keywords_df is not None:
                    ui.label(f"({len(state.keywords_df):,} documents)").classes(
                        "text-caption text-grey-7"
                    )

            # Metadata display section
            metadata_container = ui.row().classes("w-full q-px-md q-pb-md")

            def display_metadata(metadata: dict):
                """Display metadata information"""
                if not metadata:
                    return

                with ui.column().classes("w-full gap-1"):
                    # Method info
                    if "method_type" in metadata and "model_name" in metadata:
                        method_text = f"Method: {metadata['method_type']}"
                        if metadata["model_name"]:
                            method_text += f" ({metadata['model_name']})"
                        ui.label(method_text).classes("text-caption text-grey-7")

                    # Parameters summary
                    if "parameters" in metadata and metadata["parameters"]:
                        params = metadata["parameters"]
                        param_parts = []

                        # Show key parameters
                        if "top_n" in params:
                            param_parts.append(f"top_n={params['top_n']}")
                        if "diversity" in params:
                            param_parts.append(f"diversity={params['diversity']}")
                        if "ngram_range" in params:
                            param_parts.append(f"ngrams={params['ngram_range']}")

                        if param_parts:
                            ui.label(f"Parameters: {' • '.join(param_parts)}").classes(
                                "text-caption text-grey-6"
                            )

                    # Processing info
                    info_parts = []
                    if "line_count" in metadata:
                        info_parts.append(f"{metadata['line_count']:,} documents processed")
                    if "duration_seconds" in metadata:
                        duration = metadata["duration_seconds"]
                        if duration < 60:
                            info_parts.append(f"{duration:.1f}s")
                        else:
                            mins = int(duration // 60)
                            secs = int(duration % 60)
                            info_parts.append(f"{mins}m {secs}s")
                    if "created_at" in metadata:
                        # Format timestamp nicely
                        from datetime import datetime

                        try:
                            created = datetime.fromisoformat(
                                metadata["created_at"].replace("Z", "+00:00")
                            )
                            info_parts.append(f"created {created.strftime('%Y-%m-%d %H:%M')}")
                        except:
                            pass

                    if info_parts:
                        ui.label(" • ".join(info_parts)).classes("text-caption text-grey-6")

            # Load and display metadata for current file
            if file_select.value:
                metadata = load_result_metadata(file_select.value)
                with metadata_container:
                    display_metadata(metadata)

    # Get filtered keyword data
    df = state.get_filtered_keywords()

    # Show data status
    if df is not None and len(df) > 0:
        stats = get_keyword_statistics(df)
        if stats:
            ui.label(
                f"Showing {stats['documents_with_keywords']:,} documents with keywords "
                f"({stats['total_keywords']:,} total keywords, {stats['unique_keywords']:,} unique)"
            ).classes("text-caption text-grey-7 q-mb-md")
    else:
        with ui.card().classes("w-full q-mb-md"):
            with ui.row().classes("items-center q-pa-md"):
                ui.icon("info", size="2em").classes("text-blue q-mr-md")
                with ui.column():
                    ui.label("No keyword data loaded").classes("text-subtitle1")
                    ui.label(
                        "Load keyword extraction results to see visualizations. "
                        "Run keyword extraction first: newspaper-explorer analyze keywords extract"
                    ).classes("text-caption text-grey-7")
        return

    # Statistics cards
    stats = get_keyword_statistics(df)
    if stats:
        with ui.row().classes("w-full gap-4 q-mb-md"):
            # Total keywords card
            with ui.card().classes("col"):
                with ui.card_section().classes("text-center"):
                    ui.label("Total Keywords").classes("text-caption text-grey-7")
                    ui.label(f"{stats['total_keywords']:,}").classes("text-h5 text-primary")

            # Unique keywords card
            with ui.card().classes("col"):
                with ui.card_section().classes("text-center"):
                    ui.label("Unique Keywords").classes("text-caption text-grey-7")
                    ui.label(f"{stats['unique_keywords']:,}").classes("text-h5 text-secondary")

            # Avg per document card
            with ui.card().classes("col"):
                with ui.card_section().classes("text-center"):
                    ui.label("Avg per Document").classes("text-caption text-grey-7")
                    ui.label(f"{stats['avg_keywords_per_doc']:.1f}").classes("text-h5 text-accent")

            # Avg score card
            with ui.card().classes("col"):
                with ui.card_section().classes("text-center"):
                    ui.label("Avg Relevance Score").classes("text-caption text-grey-7")
                    ui.label(f"{stats['avg_score']:.3f}").classes("text-h5 text-positive")

    # Top keywords frequency chart
    with ui.card().classes("w-full q-mt-md"):
        ui.label("Most Frequent Keywords").classes("text-h6 q-mb-sm")

        # Controls for chart
        with ui.row().classes("w-full items-center gap-4 q-mb-md"):
            ui.label("Number of keywords:").classes("text-subtitle2")
            top_n_slider = (
                ui.slider(min=10, max=50, value=20, step=5).props("label-always").classes("w-64")
            )

            async def update_frequency_chart():
                """Update the frequency chart with new top_n value"""
                freq_fig = create_keyword_frequency_chart(df, top_n=int(top_n_slider.value))
                if freq_fig:
                    freq_chart.update_figure(freq_fig)

            ui.button("Update", icon="refresh", on_click=update_frequency_chart).props(
                "outline dense"
            )

        # Initial chart
        freq_fig = create_keyword_frequency_chart(df, top_n=20)
        if freq_fig:
            freq_chart = ui.plotly(freq_fig).classes("w-full")
        else:
            with ui.column().classes("q-pa-md items-center"):
                ui.icon("bar_chart", size="3em").classes("text-grey-5")
                ui.label("No keyword data available")

    # Distribution charts
    with ui.row().classes("w-full q-mt-md gap-4"):
        # Score distribution (left)
        with ui.card().classes("col"):
            ui.label("Keyword Score Distribution").classes("text-h6 q-mb-sm")

            score_fig = create_keyword_score_distribution(df)
            if score_fig:
                ui.plotly(score_fig).classes("w-full")
            else:
                with ui.column().classes("q-pa-md items-center"):
                    ui.icon("timeline", size="3em").classes("text-grey-5")
                    ui.label("No score data available")

        # Keywords per document (right)
        with ui.card().classes("col"):
            ui.label("Keywords per Document").classes("text-h6 q-mb-sm")

            doc_fig = create_keywords_per_document_chart(df)
            if doc_fig:
                ui.plotly(doc_fig).classes("w-full")
            else:
                with ui.column().classes("q-pa-md items-center"):
                    ui.icon("description", size="3em").classes("text-grey-5")
                    ui.label("No document data available")

    # Word cloud section
    with ui.card().classes("w-full q-mt-md"):
        ui.label("Keyword Word Cloud").classes("text-h6 q-mb-sm")

        # Controls for word cloud
        with ui.row().classes("w-full items-center gap-4 q-mb-md"):
            ui.label("Max Words:").classes("text-subtitle2")
            max_words = (
                ui.slider(min=50, max=500, value=200, step=50).props("label-always").classes("w-64")
            )

            async def update_wordcloud():
                """Update the word cloud with new parameters"""
                img_bytes = create_keyword_wordcloud(df, max_words=int(max_words.value))
                if img_bytes:
                    b64_img = b64encode(img_bytes).decode()
                    wordcloud_img.set_source(f"data:image/png;base64,{b64_img}")
                    ui.notify("Word cloud updated", type="positive")

            ui.button("Update", icon="refresh", on_click=update_wordcloud).props("outline dense")

        # Initial word cloud
        img_bytes = create_keyword_wordcloud(df, max_words=200)
        if img_bytes:
            b64_img = b64encode(img_bytes).decode()
            wordcloud_img = (
                ui.image(f"data:image/png;base64,{b64_img}")
                .classes("w-full")
                .style("max-height: 500px; object-fit: contain;")
            )
        else:
            with ui.column().classes("q-pa-md items-center"):
                ui.icon("cloud", size="3em").classes("text-grey-5")
                ui.label("No data for word cloud")

    # Keyword Timeline
    with ui.card().classes("w-full q-mt-md"):
        ui.label("Keyword Timeline").classes("text-h6 q-mb-sm")
        ui.label("Track keyword frequency over time").classes("text-caption text-grey-7 q-mb-md")

        # Controls
        with ui.row().classes("w-full items-center gap-4 q-mb-md"):
            ui.label("Number of keywords:").classes("text-subtitle2")
            timeline_n_slider = (
                ui.slider(min=3, max=10, value=5, step=1).props("label-always").classes("w-48")
            )

            async def update_timeline():
                """Update the timeline with selected number of keywords"""
                timeline_fig = create_keyword_timeline(df, top_n=int(timeline_n_slider.value))
                if timeline_fig:
                    timeline_chart.update_figure(timeline_fig)
                    ui.notify("Timeline updated", type="positive")

            ui.button("Update", icon="refresh", on_click=update_timeline).props("outline dense")

        # Initial timeline
        timeline_fig = create_keyword_timeline(df, top_n=5)
        if timeline_fig:
            timeline_chart = ui.plotly(timeline_fig).classes("w-full")
        else:
            with ui.column().classes("q-pa-md items-center"):
                ui.icon("timeline", size="3em").classes("text-grey-5")
                ui.label("Timeline not available (doc_ids may not contain date information)")

    # Keyword Explorer - Interactive keyword search
    with ui.card().classes("w-full q-mt-md"):
        ui.label("Keyword Explorer").classes("text-h6 q-mb-sm")
        ui.label("Explore documents and relationships for specific keywords").classes(
            "text-caption text-grey-7 q-mb-md"
        )

        # Get top keywords for autocomplete
        top_kw_for_search = get_top_keywords(df, n=100)
        keyword_options = []
        if top_kw_for_search is not None:
            keyword_options = top_kw_for_search["keywords"].to_list()

        # Keyword search input
        with ui.row().classes("w-full items-center gap-4 q-mb-md"):
            keyword_input = ui.select(
                options=keyword_options,
                label="Select a keyword to explore",
                with_input=True,
            ).classes("col-grow")

            async def explore_keyword():
                """Show detailed info for selected keyword"""
                if not keyword_input.value:
                    ui.notify("Please select a keyword", type="warning")
                    return

                keyword = keyword_input.value

                # Get documents containing this keyword
                docs_df = get_documents_for_keyword(df, keyword, limit=20)

                # Get co-occurring keywords
                cooccur_df = get_keyword_cooccurrence(df, keyword, top_n=10)

                # Clear previous results
                results_container.clear()

                with results_container:
                    with ui.row().classes("w-full gap-4"):
                        # Documents with this keyword
                        with ui.card().classes("col").style("height: 450px;"):
                            ui.label(f"Documents with '{keyword}'").classes(
                                "text-subtitle1 q-mb-sm"
                            )

                            if docs_df is not None and len(docs_df) > 0:
                                # Show count
                                ui.label(f"Showing top {len(docs_df)} documents").classes(
                                    "text-caption text-grey-7 q-mb-sm"
                                )

                                # Create scrollable list
                                with ui.scroll_area().classes("w-full").style("height: 300px;"):
                                    for row in docs_df.iter_rows(named=True):
                                        with ui.row().classes(
                                            "w-full items-center justify-between q-px-sm q-py-xs hover:bg-grey-2"
                                        ):
                                            with ui.column().classes("col-grow"):
                                                ui.label(row["doc_id"]).classes("text-body2")
                                                if row.get("date"):
                                                    ui.label(f"Date: {row['date']}").classes(
                                                        "text-caption text-grey-7"
                                                    )
                                            ui.badge(f"Score: {row['score']:.3f}").props("rounded")
                            else:
                                with ui.column().classes("q-pa-md items-center"):
                                    ui.icon("description", size="2em").classes("text-grey-5")
                                    ui.label("No documents found")

                        # Co-occurring keywords
                        with ui.card().classes("col").style("height: 450px;"):
                            ui.label(f"Keywords co-occurring with '{keyword}'").classes(
                                "text-subtitle1 q-mb-sm"
                            )

                            if cooccur_df is not None and len(cooccur_df) > 0:
                                # Show as list with counts
                                with ui.scroll_area().classes("w-full").style("height: 300px;"):
                                    for idx, row in enumerate(cooccur_df.iter_rows(named=True), 1):
                                        with ui.row().classes(
                                            "w-full items-center justify-between q-px-sm q-py-xs hover:bg-grey-2"
                                        ):
                                            with ui.row().classes("items-center gap-2"):
                                                ui.label(f"{idx}.").classes(
                                                    "text-grey-6 text-caption"
                                                )
                                                ui.label(row["cooccurring_keyword"]).classes(
                                                    "text-body2"
                                                )
                                            ui.badge(f"{row['count']}", color="secondary").props(
                                                "rounded"
                                            )
                            else:
                                with ui.column().classes("q-pa-md items-center"):
                                    ui.icon("link", size="2em").classes("text-grey-5")
                                    ui.label("No co-occurring keywords found")

            ui.button("Explore", icon="search", on_click=explore_keyword).props("outline")

        # Results container
        results_container = ui.column().classes("w-full q-mt-md")

    # Top keywords table
    with ui.card().classes("w-full q-mt-md"):
        ui.label("Top Keywords Table").classes("text-h6 q-mb-sm")

        top_keywords = get_top_keywords(df, n=50)
        if top_keywords is not None and len(top_keywords) > 0:
            # Convert to pandas for display
            top_keywords_pd = top_keywords.to_pandas()

            # Format numeric columns
            top_keywords_pd["avg_score"] = top_keywords_pd["avg_score"].map("{:.3f}".format)
            top_keywords_pd["min_score"] = top_keywords_pd["min_score"].map("{:.3f}".format)
            top_keywords_pd["max_score"] = top_keywords_pd["max_score"].map("{:.3f}".format)

            # Create table
            ui.table(
                columns=[
                    {"name": "keywords", "label": "Keyword", "field": "keywords", "align": "left"},
                    {"name": "count", "label": "Frequency", "field": "count", "align": "right"},
                    {
                        "name": "avg_score",
                        "label": "Avg Score",
                        "field": "avg_score",
                        "align": "right",
                    },
                    {
                        "name": "min_score",
                        "label": "Min Score",
                        "field": "min_score",
                        "align": "right",
                    },
                    {
                        "name": "max_score",
                        "label": "Max Score",
                        "field": "max_score",
                        "align": "right",
                    },
                ],
                rows=top_keywords_pd.to_dict("records"),
                row_key="keywords",
            ).classes("w-full").props("dense flat bordered")
        else:
            with ui.column().classes("q-pa-md items-center"):
                ui.icon("table_chart", size="3em").classes("text-grey-5")
                ui.label("No keywords to display")
