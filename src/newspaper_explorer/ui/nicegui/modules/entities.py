"""
Entity analysis tab for the NiceGUI interface
"""

import json
from pathlib import Path
from nicegui import ui
import polars as pl
from base64 import b64encode

from newspaper_explorer.ui.nicegui.visualizations.entities import (
    create_entity_timeline_chart,
    create_entity_type_pie_chart,
    get_top_entities,
    create_wordcloud_image,
    get_available_entity_types,
    assign_entity_type_colors,
    get_entity_type_color,
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


async def create_entity_tab(state):
    """
    Create the entities analysis tab

    Args:
        state: AppState instance with filtered data
    """
    ui.label("Entity Analysis").classes("text-h4 q-mb-md")

    # Get available entity result files
    available_files = state.get_available_entity_files()

    # File selector at the top if multiple files available
    if len(available_files) > 0:
        with ui.card().classes("w-full q-mb-md"):
            with ui.row().classes("w-full items-center gap-4 q-pa-md"):
                ui.label("Entity Dataset:").classes("text-subtitle1")

                # Create options dict for select
                file_options = {
                    file_path: display_name for display_name, file_path in available_files
                }

                # Get currently loaded file path or default to first
                current_file = None
                if state.entities_df is not None and available_files:
                    current_file = available_files[0][1]  # Default to first file

                file_select = ui.select(
                    options=file_options,
                    value=current_file,
                    label="Select entity extraction results",
                ).classes("col-grow")

                # Reload button
                async def reload_entities():
                    """Reload entities with selected file"""
                    if file_select.value:
                        state.load_entities(file_path=file_select.value)
                        ui.notify(f"Loaded entity data", type="positive")
                        # Update metadata display
                        metadata = load_result_metadata(file_select.value)
                        metadata_container.clear()
                        with metadata_container:
                            display_metadata(metadata)

                ui.button("Load", icon="refresh", on_click=reload_entities).props("outline")

                # Show info about current dataset
                if state.entities_df is not None:
                    ui.label(f"({len(state.entities_df):,} records)").classes(
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
                        if "threshold" in params:
                            param_parts.append(f"threshold={params['threshold']}")
                        if "temperature" in params:
                            param_parts.append(f"temperature={params['temperature']}")
                        if "labels" in params and isinstance(params["labels"], list):
                            param_parts.append(f"labels: {', '.join(params['labels'])}")

                        if param_parts:
                            ui.label(f"Parameters: {' • '.join(param_parts)}").classes(
                                "text-caption text-grey-6"
                            )

                    # Processing info
                    info_parts = []
                    if "line_count" in metadata:
                        info_parts.append(f"{metadata['line_count']:,} lines processed")
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

    # Get filtered entity data
    df = state.get_filtered_entities()

    # Assign global colors when data is available
    if df is not None and len(df) > 0:
        assign_entity_type_colors(df)

    # Show data status
    if df is not None and len(df) > 0:
        ui.label(
            f"Showing {len(df):,} entity mentions from {state.start_date} to {state.end_date}"
        ).classes("text-caption text-grey-7 q-mb-md")
    else:
        with ui.card().classes("w-full q-mb-md"):
            with ui.row().classes("items-center q-pa-md"):
                ui.icon("info", size="2em").classes("text-blue q-mr-md")
                with ui.column():
                    ui.label("No entity data loaded").classes("text-subtitle1")
                    ui.label(
                        "Load entity extraction results to see visualizations. "
                        "Run entity extraction first: newspaper-explorer analyze entities extract"
                    ).classes("text-caption text-grey-7")
        return

    # Timeline chart
    with ui.card().classes("w-full"):
        ui.label("Entity Mentions Over Time").classes("text-h6 q-mb-sm")

        timeline_fig = create_entity_timeline_chart(df)
        if timeline_fig:
            ui.plotly(timeline_fig).classes("w-full")
        else:
            with ui.column().classes("q-pa-md items-center"):
                ui.icon("timeline", size="3em").classes("text-grey-5")
                ui.label("No data in selected time range")

    # Entity distribution and top entities
    with ui.row().classes("w-full q-mt-md gap-4"):
        # Type distribution (left)
        with ui.card().classes("col").style("height: 600px;"):
            ui.label("Entity Type Distribution").classes("text-h6 q-mb-sm")

            pie_fig = create_entity_type_pie_chart(df)
            if pie_fig:
                ui.plotly(pie_fig).classes("w-full")
            else:
                with ui.column().classes("q-pa-md items-center"):
                    ui.icon("pie_chart", size="3em").classes("text-grey-5")
                    ui.label("No type data available")

        # Top entities (right)
        with ui.card().classes("col").style("height: 600px; overflow-y: auto;"):
            ui.label("Top Entities by Type").classes("text-h6 q-mb-sm")

            # Get available entity types
            entity_types = get_available_entity_types(df)

            if len(entity_types) > 0:
                # Create tabs for each entity type
                with ui.tabs().classes("w-full") as entity_tabs:
                    type_tabs = {}
                    for entity_type in entity_types:
                        type_tabs[entity_type] = ui.tab(entity_type)

                with ui.tab_panels(entity_tabs, value=type_tabs[entity_types[0]]).classes("w-full"):
                    for entity_type in entity_types:
                        with ui.tab_panel(type_tabs[entity_type]):
                            # Get top entities for this type
                            type_df = df.filter(
                                (
                                    pl.col("type") if "type" in df.columns else pl.col("Type")
                                ).str.to_lowercase()
                                == entity_type.lower()
                            )

                            # Group by entity and count
                            entity_col = "Entity" if "Entity" in type_df.columns else "entity"
                            top_for_type = (
                                type_df.group_by(entity_col)
                                .agg(pl.count().alias("Count"))
                                .sort("Count", descending=True)
                                .head(15)
                            )

                            if len(top_for_type) > 0:
                                # Create a compact list display
                                for idx, row in enumerate(top_for_type.iter_rows(named=True), 1):
                                    with ui.row().classes(
                                        "w-full items-center justify-between q-px-sm q-py-xs hover:bg-grey-2"
                                    ):
                                        with ui.row().classes("items-center gap-2"):
                                            ui.label(f"{idx}.").classes("text-grey-6 text-caption")
                                            ui.label(row[entity_col]).classes("text-body2")
                                        ui.badge(f"{row['Count']}", color="primary").props(
                                            "rounded"
                                        )
                            else:
                                with ui.column().classes("q-pa-md items-center"):
                                    ui.icon("info", size="2em").classes("text-grey-5")
                                    ui.label(f"No {entity_type} entities found").classes(
                                        "text-caption"
                                    )
            else:
                with ui.column().classes("q-pa-md items-center"):
                    ui.icon("list", size="3em").classes("text-grey-5")
                    ui.label("No entities found")

    # Word cloud section
    with ui.card().classes("w-full q-mt-md"):
        ui.label("Entity Word Cloud").classes("text-h6 q-mb-sm")

        # Get available entity types
        entity_types = get_available_entity_types(df)

        if entity_types:
            # Entity type selector
            with ui.row().classes("w-full items-center gap-4 q-mb-md"):
                ui.label("Entity Type:").classes("text-subtitle2")
                selected_type = ui.select(
                    options=entity_types,
                    value=entity_types[0] if entity_types else None,
                ).classes("w-48")

                ui.label("Max Words:").classes("text-subtitle2 q-ml-md")
                max_words = (
                    ui.slider(min=20, max=1000, value=200, step=10)
                    .props("label-always")
                    .classes("w-64")
                )

            # Placeholder for word cloud
            wordcloud_container = ui.column().classes("w-full")

            # Track last update to debounce slider changes
            update_timer = None

            def update_wordcloud():
                """Update word cloud based on selected type and max words"""
                nonlocal update_timer

                # Cancel any pending update
                if update_timer is not None:
                    update_timer.active = False

                # Ensure we have a valid type selected
                current_type = selected_type.value
                if not current_type:
                    return

                wordcloud_container.clear()

                with wordcloud_container:
                    # Show loading placeholder with fixed height (800px matches image height at typical screen widths)
                    ui.html(
                        '<div style="width: 100%; min-height: 600px; background: #f5f5f5; '
                        "border-radius: 4px; display: flex; flex-direction: column; "
                        'align-items: center; justify-content: center;">'
                        '<div class="q-spinner q-spinner-mat" style="font-size: 3rem; color: var(--q-primary);"></div>'
                        '<p style="margin-top: 1rem; color: #666; font-size: 0.875rem;">Generating word cloud...</p>'
                        "</div>",
                        sanitize=False,
                    )

                # Generate word cloud in background (simulated async)
                def generate_and_display():
                    img_bytes = create_wordcloud_image(
                        df, current_type, max_words=int(max_words.value)
                    )

                    # Clear and show result
                    wordcloud_container.clear()

                    with wordcloud_container:
                        if img_bytes:
                            # Convert to base64 for display
                            img_b64 = b64encode(img_bytes).decode()
                            ui.html(
                                f'<img src="data:image/png;base64,{img_b64}" '
                                f'style="width: 100%; height: auto; display: block;" '
                                f'alt="Word cloud for {current_type}"/>',
                                sanitize=False,
                            )
                        else:
                            # Error state with same height
                            ui.html(
                                f'<div style="width: 100%; min-height: 600px; background: #f5f5f5; '
                                f"border-radius: 4px; display: flex; flex-direction: column; "
                                f'align-items: center; justify-content: center;">'
                                f'<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" '
                                f'fill="none" stroke="#9e9e9e" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
                                f'<circle cx="12" cy="12" r="10"></circle>'
                                f'<line x1="12" y1="8" x2="12" y2="12"></line>'
                                f'<line x1="12" y1="16" x2="12.01" y2="16"></line>'
                                f"</svg>"
                                f'<p style="margin-top: 1rem; color: #666; font-size: 0.875rem;">No entities found for type: {current_type}</p>'
                                f"</div>",
                                sanitize=False,
                            )

                # Use timer to allow UI to update with loading state first
                update_timer = ui.timer(0.1, generate_and_display, once=True)

            def debounced_update():
                """Debounced update for slider to prevent rapid regeneration"""
                nonlocal update_timer

                # Cancel any pending update
                if update_timer is not None:
                    update_timer.active = False

                # Schedule new update after delay
                update_timer = ui.timer(0.3, update_wordcloud, once=True)

            # Update on change
            selected_type.on_value_change(lambda: update_wordcloud())
            max_words.on_value_change(lambda: debounced_update())

            # Initial render
            update_wordcloud()
        else:
            with ui.column().classes("q-pa-md items-center"):
                ui.icon("error", size="2em").classes("text-grey-5")
                ui.label("No entity types found in data")

    # Entity Browser
    with ui.card().classes("w-full q-mt-md"):
        ui.label("Entity Browser").classes("text-h6 q-mb-sm")

        # Search and filter controls
        with ui.row().classes("w-full items-center gap-4 q-mb-md"):
            search_input = (
                ui.input(placeholder="Search entities...")
                .classes("col-grow")
                .props("outlined dense clearable")
            )

            entity_types = get_available_entity_types(df)
            type_filter = (
                ui.select(options=["All Types"] + entity_types, value="All Types", label="Type")
                .classes("w-48")
                .props("outlined dense")
            )

            ui.button("Search", icon="search").props("color=primary")

        # Results container
        results_container = ui.column().classes("w-full")

        def search_entities():
            """Search and display entities based on filters"""
            results_container.clear()

            search_term = search_input.value or ""
            selected_type = type_filter.value

            # Filter dataframe
            filtered_df = df

            # Filter by type if not "All Types"
            if selected_type != "All Types" and selected_type:
                type_col = "type" if "type" in df.columns else "Type"
                filtered_df = filtered_df.filter(
                    pl.col(type_col).str.to_lowercase() == selected_type.lower()
                )

            # Filter by search term if provided
            if search_term:
                entity_col = "Entity" if "Entity" in filtered_df.columns else "entity"
                filtered_df = filtered_df.filter(
                    pl.col(entity_col).str.to_lowercase().str.contains(search_term.lower())
                )

            # Group by entity and count occurrences
            entity_col = "Entity" if "Entity" in filtered_df.columns else "entity"
            type_col = "type" if "type" in filtered_df.columns else "Type"
            date_col = "date" if "date" in filtered_df.columns else "Date"

            entity_stats = (
                filtered_df.group_by([entity_col, type_col])
                .agg(
                    [
                        pl.count().alias("count"),
                        pl.col(date_col).min().alias("first_seen"),
                        pl.col(date_col).max().alias("last_seen"),
                    ]
                )
                .sort("count", descending=True)
                .head(50)  # Limit to 50 results
            )

            with results_container:
                if len(entity_stats) > 0:
                    ui.label(f"Found {len(entity_stats)} entities (showing top 50)").classes(
                        "text-caption text-grey-7 q-mb-sm"
                    )

                    # Display results as cards (5 per row)
                    with ui.grid(columns=5).classes("w-full gap-2"):
                        for row in entity_stats.iter_rows(named=True):
                            with ui.card().classes("w-full"):
                                with ui.card_section():
                                    # Entity type badge (moved above entity name)
                                    entity_type = row[type_col]
                                    entity_color = get_entity_type_color(entity_type)
                                    with ui.row().classes("w-full justify-start q-mb-xs"):
                                        if entity_color:
                                            ui.html(
                                                f'<span class="q-badge q-badge--outline" '
                                                f'style="background-color: {entity_color}20; '
                                                f"color: {entity_color}; border: 1px solid {entity_color}; "
                                                f'font-size: 0.7rem; padding: 2px 6px;">'
                                                f"{entity_type}</span>",
                                                sanitize=False,
                                            )
                                        else:
                                            ui.badge(entity_type, color="primary").props(
                                                "outline"
                                            ).style("font-size: 0.7rem; padding: 2px 6px;")

                                    # Entity name
                                    ui.label(row[entity_col]).classes(
                                        "text-subtitle1 text-weight-bold q-mb-xs"
                                    )

                                    # Statistics
                                    with ui.row().classes("w-full items-center gap-4 q-mt-sm"):
                                        with ui.column().classes("items-center").style("gap: 0px"):
                                            ui.label(str(row["count"])).classes(
                                                "text-h6 text-primary"
                                            ).style("line-height: 1.2; margin-bottom: 0")
                                            ui.label("mentions").classes(
                                                "text-caption text-grey-7"
                                            ).style("line-height: 1; margin-top: -2px")

                                        with ui.column().classes("col-grow gap-0"):
                                            first = (
                                                row["first_seen"].strftime("%Y-%m-%d")
                                                if row["first_seen"]
                                                else "N/A"
                                            )
                                            last = (
                                                row["last_seen"].strftime("%Y-%m-%d")
                                                if row["last_seen"]
                                                else "N/A"
                                            )
                                            ui.label(f"First: {first}").classes("text-caption")
                                            ui.label(f"Last: {last}").classes("text-caption")

                                    # View details button
                                    def create_detail_dialog(entity_name, entity_type):
                                        def show_details():
                                            # Get all occurrences of this entity
                                            type_col = "type" if "type" in df.columns else "Type"
                                            date_col = "date" if "date" in df.columns else "Date"

                                            entity_occurrences = filtered_df.filter(
                                                pl.col(entity_col) == entity_name
                                            ).sort(date_col)

                                            # Get first and last dates
                                            first_date = (
                                                entity_occurrences[date_col][0].strftime("%Y-%m-%d")
                                                if len(entity_occurrences) > 0
                                                and entity_occurrences[date_col][0]
                                                else "N/A"
                                            )
                                            last_date = (
                                                entity_occurrences[date_col][-1].strftime(
                                                    "%Y-%m-%d"
                                                )
                                                if len(entity_occurrences) > 0
                                                and entity_occurrences[date_col][-1]
                                                else "N/A"
                                            )

                                            with (
                                                ui.dialog() as dialog,
                                                ui.card()
                                                .classes("w-full")
                                                .style("max-width: 1200px; max-height: 80vh;"),
                                            ):
                                                with ui.card_section():
                                                    with ui.row().classes(
                                                        "w-full items-center justify-between"
                                                    ):
                                                        with ui.row().classes("items-center gap-2"):
                                                            ui.label(f"{entity_name}").classes(
                                                                "text-h6"
                                                            )
                                                            # Use assigned color for the badge
                                                            entity_color = get_entity_type_color(
                                                                entity_type
                                                            )
                                                            if entity_color:
                                                                ui.html(
                                                                    f'<span class="q-badge q-badge--outline" '
                                                                    f'style="background-color: {entity_color}20; '
                                                                    f"color: {entity_color}; border: 1px solid {entity_color}; "
                                                                    f'font-size: 0.7rem; padding: 2px 6px;">'
                                                                    f"{entity_type}</span>",
                                                                    sanitize=False,
                                                                )
                                                            else:
                                                                ui.badge(
                                                                    entity_type, color="primary"
                                                                ).props("outline").style(
                                                                    "font-size: 0.7rem; padding: 2px 6px;"
                                                                )
                                                        with ui.column().classes("items-end gap-0"):
                                                            ui.label(
                                                                f"Total mentions: {len(entity_occurrences)}"
                                                            ).classes("text-subtitle2 text-grey-7")
                                                            ui.label(
                                                                f"First: {first_date}  •  Last: {last_date}"
                                                            ).classes("text-caption text-grey-6")

                                                ui.separator()

                                                with (
                                                    ui.card_section()
                                                    .classes("scroll")
                                                    .style("max-height: 60vh; overflow-y: auto;")
                                                ):
                                                    # Display all occurrences as wrapping cards
                                                    if len(entity_occurrences) > 0:
                                                        with (
                                                            ui.row()
                                                            .classes("w-full gap-2")
                                                            .style("flex-wrap: wrap;")
                                                        ):
                                                            for occ in entity_occurrences.iter_rows(
                                                                named=True
                                                            ):
                                                                with (
                                                                    ui.card()
                                                                    .props("flat bordered")
                                                                    .style(
                                                                        "flex: 0 1 auto; min-width: 150px;"
                                                                    )
                                                                ):
                                                                    with ui.card_section().classes(
                                                                        "q-pa-xs"
                                                                    ):
                                                                        # Date
                                                                        date_str = (
                                                                            occ[date_col].strftime(
                                                                                "%Y-%m-%d"
                                                                            )
                                                                            if occ.get(date_col)
                                                                            else "N/A"
                                                                        )
                                                                        ui.label(date_str).classes(
                                                                            "text-caption text-weight-bold text-primary"
                                                                        )

                                                                        # Context if available
                                                                        if (
                                                                            "context" in occ
                                                                            and occ["context"]
                                                                        ):
                                                                            ui.label(
                                                                                occ["context"]
                                                                            ).classes(
                                                                                "text-caption q-mt-xs"
                                                                            ).style(
                                                                                "word-wrap: break-word;"
                                                                            )

                                                                        # Additional fields if available
                                                                        extra_fields = []
                                                                        if (
                                                                            "page_number" in occ
                                                                            and occ["page_number"]
                                                                        ):
                                                                            extra_fields.append(
                                                                                f"Page {occ['page_number']}"
                                                                            )
                                                                        if (
                                                                            "confidence" in occ
                                                                            and occ["confidence"]
                                                                        ):
                                                                            extra_fields.append(
                                                                                f"Confidence: {occ['confidence']:.2f}"
                                                                            )

                                                                        if extra_fields:
                                                                            ui.label(
                                                                                " • ".join(
                                                                                    extra_fields
                                                                                )
                                                                            ).classes(
                                                                                "text-caption text-grey-7 q-mt-xs"
                                                                            )

                                                with ui.card_actions().classes("q-pa-md"):
                                                    ui.button("Close", on_click=dialog.close).props(
                                                        "flat color=primary"
                                                    )

                                            dialog.open()

                                        return show_details

                                    ui.button(
                                        "Details",
                                        icon="info",
                                        on_click=create_detail_dialog(
                                            row[entity_col],
                                            row[type_col],
                                        ),
                                    ).props("flat size=sm").classes("w-full q-mt-sm")
                else:
                    with ui.column().classes("q-pa-xl items-center"):
                        ui.icon("search_off", size="3em").classes("text-grey-5")
                        ui.label("No entities found").classes("text-subtitle1 text-grey-7")
                        ui.label("Try adjusting your search filters").classes(
                            "text-caption text-grey-6"
                        )

        # Connect search triggers
        search_input.on("keydown.enter", search_entities)
        type_filter.on_value_change(search_entities)

        # Initial search (show all)
        search_entities()
