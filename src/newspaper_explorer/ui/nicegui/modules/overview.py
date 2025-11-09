"""
Overview tab for the NiceGUI interface - Landing page with source info and statistics
"""

import polars as pl
from nicegui import ui


async def create_overview_tab(state):
    """
    Create the overview landing page tab

    Args:
        state: AppState instance
    """

    # Hero section
    with ui.row().classes("w-full items-center justify-center q-mb-md"):
        with ui.column().classes("items-center"):
            ui.label("Newspaper Explorer").classes("text-h3 q-mb-md")

    # Statistics overview for selected source
    if state.selected_source:
        # Uniform card style for all statistics
        card_style = """
            border: 1px solid #e0e0e0;
            border-radius: 4px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.05);
            min-height: 140px;
        """

        with ui.row().classes("w-full q-gutter-md q-mb-md").style("align-items: stretch;"):
            # Source selector card
            with ui.card().classes("flex-1").style(card_style):
                with (
                    ui.card_section()
                    .classes("q-pa-md")
                    .style(
                        "height: 100%; display: flex; flex-direction: column; justify-content: flex-start;"
                    )
                ):
                    with ui.row().classes("items-center q-mb-md").style("gap: 8px;"):
                        ui.icon("library_books", size="1.2em").classes("text-grey-7")
                        ui.label("Collection").classes("text-caption text-grey-7").style(
                            "font-weight: 500;"
                        )

                    # Create dropdown options
                    source_options = {}
                    for src in state.available_sources:
                        config = state.get_source_config(src)
                        label = (
                            config.metadata.newspaper_title if config and config.metadata else src
                        )
                        source_options[src] = label

                    def on_source_change(e):
                        selected = e.value
                        if selected:
                            state.load_source(selected)
                            ui.notify(f"Loaded: {source_options[selected]}", type="positive")
                            ui.navigate.reload()

                    ui.select(
                        options=source_options,
                        value=state.selected_source,
                        on_change=on_source_change,
                    ).classes("w-full q-mb-xs").props("outlined dense")

                    # Number of available sources as subtitle
                    ui.label(f"{len(state.available_sources)} available").classes(
                        "text-caption text-grey-6"
                    )

            # Get real statistics from DuckDB
            stats = state.get_comprehensive_stats()

            # Get image statistics from index if available
            image_stats = None
            if state.image_indexer:
                image_stats = state.image_indexer.get_stats()

            # Total documents card
            with ui.card().classes("flex-1").style(card_style):
                with (
                    ui.card_section()
                    .classes("q-pa-md")
                    .style(
                        "height: 100%; display: flex; flex-direction: column; justify-content: flex-start;"
                    )
                ):
                    with ui.row().classes("items-center q-mb-md").style("gap: 8px;"):
                        ui.icon("description", size="1.2em").classes("text-grey-7")
                        ui.label("Text Lines").classes("text-caption text-grey-7").style(
                            "font-weight: 500;"
                        )
                    ui.label(f"{stats['total_lines']:,}").classes(
                        "text-h4 text-grey-9 q-mb-xs"
                    ).style("line-height: 1;")
                    # Show text blocks count if available
                    if stats.get("total_blocks", 0) > 0:
                        ui.label(f"{stats['total_blocks']:,} text blocks").classes(
                            "text-caption text-grey-6"
                        )

            # Date range card
            with ui.card().classes("flex-1").style(card_style):
                with (
                    ui.card_section()
                    .classes("q-pa-md")
                    .style(
                        "height: 100%; display: flex; flex-direction: column; justify-content: flex-start;"
                    )
                ):
                    with ui.row().classes("items-center q-mb-md").style("gap: 8px;"):
                        ui.icon("calendar_today", size="1.2em").classes("text-grey-7")
                        ui.label("Date Range").classes("text-caption text-grey-7").style(
                            "font-weight: 500;"
                        )
                    ui.label(f"{stats['years']} years").classes(
                        "text-h4 text-grey-9 q-mb-xs"
                    ).style("line-height: 1;")
                    ui.label(f"{stats['min_date']} – {stats['max_date']}").classes(
                        "text-caption text-grey-6"
                    )

            # Total files card
            with ui.card().classes("flex-1").style(card_style):
                with (
                    ui.card_section()
                    .classes("q-pa-md")
                    .style(
                        "height: 100%; display: flex; flex-direction: column; justify-content: flex-start;"
                    )
                ):
                    with ui.row().classes("items-center q-mb-md").style("gap: 8px;"):
                        ui.icon("folder", size="1.2em").classes("text-grey-7")
                        ui.label("Source Files").classes("text-caption text-grey-7").style(
                            "font-weight: 500;"
                        )
                    ui.label(f"{stats['total_files']:,}").classes(
                        "text-h4 text-grey-9 q-mb-xs"
                    ).style("line-height: 1;")
                    ui.label("ALTO XML format").classes("text-caption text-grey-6")

            # Average pages per issue
            with ui.card().classes("flex-1").style(card_style):
                with (
                    ui.card_section()
                    .classes("q-pa-md")
                    .style(
                        "height: 100%; display: flex; flex-direction: column; justify-content: flex-start;"
                    )
                ):
                    with ui.row().classes("items-center q-mb-md").style("gap: 8px;"):
                        ui.icon("auto_stories", size="1.2em").classes("text-grey-7")
                        ui.label("Avg Pages/Issue").classes("text-caption text-grey-7").style(
                            "font-weight: 500;"
                        )
                    ui.label(f"{stats['avg_pages']:.1f}").classes(
                        "text-h4 text-grey-9 q-mb-xs"
                    ).style("line-height: 1;")
                    ui.label(f"{stats['total_issues']:,} issues total").classes(
                        "text-caption text-grey-6"
                    )

            # Images card
            with ui.card().classes("flex-1").style(card_style):
                with (
                    ui.card_section()
                    .classes("q-pa-md")
                    .style(
                        "height: 100%; display: flex; flex-direction: column; justify-content: center;"
                    )
                ):
                    with ui.row().classes("items-center q-mb-md").style("gap: 8px;"):
                        ui.icon("image", size="1.2em").classes("text-grey-7")
                        ui.label("Images").classes("text-caption text-grey-7").style(
                            "font-weight: 500;"
                        )
                    if image_stats and image_stats["total_images"] > 0:
                        ui.label(f"{image_stats['total_images']:,}").classes(
                            "text-h4 text-grey-9"
                        ).style("line-height: 1; margin-bottom: 4px;")
                        ui.label(f"{image_stats['total_size_gb']:.1f} GB").classes(
                            "text-caption text-grey-6"
                        )
                    else:
                        sample_images = state.get_sample_images(limit=1)
                        ui.label(f"{len(sample_images) > 0 and '✓' or '—'}").classes(
                            "text-h4 text-grey-9"
                        ).style("line-height: 1;")

        # Sample data and images in one row
        with (
            ui.card().classes("w-full q-mt-sm").style("box-shadow: none; border: 1px solid #e0e0e0")
        ):
            with ui.card_section():
                ui.label("Sample Data & Images").classes("text-h6 q-mb-md")

                # Single row layout - table and images together
                with (
                    ui.row()
                    .classes("w-full")
                    .style("gap: 16px; align-items: flex-start; flex-wrap: nowrap")
                ):
                    # Left: Sample data table
                    with ui.column().style("flex: 0 1 50%; min-width: 0"):
                        ui.label("Text Sample").classes("text-subtitle2 q-mb-sm")
                        sample = state.get_sample_data(limit=5)

                        if sample:
                            # Create a simple table
                            columns = [
                                {"name": "date", "label": "Date", "field": "date", "align": "left"},
                                {
                                    "name": "text",
                                    "label": "Text Preview",
                                    "field": "text",
                                    "align": "left",
                                },
                                {
                                    "name": "page",
                                    "label": "Page",
                                    "field": "page",
                                    "align": "center",
                                },
                            ]

                            ui.table(columns=columns, rows=sample).classes("w-full")
                        else:
                            ui.label("No data available").classes("text-grey-5")

                    # Right: Sample images - horizontal row, 6 images side by side
                    sample_images = state.get_sample_images(limit=6)
                    if sample_images:
                        with ui.column().style("flex: 0 1 50%; min-width: 0; max-width: 50%"):
                            ui.label("Page Images").classes("text-subtitle2 q-mb-sm")

                            # Card container matching table style with drop shadow
                            with (
                                ui.card()
                                .classes("w-full")
                                .style(
                                    "padding: 6px; height: 288px; overflow-x: auto; overflow-y: hidden"
                                )
                            ):
                                # Horizontal row with scrolling
                                with ui.row().style(
                                    "gap: 6px; flex-wrap: nowrap; overflow-x: auto"
                                ):
                                    for img_path in sample_images:
                                        # Parse date from path (format: YYYY/MM/DD/...)
                                        path_parts = img_path.split("/")
                                        date_str = "Unknown"
                                        if len(path_parts) >= 3:
                                            year, month, day = (
                                                path_parts[0],
                                                path_parts[1],
                                                path_parts[2],
                                            )
                                            date_str = f"{year}-{month}-{day}"

                                        # Extract page number from filename (e.g., "max_7.jpg" -> "Page 7")
                                        filename = path_parts[-1] if path_parts else ""
                                        page_num = "Unknown"
                                        if "max_" in filename:
                                            try:
                                                page_num = f"Page {filename.split('max_')[1].split('.')[0]}"
                                            except (IndexError, ValueError):
                                                page_num = filename

                                        # Thumbnail card - wider to better display images
                                        with ui.card().style(
                                            "box-shadow: none; border: 1px solid #e0e0e0; flex-shrink: 0; width: 150px"
                                        ):
                                            ui.image(
                                                f"/static/{state.selected_source}/images/{img_path}"
                                            ).classes("w-full").style(
                                                "height: 193px; object-fit: cover"
                                            )
                                            with (
                                                ui.card_section()
                                                .classes("q-pa-none text-center")
                                                .style("padding: 2px 4px")
                                            ):
                                                ui.label(date_str).classes(
                                                    "text-caption text-grey-7"
                                                ).style("line-height: 1.2; margin: 0")
                                                ui.label(page_num).classes(
                                                    "text-caption text-grey-6"
                                                ).style("line-height: 1.2; margin: 0")
    else:
        with ui.card().classes("w-full"):
            with ui.card_section().classes("text-center q-pa-xl"):
                ui.icon("info", size="3em").classes("text-grey-5 q-mb-md")
                ui.label("Please select a newspaper source to begin").classes("text-h6 text-grey-7")
