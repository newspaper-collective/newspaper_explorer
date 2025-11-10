"""Layout tab for the NiceGUI UI."""

import logging
from datetime import date
from typing import Optional

from nicegui import ui

from newspaper_explorer.ui.nicegui.modules.base import AppState

logger = logging.getLogger(__name__)


def create_layout_tab(state: AppState) -> None:
    """Create the layout tab with layout detection visualization."""

    with ui.column().classes("w-full gap-4"):
        # Header
        ui.label("Layout Detection").classes("text-2xl font-bold")
        ui.markdown(
            "View newspaper page images with detected layout elements (text blocks, images, tables, etc.)"
        )

        # Check if source is selected
        if not state.selected_source:
            ui.label("Please select a source first.").classes("text-orange-500")
            return

        # Check if layout data is available
        available_layouts = state.get_available_layout_files()
        if not available_layouts:
            ui.label("No layout detection results found for this source.").classes(
                "text-orange-500"
            )
            ui.markdown(
                f"Run layout detection using: `newspaper-explorer analyze layout detect --source {state.selected_source}`"
            )
            return

        # Analysis selector
        with ui.row().classes("w-full items-center gap-4"):
            ui.label("Analysis:").classes("font-semibold")

            analysis_options = {
                analysis_id: display_name for analysis_id, display_name, _ in available_layouts
            }

            selected_analysis = ui.select(
                options=analysis_options,
                value=list(analysis_options.keys())[0] if analysis_options else None,
            ).classes("flex-grow")

            def on_analysis_change():
                """Load selected analysis"""
                if selected_analysis.value:
                    state.load_layout(selected_analysis.value)
                    update_ui()

            selected_analysis.on("update:model-value", lambda: on_analysis_change())

        # Load initial analysis
        if selected_analysis.value:
            state.load_layout(selected_analysis.value)

        # Statistics panel
        stats_container = ui.column().classes("w-full")

        # Filters
        with ui.expansion("Filters", icon="filter_alt").classes("w-full"):
            with ui.row().classes("w-full gap-4"):
                # Date range filter
                with ui.column().classes("flex-grow"):
                    ui.label("Date Range:").classes("font-semibold")
                    date_from = ui.date(value=None).classes("w-full")
                    date_to = ui.date(value=None).classes("w-full")

                # Class filter
                with ui.column().classes("flex-grow"):
                    ui.label("Detection Classes:").classes("font-semibold")
                    class_checkboxes = ui.column().classes("gap-2")

            def apply_filters():
                """Apply filters and update display"""
                # Update state date range if provided
                if date_from.value:
                    state.start_date = date_from.value
                if date_to.value:
                    state.end_date = date_to.value

                # Filter data
                filtered_df = state.get_filtered_layout()

                if filtered_df is not None:
                    update_statistics(filtered_df)
                    update_image_gallery(filtered_df)

            ui.button("Apply Filters", on_click=apply_filters).classes("bg-blue-500 text-white")

        # Image gallery
        gallery_container = ui.column().classes("w-full")

        def update_statistics(df):
            """Update statistics display"""
            stats_container.clear()

            with stats_container:
                if df is None or len(df) == 0:
                    ui.label("No data loaded").classes("text-gray-500")
                    return

                with ui.row().classes("w-full gap-4"):
                    # Total detections
                    with ui.card().classes("p-4"):
                        ui.label("Total Detections").classes("text-sm text-gray-600")
                        ui.label(str(len(df))).classes("text-3xl font-bold")

                    # Unique pages
                    with ui.card().classes("p-4"):
                        ui.label("Pages Processed").classes("text-sm text-gray-600")
                        ui.label(str(df["page_id"].n_unique())).classes("text-3xl font-bold")

                    # Date range
                    with ui.card().classes("p-4"):
                        ui.label("Date Range").classes("text-sm text-gray-600")
                        if "date" in df.columns and len(df) > 0:
                            min_date = df["date"].min()
                            max_date = df["date"].max()
                            ui.label(f"{min_date} to {max_date}").classes("text-lg")
                        else:
                            ui.label("N/A").classes("text-lg text-gray-500")

                # Class distribution
                with ui.card().classes("w-full p-4 mt-4"):
                    ui.label("Detection Classes").classes("text-lg font-semibold mb-2")

                    class_counts = (
                        df.group_by("class")
                        .agg(df["detection_id"].count().alias("count"))
                        .sort("count", descending=True)
                    )

                    with ui.row().classes("w-full gap-2 flex-wrap"):
                        for row in class_counts.iter_rows(named=True):
                            class_name = row["class"]
                            count = row["count"]
                            percentage = (count / len(df)) * 100

                            with ui.card().classes("p-3"):
                                ui.label(class_name).classes("font-semibold")
                                ui.label(f"{count} ({percentage:.1f}%)").classes(
                                    "text-sm text-gray-600"
                                )

        def update_image_gallery(df):
            """Update image gallery with filtered data"""
            gallery_container.clear()

            with gallery_container:
                if df is None or len(df) == 0:
                    ui.label("No images to display").classes("text-gray-500")
                    return

                # Group by page
                unique_pages = df["page_id"].unique().to_list()

                ui.label(f"Showing {len(unique_pages)} pages").classes("text-lg font-semibold mt-4")

                # Pagination
                items_per_page = 10
                total_pages = (len(unique_pages) + items_per_page - 1) // items_per_page
                current_page = [1]  # Use list to allow mutation in nested function

                page_container = ui.column().classes("w-full mt-4")
                pagination_controls = ui.row().classes(
                    "w-full justify-center items-center gap-4 mt-4"
                )

                def show_page(page_num: int):
                    """Display images for current page"""
                    current_page[0] = page_num
                    start_idx = (page_num - 1) * items_per_page
                    end_idx = min(start_idx + items_per_page, len(unique_pages))
                    page_ids = unique_pages[start_idx:end_idx]

                    page_container.clear()
                    with page_container:
                        with ui.row().classes("w-full gap-4 flex-wrap"):
                            for page_id in page_ids:
                                # Get detections for this page
                                page_detections = df.filter(df["page_id"] == page_id)

                                with ui.card().classes("p-4"):
                                    ui.label(page_id).classes("font-semibold")
                                    ui.label(f"{len(page_detections)} detections").classes(
                                        "text-sm text-gray-600"
                                    )

                                    # TODO: Display image with bounding boxes
                                    # This requires image loading and rendering
                                    # For now, just show detection list
                                    with ui.expansion("Detections", icon="list"):
                                        for det in page_detections.iter_rows(named=True):
                                            ui.label(
                                                f"{det['class']} - conf: {det['confidence']:.2f}"
                                            ).classes("text-sm")

                    # Update pagination controls
                    pagination_controls.clear()
                    with pagination_controls:
                        prev_btn = ui.button(
                            "Previous",
                            on_click=lambda: show_page(current_page[0] - 1),
                        ).props("flat")
                        page_label = ui.label(f"Page {current_page[0]} of {total_pages}")
                        next_btn = ui.button(
                            "Next", on_click=lambda: show_page(current_page[0] + 1)
                        ).props("flat")

                        prev_btn.set_enabled(current_page[0] > 1)
                        next_btn.set_enabled(current_page[0] < total_pages)

                # Show first page
                show_page(1)

        def update_ui():
            """Update entire UI with current data"""
            if state.layout_df is not None:
                update_statistics(state.layout_df)
                update_image_gallery(state.layout_df)

                # Update class checkboxes
                class_checkboxes.clear()
                with class_checkboxes:
                    unique_classes = state.layout_df["class"].unique().sort().to_list()
                    for cls in unique_classes:
                        ui.checkbox(cls, value=True).classes("text-sm")

        # Initial UI update
        update_ui()
