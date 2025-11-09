"""
Sidebar component with global filters for the NiceGUI interface
"""

from nicegui import ui
from datetime import date


async def create_sidebar(state):
    """
    Create the sidebar drawer with filters and settings

    Args:
        state: AppState instance
    """
    # Create collapsible drawer with fixed width
    drawer = (
        ui.left_drawer(value=True, elevated=True).classes("bg-blue-grey-1").style("width: 320px")
    )

    with drawer:
        # Header with collapse button
        with ui.row().classes("w-full items-center q-pa-sm no-wrap"):
            ui.label("📰 Newspaper Explorer").classes("text-h6 text-no-wrap")
            ui.space()
            ui.button(icon="chevron_left", on_click=drawer.toggle).props(
                "flat round dense"
            ).classes("text-grey-7")

        ui.separator()

        with ui.column().classes("w-full q-pa-md"):
            # Current source info
            with ui.card().classes("w-full q-mb-md"):
                with ui.card_section():
                    ui.label("Current Source").classes("text-subtitle2 q-mb-sm")

                    if state.selected_source:
                        stats = state.get_source_stats()

                        # Newspaper title
                        ui.label(stats["newspaper_title"]).classes("text-weight-bold text-body1")

                        # Language and years in one row
                        with ui.row().classes("items-center gap-2 q-mt-xs no-wrap"):
                            with ui.row().classes("items-center gap-1 no-wrap"):
                                ui.icon("language", size="sm").classes("text-grey-7 flex-shrink-0")
                                ui.label(stats["language"].upper()).classes(
                                    "text-caption text-grey-7"
                                )

                            ui.label("•").classes("text-grey-5")

                            with ui.row().classes("items-center gap-1 no-wrap"):
                                ui.icon("calendar_month", size="sm").classes(
                                    "text-grey-7 flex-shrink-0"
                                )
                                ui.label(stats["years_available"]).classes(
                                    "text-caption text-grey-7"
                                )

                        # Provider
                        if stats["source_provider"]:
                            with ui.row().classes("items-center gap-1 no-wrap"):
                                ui.icon("business", size="sm").classes("text-grey-7 flex-shrink-0")
                                ui.label(stats["source_provider"]).classes(
                                    "text-caption text-grey-7"
                                )

                        # Collection size
                        if stats["total_archive_size"]:
                            with ui.row().classes("items-center gap-1 no-wrap"):
                                ui.icon("storage", size="sm").classes("text-grey-7 flex-shrink-0")
                                ui.label(stats["total_archive_size"]).classes(
                                    "text-caption text-grey-7"
                                )

                        # Document count (if available)
                        if stats["total_documents"] > 0:
                            with ui.row().classes("items-center gap-1 no-wrap"):
                                ui.icon("description", size="sm").classes(
                                    "text-grey-7 flex-shrink-0"
                                )
                                ui.label(f"{stats['total_documents']:,} documents").classes(
                                    "text-caption text-grey-7"
                                )

                        # License
                        if stats["license"]:
                            with ui.row().classes("items-center gap-1 no-wrap"):
                                ui.icon("copyright", size="sm").classes("text-grey-7 flex-shrink-0")
                                ui.label(stats["license"]).classes("text-caption text-grey-7")

                        # Collapsible additional info
                        if stats["info"] or stats["citation"]:
                            with (
                                ui.expansion("More Info", icon="info_outline")
                                .classes("q-mt-sm w-full")
                                .props("dense")
                            ):
                                with ui.column().classes("w-full q-pa-md gap-2"):
                                    # Info text
                                    if stats["info"]:
                                        ui.label(stats["info"]).classes(
                                            "text-caption text-grey-8"
                                        ).style("line-height: 1.4")

                                    # Citation
                                    if stats["citation"]:
                                        if stats["info"]:
                                            ui.separator().classes("q-my-sm")
                                        ui.label("Citation:").classes(
                                            "text-caption text-weight-bold text-grey-8 q-mb-xs"
                                        )
                                        ui.label(stats["citation"]).classes(
                                            "text-caption text-grey-7"
                                        ).style("line-height: 1.4; font-style: italic")
                    else:
                        ui.label("No source selected").classes("text-grey-5")

            # Date range filter
            with ui.card().classes("w-full q-mb-md"):
                with ui.card_section():
                    ui.label("Date Range Filter").classes("text-subtitle2 q-mb-sm")

                    # Side-by-side date inputs
                    with ui.row().classes("w-full gap-2"):
                        # Start date
                        start_date_input = (
                            ui.input(label="From", value=str(state.start_date))
                            .props("outlined dense type=date fit anchor=bottom-left")
                            .classes("flex-1")
                        )

                        # End date
                        end_date_input = (
                            ui.input(label="To", value=str(state.end_date))
                            .props("outlined dense type=date fit anchor=bottom-left")
                            .classes("flex-1")
                        )

                    # Update state when dates change
                    def update_dates():
                        try:
                            state.start_date = date.fromisoformat(start_date_input.value)
                            state.end_date = date.fromisoformat(end_date_input.value)
                            ui.notify("Date range updated", type="positive")
                        except ValueError:
                            ui.notify("Invalid date format", type="negative")

                    start_date_input.on("change", update_dates)
                    end_date_input.on("change", update_dates)

            # Available Analysis & Preprocessed Data
            with ui.card().classes("w-full"):
                with ui.card_section():
                    if state.selected_source:
                        # Available Analysis Results
                        analysis_results = state.get_analysis_results()
                        if analysis_results:
                            ui.label("Available Analysis:").classes(
                                "text-caption text-weight-medium q-mb-xs q-mt-sm"
                            )
                            for analysis_type, info in analysis_results.items():
                                # Icon based on analysis type
                                icon_map = {
                                    "entities": "people",
                                    "emotions": "mood",
                                    "topics": "topic",
                                    "keywords": "label",
                                    "concepts": "psychology",
                                    "layout": "view_quilt",
                                }
                                with ui.row().classes("items-center gap-1 no-wrap"):
                                    ui.icon(
                                        icon_map.get(analysis_type, "analytics"), size="sm"
                                    ).classes("text-grey-7 flex-shrink-0")
                                    ui.label(
                                        f"{analysis_type.capitalize()}: {info['count']} files"
                                    ).classes("text-caption text-grey-7")

                        # Preprocessed Datasets
                        # Check if text blocks exist
                        stats = state.get_comprehensive_stats()
                        if stats.get("total_blocks", 0) > 0 or stats.get("total_lines", 0) > 0:
                            ui.label("Preprocessed Data:").classes(
                                "text-caption text-weight-medium q-mb-xs q-mt-sm"
                            )
                            if stats.get("total_lines", 0) > 0:
                                with ui.row().classes("items-center gap-1 no-wrap"):
                                    ui.icon("description", size="sm").classes(
                                        "text-grey-7 flex-shrink-0"
                                    )
                                    ui.label(f"Lines: {stats['total_lines']:,}").classes(
                                        "text-caption text-grey-7"
                                    )
                            if stats.get("total_blocks", 0) > 0:
                                with ui.row().classes("items-center gap-1 no-wrap"):
                                    ui.icon("view_agenda", size="sm").classes(
                                        "text-grey-7 flex-shrink-0"
                                    )
                                    ui.label(f"Text Blocks: {stats['total_blocks']:,}").classes(
                                        "text-caption text-grey-7"
                                    )
                    else:
                        ui.label("Select a source to see stats").classes("text-caption text-grey-5")

    # Return drawer so it can be controlled from outside
    return drawer
