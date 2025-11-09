"""
Search and browse tab for the NiceGUI interface
"""

from nicegui import ui
import polars as pl
from pathlib import Path


async def create_search_tab(state):
    """
    Create the search and browse tab

    Args:
        state: AppState instance
    """
    ui.label("Search & Browse Collection").classes("text-h4 q-mb-md")

    # Main layout with filters on left and content on right
    with ui.row().classes("w-full gap-4"):
        # Left sidebar - Browse filters
        with ui.column().classes("w-64"):
            with ui.card().classes("w-full"):
                with ui.card_section():
                    ui.label("Browse Filters").classes("text-h6 q-mb-md")

                    # Date range filter
                    ui.label("Time Period").classes("text-subtitle2 q-mb-xs")
                    with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                        year_from = ui.number(
                            label="From", value=1900, min=1800, max=2000, format="%.0f"
                        ).classes("col")
                        ui.label("-").classes("text-grey-7")
                        year_to = ui.number(
                            label="To", value=1930, min=1800, max=2000, format="%.0f"
                        ).classes("col")

                    ui.separator()

                    # Source filter
                    ui.label("Source").classes("text-subtitle2 q-mb-xs q-mt-md")
                    source_select = ui.select(
                        label="Newspaper",
                        options=["All", "Der Tag", "Berliner Tageblatt"],
                        value="All",
                    ).classes("w-full q-mb-md")

                    ui.separator()

                    # Content type filter
                    ui.label("Content Type").classes("text-subtitle2 q-mb-xs q-mt-md")
                    with ui.column().classes("w-full q-mb-md"):
                        ui.checkbox("Articles", value=True)
                        ui.checkbox("Advertisements", value=False)
                        ui.checkbox("Images", value=False)

                    ui.separator()

                    # Topic filter
                    ui.label("Topics").classes("text-subtitle2 q-mb-xs q-mt-md")
                    topic_select = ui.select(
                        label="Filter by topic",
                        options=["All", "Politics", "Culture", "Economy", "Local News"],
                        value="All",
                        multiple=True,
                    ).classes("w-full q-mb-md")

                    ui.separator()

                    # Entity filter
                    ui.label("Entities").classes("text-subtitle2 q-mb-xs q-mt-md")
                    with ui.column().classes("w-full q-mb-md"):
                        entity_input = ui.input(
                            placeholder="Person, location, organization..."
                        ).classes("w-full")

                    # Apply filters button
                    ui.button("Apply Filters", icon="filter_list").props(
                        "color=primary flat"
                    ).classes("w-full q-mt-md")

        # Right content area - Search and results
        with ui.column().classes("col-grow"):
            # Search box
            with ui.card().classes("w-full"):
                with ui.card_section():
                    ui.label("Search Collection").classes("text-h6 q-mb-sm")

                    with ui.row().classes("w-full items-center gap-2"):
                        search_input = (
                            ui.input(placeholder="Search for text, entities, topics, events...")
                            .classes("col-grow")
                            .props("outlined dense")
                        )

                        ui.button("Search", icon="search").props("color=primary")
                        ui.button("Clear", icon="clear").props("flat")

                    # Search type selector
                    with ui.row().classes("w-full q-mt-sm"):
                        ui.label("Search in:").classes("text-caption text-grey-7")
                        search_type = ui.radio(
                            ["All content", "Headlines", "Body text", "Image captions"],
                            value="All content",
                        ).props("inline dense")

            # Browse view / Results area
            with ui.card().classes("w-full q-mt-md"):
                with ui.card_section():
                    # View controls
                    with ui.row().classes("w-full items-center justify-between q-mb-md"):
                        ui.label("Browse Results").classes("text-h6")

                        with ui.row().classes("items-center gap-2"):
                            ui.label("Sort by:").classes("text-caption")
                            sort_select = (
                                ui.select(
                                    options=["Date (newest)", "Date (oldest)", "Relevance"],
                                    value="Date (newest)",
                                )
                                .props("dense outlined")
                                .classes("w-40")
                            )

                            ui.label("View:").classes("text-caption q-ml-md")
                            with ui.button_group():
                                ui.button(icon="view_list").props("flat dense")
                                ui.button(icon="view_module").props("flat dense")
                                ui.button(icon="view_comfy").props("flat dense color=primary")

                    ui.separator()

                    # Results placeholder
                    with ui.column().classes("items-center q-pa-xl"):
                        ui.icon("auto_stories", size="4em").classes("text-grey-4")
                        ui.label("Browse the collection or enter a search query").classes(
                            "text-subtitle1 text-grey-7"
                        )
                        ui.label(
                            "Use filters on the left to narrow down results by time period, source, or topic"
                        ).classes("text-caption text-grey-6 text-center")

            # Pagination placeholder
            with ui.row().classes("w-full items-center justify-center q-mt-md gap-2"):
                ui.button(icon="chevron_left").props("flat round")
                ui.label("Page 1 of 1").classes("text-caption")
                ui.button(icon="chevron_right").props("flat round")
