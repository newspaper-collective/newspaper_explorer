"""
Entity analysis tab for the NiceGUI interface
"""

from nicegui import ui
import polars as pl


async def create_entity_tab(state):
    """
    Create the entities analysis tab

    Args:
        state: AppState instance with filtered data
    """
    ui.label("Entity Analysis").classes("text-h4 q-mb-md")

    # Timeline chart placeholder
    with ui.card().classes("w-full"):
        ui.label("Entity Mentions Over Time").classes("text-h6 q-mb-sm")

        # Check if we have real data
        df = state.get_filtered_entities()

        if df is not None and len(df) > 0:
            # Real data available
            ui.label(f"Showing {len(df)} entity mentions").classes("text-caption")

            # For now, show message. We'll add Plotly integration next
            with ui.column().classes("q-pa-md items-center"):
                ui.icon("timeline", size="3em").classes("text-grey-5")
                ui.label("Chart will appear here")
                ui.label(f"Date range: {state.start_date} to {state.end_date}").classes(
                    "text-caption"
                )
        else:
            # Mock data message
            with ui.column().classes("q-pa-md items-center"):
                ui.icon("info", size="2em").classes("text-blue")
                ui.label("No entity data loaded").classes("text-subtitle1")
                ui.label("Using mock data for demonstration").classes("text-caption")

    # Entity distribution and top entities
    with ui.row().classes("w-full q-mt-md"):
        # Type distribution (left)
        with ui.card().classes("col-6"):
            ui.label("Entity Type Distribution").classes("text-h6 q-mb-sm")

            with ui.column().classes("q-pa-md items-center"):
                ui.icon("pie_chart", size="3em").classes("text-grey-5")
                ui.label("Pie chart will appear here")

        # Top entities (right)
        with ui.card().classes("col-6"):
            ui.label("Top Entities This Period").classes("text-h6 q-mb-sm")

            # Mock data table
            mock_entities = [
                {"Entity": "Kaiser Wilhelm II", "Type": "PERSON", "Count": 145},
                {"Entity": "Berlin", "Type": "LOCATION", "Count": 132},
                {"Entity": "Reichstag", "Type": "ORG", "Count": 98},
                {"Entity": "Deutschland", "Type": "LOCATION", "Count": 156},
                {"Entity": "Frankreich", "Type": "LOCATION", "Count": 89},
            ]

            columns = [
                {"name": "entity", "label": "Entity", "field": "Entity", "align": "left"},
                {"name": "type", "label": "Type", "field": "Type", "align": "left"},
                {"name": "count", "label": "Count", "field": "Count", "align": "right"},
            ]

            ui.table(columns=columns, rows=mock_entities, row_key="Entity").classes("w-full")
