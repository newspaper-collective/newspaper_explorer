"""
Emotions tab for the NiceGUI interface
"""

from nicegui import ui


async def create_emotions_tab(state):
    """
    Create the emotions analysis tab

    Args:
        state: AppState instance
    """
    ui.label("Emotion Analysis").classes("text-h4 q-mb-md")

    # Timeline chart
    with ui.card().classes("w-full"):
        ui.label("Emotion Intensity Over Time").classes("text-h6 q-mb-sm")

        with ui.column().classes("q-pa-md items-center"):
            ui.icon("mood", size="3em").classes("text-grey-5")
            ui.label("Emotion timeline will appear here")

    # Outlier events
    with ui.card().classes("w-full q-mt-md"):
        ui.label("Notable Emotional Peaks").classes("text-h6 q-mb-sm")

        peaks = [
            {"date": "1914-08-01", "value": 0.89, "event": "War Declaration"},
            {"date": "1915-05-07", "value": 0.85, "event": "Lusitania Sinking"},
            {"date": "1918-11-11", "value": 0.92, "event": "Armistice Day"},
        ]

        for peak in peaks:
            with ui.card().classes("q-mb-sm"):
                with ui.card_section():
                    with ui.row().classes("items-center"):
                        ui.icon("trending_up").classes("text-red")
                        ui.label(peak["date"]).classes("text-weight-bold q-mx-sm")
                        ui.label(peak["event"]).classes("text-grey-8")
                        ui.space()
                        ui.badge(f"{peak['value']:.2f}").props("color=orange")
