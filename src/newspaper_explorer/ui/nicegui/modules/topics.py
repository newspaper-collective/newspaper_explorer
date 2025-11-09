"""
Topics tab for the NiceGUI interface
"""

from nicegui import ui


async def create_topics_tab(state):
    """
    Create the topics analysis tab

    Args:
        state: AppState instance
    """
    ui.label("Topic Modeling").classes("text-h4 q-mb-md")

    with ui.card().classes("w-full"):
        ui.label("Topic Distribution Over Time").classes("text-h6 q-mb-sm")

        with ui.column().classes("q-pa-md items-center"):
            ui.icon("topic", size="3em").classes("text-grey-5")
            ui.label("Topic charts will appear here")

    # Topic list
    with ui.card().classes("w-full q-mt-md"):
        ui.label("Discovered Topics").classes("text-h6 q-mb-sm")

        topics = [
            {
                "id": 1,
                "name": "Politics & Government",
                "docs": 234,
                "keywords": "kaiser, reichstag, politik",
            },
            {"id": 2, "name": "War & Military", "docs": 189, "keywords": "krieg, armee, truppen"},
            {
                "id": 3,
                "name": "Economy & Trade",
                "docs": 156,
                "keywords": "handel, wirtschaft, markt",
            },
            {
                "id": 4,
                "name": "Culture & Society",
                "docs": 145,
                "keywords": "kultur, theater, kunst",
            },
        ]

        for topic in topics:
            with ui.card().classes("q-mb-sm"):
                with ui.card_section():
                    with ui.row().classes("items-center"):
                        ui.label(f"Topic {topic['id']}:").classes("text-weight-bold")
                        ui.label(topic["name"]).classes("text-h6 q-ml-sm")
                        ui.space()
                        ui.badge(f"{topic['docs']} docs").props("color=blue")

                    ui.label(f"Keywords: {topic['keywords']}").classes("text-caption text-grey-7")
