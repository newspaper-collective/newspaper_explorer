"""
Knowledge graph tab for the NiceGUI interface
"""

from nicegui import ui


async def create_knowledge_graph_tab(state):
    """
    Create the knowledge graph tab

    Args:
        state: AppState instance
    """
    ui.label("Entity Relationship Network").classes("text-h4 q-mb-md")

    with ui.card().classes("w-full"):
        # Embed the existing knowledge graph
        ui.html(
            """
            <iframe 
                src="https://dertag.proto-khora.pages.dev/" 
                style="width: 100%; height: 800px; border: none;"
                frameborder="0"
            ></iframe>
        """,
            sanitize=False,  # Required in NiceGUI 3.x
        )
