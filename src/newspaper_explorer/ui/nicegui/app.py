"""
Historical Newspaper Explorer - NiceGUI Interface

A modern web interface for exploring historical newspaper collections.
"""

import os
from pathlib import Path

from nicegui import app, ui

from newspaper_explorer.config.base import get_config
from newspaper_explorer.ui.nicegui.modules.base import AppState
from newspaper_explorer.ui.nicegui.modules.sidebar import create_sidebar
from newspaper_explorer.ui.nicegui.modules.overview import create_overview_tab
from newspaper_explorer.ui.nicegui.modules.browse import create_browse_tab
from newspaper_explorer.ui.nicegui.modules.entities import create_entity_tab
from newspaper_explorer.ui.nicegui.modules.concepts import create_knowledge_graph_tab
from newspaper_explorer.ui.nicegui.modules.images import create_images_tab
from newspaper_explorer.ui.nicegui.modules.search import create_search_tab
from newspaper_explorer.ui.nicegui.modules.topics import create_topics_tab
from newspaper_explorer.ui.nicegui.modules.emotions import create_emotions_tab


# Global state instance
state = AppState()


def setup_and_run(host: str = "0.0.0.0", port: int = 8080, reload: bool = True):
    """
    Setup pages and run the NiceGUI application

    Args:
        host: Host address to bind to
        port: Port number to use
        reload: Enable auto-reload on code changes
    """

    # Setup static file serving for images
    config = get_config()
    for source in state.available_sources:
        images_path = Path(config.data_dir) / "raw" / source / "images"
        if images_path.exists():
            app.add_static_files(f"/static/{source}/images", images_path)

    # Register the main page
    @ui.page("/")
    async def main_page():
        """Main application page"""

        # Create sidebar with filters (data loading happens in AppState.__init__)
        drawer = await create_sidebar(state)

        # Add floating menu button to toggle sidebar
        ui.button(icon="menu", on_click=drawer.toggle).props("fab-mini").classes(
            "fixed top-4 left-4 z-50"
        )

        # Main content area with tabs
        with ui.tabs().classes("w-full") as tabs:
            overview_tab = ui.tab("Overview", icon="home")
            browse_tab = ui.tab("Browse", icon="calendar_view_month")
            search_tab = ui.tab("Search", icon="search")
            entities_tab = ui.tab("Entities", icon="person")
            graph_tab = ui.tab("Knowledge Graph", icon="hub")
            images_tab = ui.tab("Images", icon="image")
            topics_tab = ui.tab("Topics", icon="topic")
            emotions_tab = ui.tab("Emotions", icon="mood")

        with ui.tab_panels(tabs, value=overview_tab).classes("w-full q-pa-md"):
            # Overview tab
            with ui.tab_panel(overview_tab):
                await create_overview_tab(state)

            # Browse tab
            with ui.tab_panel(browse_tab):
                await create_browse_tab(state)

            # Search tab
            with ui.tab_panel(search_tab):
                await create_search_tab(state)

            # Entities tab
            with ui.tab_panel(entities_tab):
                await create_entity_tab(state)

            # Knowledge Graph tab
            with ui.tab_panel(graph_tab):
                await create_knowledge_graph_tab(state)

            # Images tab
            with ui.tab_panel(images_tab):
                await create_images_tab(state)

            # Topics tab
            with ui.tab_panel(topics_tab):
                await create_topics_tab(state)

            # Emotions tab
            with ui.tab_panel(emotions_tab):
                await create_emotions_tab(state)

    # Now run the server
    ui.run(
        host=host,
        port=port,
        title="Historical Newspaper Explorer",
        reload=reload,
        show=True,  # Auto-open browser
        favicon="📰",
    )


# Alias for backwards compatibility
run_app = setup_and_run


if __name__ in {"__main__", "__mp_main__"}:
    # Allow configuration via environment variables
    host = os.environ.get("NICEGUI_HOST", "0.0.0.0")
    port = int(os.environ.get("NICEGUI_PORT", 7860))
    reload = os.environ.get("NICEGUI_RELOAD", "1") == "1"

    setup_and_run(host=host, port=port, reload=reload)
