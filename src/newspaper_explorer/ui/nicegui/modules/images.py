"""
Image explorer tab for the NiceGUI interface
"""

from nicegui import ui


async def create_images_tab(state):
    """
    Create the image explorer tab

    Args:
        state: AppState instance
    """
    ui.label("Image Explorer").classes("text-h4 q-mb-md")

    # Statistics cards
    with ui.row().classes("w-full q-mb-md"):
        with ui.card().classes("col"):
            with ui.card_section():
                ui.label("Total Pictures").classes("text-caption")
                ui.label("1,234").classes("text-h5")

        with ui.card().classes("col"):
            with ui.card_section():
                ui.label("With Captions").classes("text-caption")
                ui.label("856 (69%)").classes("text-h5 text-positive")

        with ui.card().classes("col"):
            with ui.card_section():
                ui.label("Without Captions").classes("text-caption")
                ui.label("378 (31%)").classes("text-h5 text-warning")

        with ui.card().classes("col"):
            with ui.card_section():
                ui.label("Avg Confidence").classes("text-caption")
                ui.label("0.87").classes("text-h5")

    # View toggle
    with ui.row().classes("q-mb-md"):
        ui.label("View:").classes("self-center q-mr-sm")
        view_toggle = ui.toggle(["Gallery", "List"], value="Gallery")

    # Image gallery
    with ui.card().classes("w-full"):
        ui.label("Image Gallery").classes("text-h6 q-mb-sm")

        # Placeholder for images
        with ui.grid(columns=4).classes("w-full gap-4"):
            for i in range(8):
                with ui.card().classes("cursor-pointer hover:shadow-lg transition-shadow"):
                    # Placeholder image
                    with ui.card_section().classes("q-pa-none"):
                        ui.image("https://via.placeholder.com/300x200?text=Image+" + str(i + 1))

                    with ui.card_section():
                        ui.label(f"1912-02-{11+i:02d}").classes("text-caption")
                        ui.label("Caption text preview...").classes("text-caption text-grey-7")

                        with ui.row().classes("text-caption"):
                            ui.label("🎯 0.92")
                            ui.label("📏 800×600")

        # Pagination
        with ui.row().classes("justify-center q-mt-md"):
            ui.button(icon="chevron_left", on_click=lambda: None).props("flat round")
            ui.label(f"Page {state.current_image_page} of 62").classes("self-center q-mx-md")
            ui.button(icon="chevron_right", on_click=lambda: None).props("flat round")
