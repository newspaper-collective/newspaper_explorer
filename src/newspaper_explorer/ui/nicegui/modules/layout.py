"""Layout tab for the NiceGUI UI."""

import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
from nicegui import ui
from PIL import Image, ImageDraw, ImageFont

from newspaper_explorer.ui.nicegui.modules.base import AppState

logger = logging.getLogger(__name__)


# Color map for detection classes
DETECTION_COLORS = {
    "Text": "#FF4444",
    "Picture": "#44FF44",
    "Section-header": "#4444FF",
    "Table": "#FFFF44",
    "Page-header": "#FF44FF",
    "Page-footer": "#44FFFF",
    "Caption": "#FFA500",
    "List": "#800080",
    "Title": "#FF1493",
    "Figure": "#00CED1",
    "Formula": "#FFD700",
}


def create_annotated_image(
    image_path: str, detections_df: pl.DataFrame, max_width: int = 800
) -> str:
    """
    Generate annotated image with bounding boxes drawn on it.

    Args:
        image_path: Path to the original page image
        detections_df: DataFrame with detections for this page
        max_width: Maximum width for display (image will be scaled down)

    Returns:
        Base64-encoded image data URI
    """
    try:
        # Load image
        img = Image.open(image_path)

        # Scale down if too large
        if img.width > max_width:
            scale = max_width / img.width
            new_height = int(img.height * scale)
            img = img.resize((max_width, new_height), Image.Resampling.LANCZOS)
        else:
            scale = 1.0

        # Create drawing context
        draw = ImageDraw.Draw(img)

        # Try to load a font, fall back to default if not available
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except:
            font = ImageFont.load_default()

        # Draw each detection
        for row in detections_df.iter_rows(named=True):
            # Scale coordinates if image was resized
            x1 = int(row["bbox_x1"] * scale)
            y1 = int(row["bbox_y1"] * scale)
            x2 = int(row["bbox_x2"] * scale)
            y2 = int(row["bbox_y2"] * scale)

            class_name = row["class_name"]
            confidence = row["confidence"]
            color = DETECTION_COLORS.get(class_name, "#FFFFFF")

            # Draw rectangle
            draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            # Get text size using textbbox
            bbox = draw.textbbox((x1, y1 - 15), label, font=font)
            label_width = bbox[2] - bbox[0]
            label_height = bbox[3] - bbox[1]

            # Draw background rectangle for label
            draw.rectangle(
                [x1, y1 - 15 - label_height, x1 + label_width + 4, y1 - 15],
                fill=color,
            )

            # Draw text
            draw.text((x1 + 2, y1 - 15 - label_height), label, fill="black", font=font)

        # Convert to base64
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        img_str = base64.b64encode(buffer.getvalue()).decode()

        return f"data:image/jpeg;base64,{img_str}"

    except Exception as e:
        logger.error(f"Error creating annotated image for {image_path}: {e}")
        return ""


def load_result_metadata(result_file_path: str) -> dict:
    """
    Load layout.json metadata from the same directory as the result file.

    Args:
        result_file_path: Path to the result file (parquet)

    Returns:
        Dictionary with metadata, or empty dict if not found
    """
    result_path = Path(result_file_path)
    metadata_path = result_path.parent / "layout.json"

    if metadata_path.exists():
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata from {metadata_path}: {e}")
            return {}
    return {}


async def create_layout_tab(state: AppState) -> None:
    """Create the layout tab with layout detection visualization."""

    ui.label("Layout Detection").classes("text-h4 q-mb-md")

    # Create a container for the entire content that we can refresh
    content_container = ui.column().classes("w-full")

    async def refresh_layout_tab():
        """Refresh the entire layout tab content"""
        # Re-fetch available files each time we refresh
        available_files = state.get_available_layout_files()
        content_container.clear()

        with content_container:
            await _create_layout_content(state, available_files, refresh_layout_tab)

    # Initial render
    await refresh_layout_tab()


async def _create_layout_content(state: AppState, available_files: list, refresh_callback) -> None:
    """Create the actual layout content (separated for easy refresh)"""

    # File selector at the top if files available
    if len(available_files) > 0:
        with ui.card().classes("w-full q-mb-md"):
            with ui.column().classes("w-full q-pa-md gap-3"):
                # Top row: selector and metadata
                with ui.row().classes("w-full items-start gap-4"):
                    # Left side: selector (takes less space)
                    with (
                        ui.column()
                        .classes("gap-2 flex-shrink-0")
                        .style("min-width: 350px; max-width: 450px;")
                    ):
                        ui.label("Layout Dataset").classes("text-subtitle2 font-semibold")

                        # Create options dict for select: {file_path: display_name}
                        file_options = {
                            file_path: display_name
                            for display_name, file_path, metadata in available_files
                        }

                        # Get currently loaded file path or default to first
                        current_file = None
                        if state.layout_df is not None and available_files:
                            # Try to preserve the currently loaded file
                            # (This will be set after first load)
                            current_file = getattr(
                                state, "_current_layout_file", available_files[0][1]
                            )
                        elif available_files:
                            current_file = available_files[0][1]

                        file_select = (
                            ui.select(
                                options=file_options,
                                value=current_file,
                                label="Select results",
                            )
                            .props("dense outlined")
                            .classes("w-full")
                        )

                        # Reload button
                        async def reload_layout():
                            """Reload layout with selected file"""
                            if file_select.value:
                                # Load layout data (file_select.value is the parquet file path)
                                state.load_layout(file_select.value)
                                # Store the currently loaded file so we can preserve selection
                                state._current_layout_file = file_select.value
                                ui.notify(f"Loaded layout data", type="positive")
                                # Refresh the entire tab to show new data
                                await refresh_callback()

                        ui.button("Load", icon="refresh", on_click=reload_layout).props(
                            "outline dense"
                        )

                    # Right side: metadata (takes remaining space)
                    metadata_container = ui.column().classes("flex-grow gap-1")

                # Bottom row: inline statistics summary
                stats_summary_container = (
                    ui.row()
                    .classes("w-full gap-4 items-center q-pt-sm")
                    .style("border-top: 1px solid #e0e0e0;")
                )

            def display_metadata(metadata: dict):
                """Display metadata information compactly"""
                if not metadata:
                    return

                # Analysis ID (smaller, less prominent)
                if "analysis_id" in metadata:
                    ui.label(f"ID: {metadata['analysis_id']}").classes("text-caption text-grey-6")

                # Combine parameters and stats in one line
                info_parts = []

                # Key parameters
                if "parameters" in metadata and metadata["parameters"]:
                    params = metadata["parameters"]
                    if "confidence_threshold" in params:
                        info_parts.append(f"confidence ≥ {params['confidence_threshold']}")
                    if "batch_size" in params:
                        info_parts.append(f"batch {params['batch_size']}")

                # Processing stats
                if "duration_seconds" in metadata:
                    duration = metadata["duration_seconds"]
                    if duration < 60:
                        info_parts.append(f"{duration:.1f}s")
                    else:
                        mins = int(duration // 60)
                        secs = int(duration % 60)
                        info_parts.append(f"{mins}m {secs}s")

                # Created timestamp
                if "created_at" in metadata:
                    try:
                        created = datetime.fromisoformat(
                            metadata["created_at"].replace("Z", "+00:00")
                        )
                        info_parts.append(created.strftime("%Y-%m-%d %H:%M"))
                    except:
                        pass

                if info_parts:
                    ui.label(" • ".join(info_parts)).classes("text-caption text-grey-7")

            def display_stats_summary(df):
                """Display inline statistics summary"""
                if df is None or len(df) == 0:
                    ui.label("No data").classes("text-caption text-grey-5")
                    return

                # Compact stats in a single line with separators
                stats_parts = []

                # Total detections
                stats_parts.append(f"{len(df):,} detections")

                # Unique pages
                stats_parts.append(f"{df['page_id'].n_unique()} pages")

                # Date range
                if "date" in df.columns:
                    min_date = str(df["date"].min()).split()[0]
                    max_date = str(df["date"].max()).split()[0]
                    if min_date == max_date:
                        stats_parts.append(f"{min_date}")
                    else:
                        stats_parts.append(f"{min_date} → {max_date}")

                # Top 3 classes
                top_classes = (
                    df.group_by("class_name")
                    .agg(pl.count("detection_id").alias("count"))
                    .sort("count", descending=True)
                    .head(3)
                )

                class_strs = []
                for row in top_classes.iter_rows(named=True):
                    class_name = row["class_name"]
                    count = row["count"]
                    percentage = (count / len(df)) * 100
                    class_strs.append(f"{class_name} {percentage:.0f}%")

                if class_strs:
                    stats_parts.append(" | ".join(class_strs))

                # Display all parts
                ui.label(" • ".join(stats_parts)).classes("text-caption text-grey-7")

            # Load and display metadata for current file
            if file_select.value:
                metadata = load_result_metadata(file_select.value)
                with metadata_container:
                    display_metadata(metadata)

                # Only auto-load data if nothing is loaded yet
                if state.layout_df is None:
                    state.load_layout(file_select.value)
                    state._current_layout_file = file_select.value  # type: ignore

                # Display initial stats summary
                with stats_summary_container:
                    display_stats_summary(state.layout_df)

    else:
        # No files available
        with ui.card().classes("w-full q-mb-md"):
            with ui.row().classes("items-center q-pa-md"):
                ui.icon("info", size="2em").classes("text-blue q-mr-md")
                with ui.column():
                    ui.label("No layout detection results found").classes("text-subtitle1")
                    ui.label(
                        f"Run layout detection first: newspaper-explorer analyze layout detect --source {state.selected_source}"
                    ).classes("text-caption text-grey-7")
        return

    # Get layout data (use unfiltered initially, or filtered if no data)
    df = state.layout_df if state.layout_df is not None else None

    # If we have data, try to get filtered version
    if df is not None and len(df) > 0:
        filtered_df = state.get_filtered_layout()
        # Use filtered if it has data, otherwise use unfiltered
        if filtered_df is not None and len(filtered_df) > 0:
            df = filtered_df

    # Show no data message if needed
    if df is None or len(df) == 0:
        with ui.card().classes("w-full q-mb-md"):
            with ui.row().classes("items-center q-pa-md"):
                ui.icon("info", size="2em").classes("text-blue q-mr-md")
                ui.label("No data in selected time range").classes("text-subtitle1")
        return

    # Image gallery
    gallery_container = ui.column().classes("w-full")

    # Filters (compact)
    with ui.expansion("Filters", icon="filter_alt").classes("w-full q-mb-md"):
        with ui.column().classes("q-pa-md gap-4"):
            # Date range filter in a row
            with ui.row().classes("items-center gap-2"):
                ui.label("Date Range:").classes("text-subtitle2")
                date_from = (
                    ui.input(
                        placeholder="From (YYYY-MM-DD)",
                        validation={
                            "Invalid date": lambda value: not value
                            or len(value) == 0
                            or len(value) == 10
                        },
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )
                ui.label("to").classes("text-caption")
                date_to = (
                    ui.input(
                        placeholder="To (YYYY-MM-DD)",
                        validation={
                            "Invalid date": lambda value: not value
                            or len(value) == 0
                            or len(value) == 10
                        },
                    )
                    .props("dense outlined")
                    .classes("w-32")
                )

            # Class filter checkboxes in compact row
            with ui.column().classes("gap-1"):
                ui.label("Detection Classes:").classes("text-subtitle2")
                class_checkboxes = ui.row().classes("gap-3 flex-wrap")

            # Apply button
            def apply_filters():
                """Apply filters and update display"""
                # Update state date range if provided
                if date_from.value:
                    try:
                        from datetime import datetime

                        state.start_date = datetime.strptime(date_from.value, "%Y-%m-%d").date()
                    except:
                        pass
                if date_to.value:
                    try:
                        from datetime import datetime

                        state.end_date = datetime.strptime(date_to.value, "%Y-%m-%d").date()
                    except:
                        pass

                # Filter data
                filtered_df = state.get_filtered_layout()

                if filtered_df is not None:
                    update_image_gallery(filtered_df)

            ui.button("Apply Filters", on_click=apply_filters, icon="filter_alt").props(
                "outline dense"
            )

    # Define update functions
    def update_image_gallery(df):
        """Update image gallery with filtered data"""
        gallery_container.clear()

        with gallery_container:
            if df is None or len(df) == 0:
                ui.label("No images to display").classes("text-gray-500")
                return

            # Group by page and sort chronologically
            unique_pages = df["page_id"].unique().sort().to_list()

            # Pagination
            items_per_page = 10
            total_pages = (len(unique_pages) + items_per_page - 1) // items_per_page
            current_page = [1]  # Use list to allow mutation in nested function

            page_container = ui.column().classes("w-full mt-4")
            pagination_controls = ui.row().classes("w-full justify-center items-center gap-4 mt-4")

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

                            # Get image path from first detection
                            if len(page_detections) > 0:
                                first_detection = page_detections.row(0, named=True)
                                image_path = first_detection.get("image_path")

                                with ui.card().classes("p-4 w-80"):
                                    # Header section with metadata
                                    # Parse page_id: {source}_{YYYY-MM-DD}_{issue}_{daily}_{page}
                                    parts = page_id.split("_")
                                    if len(parts) >= 5:
                                        date_str = parts[1]  # YYYY-MM-DD
                                        issue_num = parts[2]  # issue number
                                        daily_num = parts[3]  # daily issue count
                                        page_num = parts[4]  # page number

                                        # Format date with weekday
                                        try:
                                            from datetime import datetime

                                            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                                            weekday = date_obj.strftime("%A")
                                            formatted_date = date_obj.strftime("%d.%m.%Y")
                                        except:
                                            weekday = ""
                                            formatted_date = date_str

                                        # Display formatted metadata
                                        ui.label(f"{formatted_date} ({weekday})").classes(
                                            "font-semibold text-base"
                                        )
                                        ui.label(
                                            f"Issue {issue_num} • Daily {daily_num} • Page {page_num}"
                                        ).classes("text-xs text-gray-500 mb-1")
                                    else:
                                        # Fallback to page_id if parsing fails
                                        ui.label(page_id).classes("font-semibold mb-1")

                                    ui.label(f"{len(page_detections)} detections").classes(
                                        "text-sm text-gray-600 mb-3"
                                    )

                                    # Image display with fixed height container
                                    if image_path and Path(image_path).exists():
                                        annotated_img = create_annotated_image(
                                            image_path, page_detections, max_width=400
                                        )
                                        if annotated_img:
                                            # Fixed height container to keep cards uniform
                                            ui.image(annotated_img).classes(
                                                "w-full h-96 object-contain rounded bg-gray-100"
                                            )
                                        else:
                                            ui.label("Error loading image").classes(
                                                "text-orange-500 py-8 text-center h-96"
                                            )
                                    else:
                                        ui.label(f"Image not found: {image_path}").classes(
                                            "text-orange-500 py-8 text-center h-96"
                                        )

                                    # Footer with view details button
                                    with ui.row().classes("justify-end mt-3"):

                                        def show_detection_dialog(detections):
                                            """Show detection details in a dialog"""
                                            with (
                                                ui.dialog() as dialog,
                                                ui.card().classes(
                                                    "p-6 min-w-[600px] max-w-[800px]"
                                                ),
                                            ):
                                                ui.label("Detection Details").classes(
                                                    "text-lg font-semibold mb-4 text-center"
                                                )

                                                # Single wrapping row with all detection badges - centered
                                                with ui.column().classes("w-full"):
                                                    with ui.row().classes(
                                                        "gap-2 flex-wrap justify-center max-h-[500px] overflow-y-auto p-2"
                                                    ):
                                                        for det in detections.iter_rows(named=True):
                                                            color = DETECTION_COLORS.get(
                                                                det["class_name"], "#FFFFFF"
                                                            )
                                                            # Compact badge-style display
                                                            with ui.card().classes("p-2"):
                                                                with ui.row().classes(
                                                                    "items-center gap-2"
                                                                ):
                                                                    ui.html(
                                                                        f'<div style="width:16px;height:16px;background:{color};border:2px solid black;border-radius:2px;flex-shrink:0;"></div>',
                                                                        sanitize=False,
                                                                    )
                                                                    ui.label(
                                                                        f"{det['class_name']} {det['confidence']:.2f}"
                                                                    ).classes(
                                                                        "text-sm whitespace-nowrap"
                                                                    )

                                                with ui.row().classes("justify-center mt-4"):
                                                    ui.button("Close", on_click=dialog.close).props(
                                                        "flat"
                                                    )

                                            dialog.open()

                                        ui.button(
                                            "View Details",
                                            icon="info",
                                            on_click=lambda d=page_detections: show_detection_dialog(
                                                d
                                            ),
                                        ).props("flat dense")

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
            update_image_gallery(state.layout_df)

            # Update class checkboxes (compact inline display)
            class_checkboxes.clear()
            with class_checkboxes:
                unique_classes = state.layout_df["class_name"].unique().sort().to_list()
                for cls in unique_classes:
                    ui.checkbox(cls, value=True).props("dense")

    # Initial UI update
    update_ui()
