"""Pictures tab for the NiceGUI UI - Gallery of extracted newspaper pictures with captions."""

import base64
import io
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import polars as pl
from nicegui import ui
from PIL import Image

from newspaper_explorer.ui.nicegui.modules.base import AppState

logger = logging.getLogger(__name__)


def crop_picture_region(image_path: str, bbox_coords: dict, max_width: int = 600) -> str:
    """
    Crop picture region from full page image based on bounding box coordinates.

    Args:
        image_path: Path to the full page image
        bbox_coords: Dictionary with bbox_x1, bbox_y1, bbox_x2, bbox_y2
        max_width: Maximum width for display (image will be scaled down)

    Returns:
        Base64-encoded image data URI
    """
    try:
        # Load full page image
        img = Image.open(image_path)

        # Extract bounding box coordinates
        x1 = int(bbox_coords["bbox_x1"])
        y1 = int(bbox_coords["bbox_y1"])
        x2 = int(bbox_coords["bbox_x2"])
        y2 = int(bbox_coords["bbox_y2"])

        # Crop to picture region
        cropped = img.crop((x1, y1, x2, y2))

        # Scale down if too wide
        if cropped.width > max_width:
            scale = max_width / cropped.width
            new_height = int(cropped.height * scale)
            cropped = cropped.resize((max_width, new_height), Image.Resampling.LANCZOS)

        # Convert to base64 for inline display
        buffer = io.BytesIO()
        cropped.save(buffer, format="JPEG", quality=85)
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.read()).decode()

        return f"data:image/jpeg;base64,{img_base64}"

    except Exception as e:
        logger.error(f"Error cropping picture region: {e}")
        return ""


def load_result_metadata(parquet_path: str) -> dict:
    """Load metadata JSON file associated with a parquet file"""
    json_path = Path(parquet_path).parent / "layout.json"
    if json_path.exists():
        with open(json_path) as f:
            return json.load(f)
    return {}


async def create_pictures_tab(state: AppState):
    """
    Create the pictures gallery tab - filtered view of Picture detections with captions

    Args:
        state: AppState instance
    """
    ui.label("Picture Gallery").classes("text-h4 q-mb-md")

    # Create a container for the entire content that we can refresh
    content_container = ui.column().classes("w-full")

    async def refresh_pictures_tab():
        """Refresh the entire pictures tab content"""
        # Re-fetch available files each time we refresh
        available_files = state.get_available_layout_files()
        content_container.clear()

        with content_container:
            await _create_pictures_content(state, available_files, refresh_pictures_tab)

    # Initial render
    await refresh_pictures_tab()


async def _create_pictures_content(
    state: AppState, available_files: list, refresh_callback
) -> None:
    """Create the actual pictures content (separated for easy refresh)"""

    if len(available_files) == 0:
        with ui.card().classes("w-full q-mb-md"):
            with ui.row().classes("items-center q-pa-md"):
                ui.icon("info", size="2em").classes("text-blue q-mr-md")
                with ui.column():
                    ui.label("No layout detection results found").classes("text-subtitle1")
                    ui.label(
                        f"Run layout detection first to extract pictures: newspaper-explorer analyze layout detect --source {state.selected_source}"
                    ).classes("text-caption text-grey-7")
        return

    # Selector card
    with ui.card().classes("w-full q-mb-md"):
        with ui.column().classes("w-full q-pa-md gap-3"):
            # Top row: selector and metadata
            with ui.row().classes("w-full items-start gap-4"):
                # Left side: selector
                with (
                    ui.column()
                    .classes("gap-2 flex-shrink-0")
                    .style("min-width: 350px; max-width: 450px;")
                ):
                    ui.label("Layout Dataset").classes("text-subtitle2 font-semibold")

                    # Create options dict for select
                    file_options = {
                        file_path: display_name
                        for display_name, file_path, metadata in available_files
                    }

                    # Get currently loaded file path or default to first
                    current_file = None
                    if state.layout_df is not None and available_files:
                        # Try to preserve the currently loaded file
                        current_file = getattr(state, "_current_layout_file", available_files[0][1])
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
                    async def reload_pictures():
                        """Reload pictures with selected file"""
                        if file_select.value:
                            state.load_layout(file_select.value)
                            # Store the currently loaded file so we can preserve selection
                            state._current_layout_file = file_select.value  # type: ignore
                            ui.notify(f"Loaded picture data", type="positive")
                            # Refresh the entire tab to show new data
                            await refresh_callback()

                    ui.button("Load", icon="refresh", on_click=reload_pictures).props(
                        "outline dense"
                    )

                # Right side: metadata
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

                # Analysis ID
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
                """Display inline statistics summary for pictures only"""
                if df is None or len(df) == 0:
                    ui.label("No data").classes("text-caption text-grey-5")
                    return

                # Filter to pictures only
                pictures_df = df.filter(pl.col("class_name") == "Picture")

                if len(pictures_df) == 0:
                    ui.label("No pictures found").classes("text-caption text-grey-5")
                    return

                # Compact stats in a single line
                stats_parts = []

                # Total pictures
                stats_parts.append(f"{len(pictures_df):,} pictures")

                # Unique pages
                stats_parts.append(f"{pictures_df['page_id'].n_unique()} pages")

                # Date range
                if "date" in pictures_df.columns:
                    min_date = str(pictures_df["date"].min()).split()[0]
                    max_date = str(pictures_df["date"].max()).split()[0]
                    if min_date == max_date:
                        stats_parts.append(f"{min_date}")
                    else:
                        stats_parts.append(f"{min_date} → {max_date}")

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

    # Get layout data and filter to pictures only
    df = state.layout_df if state.layout_df is not None else None

    if df is None or len(df) == 0:
        with ui.card().classes("w-full q-mb-md"):
            with ui.row().classes("items-center q-pa-md"):
                ui.icon("info", size="2em").classes("text-blue q-mr-md")
                ui.label("No data loaded").classes("text-subtitle1")
        return

    # Filter to Picture class only
    pictures_df = df.filter(pl.col("class_name") == "Picture")

    if len(pictures_df) == 0:
        with ui.card().classes("w-full q-mb-md"):
            with ui.row().classes("items-center q-pa-md"):
                ui.icon("info", size="2em").classes("text-blue q-mr-md")
                ui.label("No pictures found in this dataset").classes("text-subtitle1")
        return

    # Search and filter section
    search_container = ui.row().classes("w-full q-mb-md gap-4")
    gallery_container = ui.column().classes("w-full")

    with search_container:
        with ui.card().classes("w-full"):
            with ui.card_section().classes("q-pa-md"):
                ui.label("Search & Filter").classes("text-subtitle1 q-mb-md")

                # Row 1: Search and date range
                with ui.row().classes("w-full gap-4 items-end q-mb-md"):
                    # Search input
                    search_input = (
                        ui.input(placeholder="Search by date, page, or caption text...")
                        .classes("flex-grow")
                        .props("outlined dense clearable")
                    )

                    # Date range filters
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

                # Row 2: Advanced filters
                with ui.row().classes("w-full gap-6 items-center"):
                    # Confidence slider
                    with ui.row().classes("gap-2 items-center").style("min-width: 300px;"):
                        with ui.column().classes("gap-1 flex-grow"):
                            ui.label("Minimum Confidence").classes("text-caption text-grey-7")
                            confidence_slider = ui.slider(
                                min=0.0, max=1.0, step=0.05, value=0.25
                            ).classes("w-full")
                        confidence_label = (
                            ui.label("0.25")
                            .classes("text-body2 font-semibold")
                            .style("min-width: 40px;")
                        )
                        confidence_slider.on_value_change(
                            lambda e: confidence_label.set_text(f"{e.value:.2f}")
                        )

                    # Exclude headers checkbox
                    exclude_headers = (
                        ui.checkbox("Exclude headers/footers", value=True)
                        .props("dense")
                        .classes("q-mt-sm")
                    )

                    # Y-coordinate threshold for headers/footers (as percentage of page height)
                    with ui.row().classes("gap-2 items-center").style("min-width: 250px;"):
                        with ui.column().classes("gap-1 flex-grow"):
                            ui.label("Header/Footer Threshold (%)").classes(
                                "text-caption text-grey-7"
                            )
                            header_threshold = ui.slider(min=0, max=30, step=1, value=10).classes(
                                "w-full"
                            )
                        threshold_label = (
                            ui.label("10")
                            .classes("text-body2 font-semibold")
                            .style("min-width: 30px;")
                        )
                        header_threshold.on_value_change(
                            lambda e: threshold_label.set_text(f"{int(e.value)}")
                        )

                # Apply button
                with ui.row().classes("w-full justify-end q-mt-sm"):

                    async def apply_search():
                        """Apply search and filters"""
                        filtered = pictures_df

                        # Confidence filter
                        if confidence_slider.value:
                            filtered = filtered.filter(
                                pl.col("confidence") >= confidence_slider.value
                            )

                        # Date filters
                        if date_from.value:
                            try:
                                from_date = datetime.strptime(date_from.value, "%Y-%m-%d").date()
                                filtered = filtered.filter(pl.col("date") >= from_date)
                            except:
                                pass

                        if date_to.value:
                            try:
                                to_date = datetime.strptime(date_to.value, "%Y-%m-%d").date()
                                filtered = filtered.filter(pl.col("date") <= to_date)
                            except:
                                pass

                        # Exclude headers/footers based on Y-coordinate
                        if exclude_headers.value and header_threshold.value > 0:
                            # Calculate threshold as percentage of page height
                            # Assume typical newspaper page height (will be more accurate with actual page dimensions)
                            # Filter out pictures in top/bottom X% of page
                            threshold_pct = header_threshold.value / 100.0

                            # Get approximate page heights from data
                            # bbox_y1 near 0 = top of page (header)
                            # bbox_y2 near page_height = bottom of page (footer)

                            # For each page, calculate thresholds
                            # Simple heuristic: exclude if y1 < 10% of max_y OR y2 > 90% of max_y
                            # Get max Y coordinate per page to estimate page height
                            page_heights = filtered.group_by("page_id").agg(
                                pl.col("bbox_y2").max().alias("page_height")
                            )

                            # Join page heights back to filtered data
                            filtered = filtered.join(page_heights, on="page_id", how="left")

                            # Filter out headers (top X%) and footers (bottom X%)
                            filtered = filtered.filter(
                                (pl.col("bbox_y1") > pl.col("page_height") * threshold_pct)
                                & (pl.col("bbox_y2") < pl.col("page_height") * (1 - threshold_pct))
                            )

                            # Drop the temporary page_height column
                            filtered = filtered.drop("page_height")

                        # Search filter (simple text search in page_id for now)
                        if search_input.value:
                            search_term = search_input.value.lower()
                            filtered = filtered.filter(
                                pl.col("page_id").str.to_lowercase().str.contains(search_term)
                            )

                        update_gallery(filtered)

                    ui.button("Apply Filters", icon="filter_alt", on_click=apply_search).props(
                        "outline dense"
                    )

    def update_gallery(df_filtered):
        """Update picture gallery with filtered data"""
        gallery_container.clear()

        with gallery_container:
            if df_filtered is None or len(df_filtered) == 0:
                ui.label("No pictures match your search").classes("text-gray-500")
                return

            # Sort by page_id (chronological)
            df_sorted = df_filtered.sort("page_id")

            ui.label(f"Found {len(df_sorted):,} pictures").classes("text-subtitle1 q-mb-md")

            # Pagination
            items_per_page = 12
            total_items = len(df_sorted)
            total_pages = (total_items + items_per_page - 1) // items_per_page
            current_page = [1]

            page_container = ui.column().classes("w-full")
            pagination_controls = ui.row().classes("w-full justify-center items-center gap-4 mt-4")

            def show_page(page_num: int):
                """Display pictures for current page"""
                current_page[0] = page_num
                start_idx = (page_num - 1) * items_per_page
                end_idx = min(start_idx + items_per_page, total_items)

                # Get pictures for this page
                page_pictures = df_sorted.slice(start_idx, end_idx - start_idx)

                page_container.clear()
                with page_container:
                    with ui.row().classes("w-full gap-4 flex-wrap"):
                        for row in page_pictures.iter_rows(named=True):
                            create_picture_card(row)

                # Update pagination controls
                pagination_controls.clear()
                with pagination_controls:
                    prev_btn = ui.button(
                        icon="chevron_left", on_click=lambda: show_page(page_num - 1)
                    ).props("flat round")
                    prev_btn.set_enabled(page_num > 1)

                    ui.label(f"Page {page_num} of {total_pages}").classes("text-caption")

                    next_btn = ui.button(
                        icon="chevron_right", on_click=lambda: show_page(page_num + 1)
                    ).props("flat round")
                    next_btn.set_enabled(page_num < total_pages)

            def create_picture_card(picture_row: dict):
                """Create a card for a single picture"""
                page_id = picture_row["page_id"]
                image_path = picture_row.get("image_path", "")
                confidence = picture_row["confidence"]

                # Get bounding box coordinates
                bbox_coords = {
                    "bbox_x1": picture_row["bbox_x1"],
                    "bbox_y1": picture_row["bbox_y1"],
                    "bbox_x2": picture_row["bbox_x2"],
                    "bbox_y2": picture_row["bbox_y2"],
                }

                # Parse metadata from page_id
                parts = page_id.split("_")
                if len(parts) >= 5:
                    date_str = parts[1]  # YYYY-MM-DD
                    issue = parts[2]
                    daily = parts[3]
                    page = parts[4]

                    try:
                        date_obj = datetime.strptime(date_str, "%Y-%m-%d")
                        weekday = date_obj.strftime("%A")
                        display_date = f"{date_obj.strftime('%d.%m.%Y')} ({weekday})"
                    except:
                        display_date = date_str

                    metadata_str = f"Issue {issue} • Daily {daily} • Page {page}"
                else:
                    display_date = "Unknown date"
                    metadata_str = page_id

                # Create card with flexible width based on image aspect ratio
                with ui.card().style("cursor: pointer; max-width: 400px;"):
                    # Cropped picture image
                    if image_path and Path(image_path).exists():
                        cropped_img = crop_picture_region(image_path, bbox_coords, max_width=400)
                        if cropped_img:
                            # Image with natural aspect ratio, max height constraint
                            ui.image(cropped_img).classes("w-full").style(
                                "max-height: 500px; object-fit: contain; background: #f5f5f5;"
                            )
                        else:
                            # Fallback to placeholder if crop failed
                            with (
                                ui.column()
                                .classes("w-full items-center justify-center bg-gray-100")
                                .style("height: 300px;")
                            ):
                                ui.icon("image", size="4em").classes("text-gray-400")
                                ui.label("Could not crop image").classes(
                                    "text-caption text-gray-500"
                                )
                    else:
                        # Placeholder
                        with (
                            ui.column()
                            .classes("w-full items-center justify-center bg-gray-100")
                            .style("height: 300px;")
                        ):
                            ui.icon("image", size="4em").classes("text-gray-400")
                            ui.label("Image not available").classes("text-caption text-gray-500")

                    # Metadata
                    with ui.card_section().classes("q-pa-sm"):
                        ui.label(display_date).classes("text-caption font-semibold")
                        ui.label(metadata_str).classes("text-caption text-grey-7")
                        ui.label(f"Confidence: {confidence:.2f}").classes(
                            "text-caption text-grey-6"
                        )

                        # TODO: Add caption extraction when available
                        # For now, show placeholder
                        with (
                            ui.expansion("Caption", icon="description")
                            .props("dense")
                            .classes("q-mt-xs")
                        ):
                            ui.label("Caption extraction not yet implemented").classes(
                                "text-caption text-grey-5"
                            )

            # Show first page
            show_page(1)

    # Initial display
    update_gallery(pictures_df)
