"""
Browse tab for the NiceGUI interface - Browse newspaper collection chronologically
"""

from nicegui import ui
from pathlib import Path
from datetime import datetime
import polars as pl


async def create_browse_tab(state):
    """
    Create the browse tab for chronological navigation

    Args:
        state: AppState instance
    """
    # State variables for navigation
    current_view = {"mode": "list"}  # "list" or "issue"
    current_page = {"value": 1}
    current_filters = {"year": None, "month": None}  # For drill-down navigation
    items_per_page = 20

    # Containers that will be initialized later
    breadcrumb_container: ui.column = None  # type: ignore
    result_container: ui.column = None  # type: ignore
    stats_container: ui.column = None  # type: ignore
    year_filter_container: ui.column = None  # type: ignore
    apply_btn_container: ui.column = None  # type: ignore
    apply_filters = None  # type: ignore

    # Store references to year controls
    year_from = None
    year_to = None
    sort_order = None

    def navigate_to_year(year):
        """Navigate to a specific year or back to all years"""
        current_filters["year"] = year
        current_filters["month"] = None
        current_page["value"] = 1
        # Update sidebar year filter and breadcrumb, then refresh content
        if (
            year_filter_container is not None
            and breadcrumb_container is not None
            and result_container is not None
            and apply_filters is not None
        ):
            # Store reference to apply_filters before clearing UI
            apply_fn = apply_filters

            update_year_filter()
            update_apply_button()
            update_breadcrumb()

            # Create task in the result_container context
            with result_container:
                ui.timer(0.01, lambda: apply_fn(1), once=True)  # type: ignore
        else:
            render_main_view()

    def navigate_to_month(year, month):
        """Navigate to a specific month (or all months in a year)"""
        # Check if we're already at month level (filter is set) - if so, just update
        # Otherwise update sidebar controls too
        was_at_month_level = current_filters["year"] is not None

        current_filters["year"] = year
        current_filters["month"] = month
        current_page["value"] = 1

        # Only do partial update if we were already at month level
        if (
            was_at_month_level
            and breadcrumb_container is not None
            and result_container is not None
            and apply_filters is not None
        ):
            # Store reference to apply_filters before clearing UI
            apply_fn = apply_filters

            update_breadcrumb()
            # Create task in the result_container context
            with result_container:
                ui.timer(0.01, lambda: apply_fn(1), once=True)  # type: ignore
        else:
            # Need to update sidebar controls (year range -> year dropdown)
            if (
                year_filter_container is not None
                and breadcrumb_container is not None
                and result_container is not None
                and apply_filters is not None
            ):
                # Store reference to apply_filters before clearing UI
                apply_fn = apply_filters

                update_year_filter()
                update_apply_button()
                update_breadcrumb()
                # Create task in the result_container context
                with result_container:
                    ui.timer(0.01, lambda: apply_fn(1), once=True)  # type: ignore
            else:
                render_main_view()

    def show_issues_for_month(year, month):
        """Show individual issues for a specific month"""
        current_filters["year"] = year
        current_filters["month"] = month
        current_page["value"] = 1
        # Only update breadcrumb and content, not full re-render
        if (
            breadcrumb_container is not None
            and result_container is not None
            and apply_filters is not None
        ):
            # Store reference to apply_filters before clearing UI
            apply_fn = apply_filters

            update_breadcrumb()
            # Create task in the result_container context
            with result_container:
                ui.timer(0.01, lambda: apply_fn(1), once=True)  # type: ignore
        else:
            render_main_view()

    def update_breadcrumb():
        """Update the breadcrumb navigation without full re-render"""
        breadcrumb_container.clear()
        with breadcrumb_container:
            # Get source title for display
            source_title = "Collection"
            if state.source_config and state.source_config.metadata:
                source_title = state.source_config.metadata.newspaper_title

            with ui.row().classes("items-center gap-2 q-mb-md wrap"):
                if current_filters["year"] is None:
                    ui.button(
                        source_title,
                        icon="home",
                        on_click=lambda: navigate_to_year(None),
                    ).props("flat").classes("text-h6 text-black")
                else:
                    ui.button(
                        source_title,
                        icon="home",
                        on_click=lambda: navigate_to_year(None),
                    ).props("flat").classes("text-h6 text-black")

                if current_filters["year"] is not None:
                    ui.label("›").classes("text-h5 text-grey-5")
                    if current_filters["month"] is None:
                        ui.label(f"{current_filters['year']}").classes("text-h5")
                    else:
                        # Capture year value at lambda creation time
                        year_value = current_filters["year"]
                        ui.button(
                            f"{year_value}",
                            on_click=lambda y=year_value: navigate_to_month(y, None),
                        ).props("flat").classes("text-h6 text-black")

                    if current_filters["month"] is not None:
                        month_names = [
                            "January",
                            "February",
                            "March",
                            "April",
                            "May",
                            "June",
                            "July",
                            "August",
                            "September",
                            "October",
                            "November",
                            "December",
                        ]
                        ui.label("›").classes("text-h5 text-grey-5")
                        ui.label(f"{month_names[current_filters['month'] - 1]}").classes("text-h5")

    def update_year_filter():
        """Update the year filter controls without full re-render"""
        nonlocal year_from, year_to, sort_order

        year_filter_container.clear()
        with year_filter_container:
            # Get date range from data
            stats = state.data_loader.get_stats()
            min_year = 1900
            max_year = 2000
            if stats["min_date"] != "N/A" and stats["max_date"] != "N/A":
                min_year = datetime.strptime(stats["min_date"], "%Y-%m-%d").year
                max_year = datetime.strptime(stats["max_date"], "%Y-%m-%d").year

            # Set filter values based on drill-down state
            if current_filters["year"] is not None:
                # When drilling down, lock to that year
                filter_year_from = current_filters["year"]
                filter_year_to = current_filters["year"]
            else:
                # Use full range
                filter_year_from = min_year
                filter_year_to = max_year

            # Date range filter
            ui.label("Time Period").classes("text-subtitle2 q-mb-xs")

            # Always show year dropdown for all levels
            # Get available years from data
            years_result = state.data_loader.db.execute(
                f"""
                SELECT DISTINCT year
                FROM read_parquet('{state.data_loader.parquet_path}')
                ORDER BY year ASC
                """
            ).fetchall()
            available_years = [int(row[0]) for row in years_result if row[0]]

            # Add "All Years" option for year level
            if current_filters["year"] is None:
                # At year level - show all years option or range selection
                with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                    year_from = ui.number(
                        label="From",
                        value=filter_year_from,
                        min=min_year,
                        max=max_year,
                        format="%.0f",
                    ).classes("col")
                    ui.label("-").classes("text-grey-7")
                    year_to = ui.number(
                        label="To",
                        value=filter_year_to,
                        min=min_year,
                        max=max_year,
                        format="%.0f",
                    ).classes("col")
            else:
                # At month/issue level - show year dropdown
                year_from = None
                year_to = None
                year_select = ui.select(
                    label="Year",
                    options=available_years,
                    value=current_filters["year"],
                    on_change=lambda e: navigate_to_month(e.value, None),
                ).classes("w-full q-mb-md")

            # Sort order - always shown
            ui.label("Sort Order").classes("text-subtitle2 q-mb-xs q-mt-md")
            sort_order = ui.select(
                label="Order",
                options=["Newest First", "Oldest First"],
                value="Oldest First",
            ).classes("w-full q-mb-md")

            # Auto-apply when sort order changes and we're drilling down (month/issue level)
            if current_filters["year"] is not None and apply_filters is not None:
                sort_order.on("change", lambda: ui.timer(0.01, lambda: apply_filters(1), once=True))  # type: ignore

            # Reset filters button - always show
            def reset_filters():
                if year_from is not None and year_to is not None:
                    year_from.set_value(min_year)
                    year_to.set_value(max_year)
                if sort_order is not None:  # type: ignore
                    sort_order.set_value("Oldest First")
                # If at month/issue level, go back to year level
                if current_filters["year"] is not None:
                    navigate_to_year(None)

            ui.button("Reset Filters", icon="refresh").props("flat color=grey-7").classes(
                "w-full q-mt-sm"
            ).on("click", reset_filters)

            ui.separator()

    def update_apply_button():
        """Update the apply button visibility"""
        apply_btn_container.clear()
        with apply_btn_container:
            # Apply button - only show at year level
            if current_filters["year"] is None:
                apply_btn = (
                    ui.button("Apply Filters", icon="filter_list")
                    .props("color=primary")
                    .classes("w-full q-mt-md")
                )
                # Connect the button after creation
                if apply_filters is not None:
                    apply_btn.on("click", lambda: apply_filters(1))  # type: ignore

    def render_browse_list():
        """Render the main browse list view"""
        nonlocal breadcrumb_container, result_container, stats_container, year_filter_container, apply_btn_container, apply_filters

        with ui.column().classes("w-full"):
            # Breadcrumb navigation container
            breadcrumb_container = ui.column().classes("w-full")
            update_breadcrumb()

            if not state.selected_source:
                with ui.card().classes("w-full"):
                    with ui.card_section():
                        ui.label(
                            "Please select a source from the sidebar to browse newspapers"
                        ).classes("text-grey-7")
                return

            # Check if data is available
            if not state.data_loader or not state.data_loader.parquet_exists():
                with ui.card().classes("w-full"):
                    with ui.card_section():
                        ui.label("No data available for the selected source").classes("text-grey-7")
                return

            # Main layout with filters on left and content on right
            with ui.row().classes("w-full gap-4"):
                # Left sidebar - Browse filters
                with ui.column().classes("w-60 flex-shrink-0"):
                    with ui.card().classes("w-full"):
                        with ui.card_section():
                            ui.label("Browse Filters").classes("text-h6 q-mb-md")

                            # Year filter container (will be updated dynamically)
                            year_filter_container = ui.column().classes("w-full")
                            update_year_filter()

                            # Apply button container (will be updated dynamically)
                            apply_btn_container = ui.column().classes("w-full")
                            update_apply_button()

                    # Stats container in sidebar (will be updated by apply_filters)
                    with ui.card().classes("w-full q-mt-sm"):
                        with ui.card_section().classes("q-pa-sm"):
                            nonlocal stats_container
                            stats_container = ui.column().classes("w-full gap-1")

                # Right content area - Browse results
                with ui.column().classes("flex-1"):
                    # Results container
                    nonlocal result_container, apply_filters
                    result_container = ui.column().classes("w-full gap-4")

                    async def apply_filters_impl(page=1):
                        """Apply browse filters and display results"""
                        result_container.clear()
                        stats_container.clear()
                        current_page["value"] = page

                        with result_container:
                            # Build query based on grouping
                            parquet_path = state.data_loader.parquet_path
                            db = state.data_loader.db

                            try:
                                # Determine what to show based on drill-down state
                                # If we have year+month filters, show issues
                                # If we have year filter only, show months
                                # Otherwise, show years or whatever grouping is selected

                                if (
                                    current_filters["year"] is not None
                                    and current_filters["month"] is not None
                                ):
                                    # Show issues for specific month
                                    show_view = "Issue"
                                elif current_filters["year"] is not None:
                                    # Show months for specific year
                                    show_view = "Month"
                                else:
                                    # Always show years at top level (group by dropdown removed)
                                    show_view = "Year"

                                if show_view == "Year":
                                    # Get date range from data for defaults
                                    stats = state.data_loader.get_stats()
                                    min_year = 1900
                                    max_year = 2000
                                    if stats["min_date"] != "N/A" and stats["max_date"] != "N/A":
                                        min_year = datetime.strptime(
                                            stats["min_date"], "%Y-%m-%d"
                                        ).year
                                        max_year = datetime.strptime(
                                            stats["max_date"], "%Y-%m-%d"
                                        ).year

                                    # Query for yearly grouping
                                    sort_dir = (
                                        "DESC"
                                        if sort_order and sort_order.value == "Newest First"
                                        else "ASC"
                                    )

                                    # Get year range from filters or UI
                                    query_year_from = min_year
                                    query_year_to = max_year
                                    if year_from is not None and year_to is not None:
                                        query_year_from = int(year_from.value)
                                        query_year_to = int(year_to.value)

                                    result = db.execute(
                                        f"""
                                        SELECT 
                                            year,
                                            COUNT(*) as line_count,
                                            COUNT(DISTINCT date) as unique_dates,
                                            COUNT(DISTINCT issue_id) as issue_count
                                        FROM read_parquet('{parquet_path}')
                                        WHERE year >= {query_year_from}
                                          AND year <= {query_year_to}
                                        GROUP BY year
                                        ORDER BY year {sort_dir}
                                    """
                                    ).fetchall()

                                    # Get random images for each year (if available)
                                    year_images = {}
                                    if state.image_indexer:
                                        index = state.image_indexer.load_index()
                                        if index is not None and len(index) > 0:
                                            for row in result:
                                                year = int(row[0]) if row[0] else 0
                                                # Get one random image from this year
                                                year_imgs = index.filter(pl.col("year") == year)
                                                if len(year_imgs) > 0:
                                                    # Get random sample
                                                    sample = year_imgs.sample(n=1)
                                                    year_images[year] = sample["image_path"][0]

                                    if not result:
                                        with ui.card().classes("w-full"):
                                            with ui.card_section():
                                                ui.label(
                                                    "No results found for the selected filters"
                                                ).classes("text-grey-7")
                                        return

                                    total_lines = 0
                                    # Display year groups in a grid
                                    with ui.grid(
                                        columns="repeat(auto-fill, minmax(400px, 1fr))"
                                    ).classes("w-full gap-4"):
                                        for row in result:
                                            year = int(row[0]) if row[0] else 0
                                            line_count = row[1]
                                            unique_dates = row[2]
                                            issue_count = row[3]
                                            total_lines += line_count

                                            with (
                                                ui.card()
                                                .classes("hover:shadow-lg cursor-pointer")
                                                .props("flat bordered")
                                                .style(
                                                    "border: 1px solid #e0e0e0; border-radius: 4px; overflow: hidden;"
                                                )
                                                .on(
                                                    "click",
                                                    lambda y=year: navigate_to_month(y, None),
                                                )
                                            ):
                                                with (
                                                    ui.row()
                                                    .classes("w-full")
                                                    .style("height: 200px;")
                                                ):
                                                    # Left column - year info
                                                    with (
                                                        ui.column()
                                                        .classes(
                                                            "items-center justify-center text-center q-pa-md"
                                                        )
                                                        .style("flex: 1; height: 100%;")
                                                    ):
                                                        ui.label(f"{year}").classes(
                                                            "text-h3 text-grey-9"
                                                        ).style("line-height: 1; font-weight: 600;")
                                                        ui.separator().classes("q-my-xs").style(
                                                            "width: 60%;"
                                                        )
                                                        with ui.column().classes(
                                                            "gap-1 w-full items-center"
                                                        ):
                                                            ui.label(
                                                                f"{issue_count} issues"
                                                            ).classes(
                                                                "text-body1 text-grey-7"
                                                            ).style(
                                                                "font-weight: 500;"
                                                            )
                                                            ui.label(
                                                                f"{line_count:,} lines"
                                                            ).classes("text-body2 text-grey-6")

                                                    # Right column - image
                                                    with ui.element("div").style(
                                                        "flex: 1; background: #e0e0e0; position: relative; overflow: hidden; height: 100%;"
                                                    ):
                                                        img_path = year_images.get(year)
                                                        if img_path:
                                                            ui.image(
                                                                f"/static/{state.selected_source}/images/{img_path}"
                                                            ).style(
                                                                "width: 100%; height: 100%; object-fit: cover; object-position: center;"
                                                            )
                                                        else:
                                                            pass

                                    # Show total count in sidebar
                                    with stats_container:
                                        ui.label("Statistics").classes("text-subtitle2 q-mb-xs")
                                        ui.label(f"{len(result)} years").classes(
                                            "text-body2 text-grey-8"
                                        )
                                        ui.label(f"{total_lines:,} total lines").classes(
                                            "text-caption text-grey-7"
                                        )

                                elif show_view == "Month":
                                    # Get date range from data for defaults
                                    stats = state.data_loader.get_stats()
                                    min_year = 1900
                                    max_year = 2000
                                    if stats["min_date"] != "N/A" and stats["max_date"] != "N/A":
                                        min_year = datetime.strptime(
                                            stats["min_date"], "%Y-%m-%d"
                                        ).year
                                        max_year = datetime.strptime(
                                            stats["max_date"], "%Y-%m-%d"
                                        ).year

                                    # Query for monthly grouping with pagination
                                    # If drill-down year is set, only show months for that year
                                    sort_dir = (
                                        "DESC"
                                        if sort_order and sort_order.value == "Newest First"
                                        else "ASC"
                                    )

                                    # Build WHERE clause based on filters
                                    if current_filters["year"] is not None:
                                        year_filter_min = current_filters["year"]
                                        year_filter_max = current_filters["year"]
                                    elif year_from is not None and year_to is not None:
                                        year_filter_min = int(year_from.value)
                                        year_filter_max = int(year_to.value)
                                    else:
                                        year_filter_min = min_year
                                        year_filter_max = max_year

                                    # Get total count first
                                    count_result = db.execute(
                                        f"""
                                        SELECT COUNT(*) FROM (
                                            SELECT year, month
                                            FROM read_parquet('{parquet_path}')
                                            WHERE year >= {year_filter_min}
                                              AND year <= {year_filter_max}
                                            GROUP BY year, month
                                        )
                                    """
                                    ).fetchone()
                                    total_count = count_result[0] if count_result else 0
                                    total_pages = (
                                        total_count + items_per_page - 1
                                    ) // items_per_page

                                    offset = (page - 1) * items_per_page
                                    result = db.execute(
                                        f"""
                                        SELECT 
                                            year,
                                            month,
                                            COUNT(*) as line_count,
                                            COUNT(DISTINCT date) as unique_dates,
                                            COUNT(DISTINCT issue_id) as issue_count
                                        FROM read_parquet('{parquet_path}')
                                        WHERE year >= {year_filter_min}
                                          AND year <= {year_filter_max}
                                        GROUP BY year, month
                                        ORDER BY year {sort_dir}, month {sort_dir}
                                        LIMIT {items_per_page} OFFSET {offset}
                                    """
                                    ).fetchall()

                                    if not result:
                                        with ui.card().classes("w-full"):
                                            with ui.card_section():
                                                ui.label(
                                                    "No results found for the selected filters"
                                                ).classes("text-grey-7")
                                        return

                                    total_lines = 0
                                    month_names = [
                                        "Jan",
                                        "Feb",
                                        "Mar",
                                        "Apr",
                                        "May",
                                        "Jun",
                                        "Jul",
                                        "Aug",
                                        "Sep",
                                        "Oct",
                                        "Nov",
                                        "Dec",
                                    ]

                                    # Display month groups in a grid
                                    with ui.grid(
                                        columns="repeat(auto-fill, minmax(180px, 1fr))"
                                    ).classes("w-full gap-4"):
                                        for row in result:
                                            year = int(row[0]) if row[0] else 0
                                            month = int(row[1]) if row[1] else 1
                                            line_count = row[2]
                                            unique_dates = row[3]
                                            issue_count = row[4]
                                            total_lines += line_count

                                            with (
                                                ui.card()
                                                .classes("hover:shadow-lg cursor-pointer")
                                                .props("flat bordered")
                                                .style(
                                                    "border: 1px solid #e0e0e0; border-radius: 4px;"
                                                )
                                            ):
                                                with (
                                                    ui.card_section()
                                                    .classes("q-pa-sm")
                                                    .on(
                                                        "click",
                                                        lambda y=year, m=month: show_issues_for_month(
                                                            y, m
                                                        ),
                                                    )
                                                ):
                                                    with (
                                                        ui.column()
                                                        .classes(
                                                            "gap-1 items-center justify-center text-center w-full"
                                                        )
                                                        .style("height: 100%;")
                                                    ):
                                                        ui.label(
                                                            f"{month_names[month - 1]}"
                                                        ).classes("text-h4 text-grey-9").style(
                                                            "line-height: 1;"
                                                        )
                                                        if current_filters["year"] is None:
                                                            ui.label(f"{year}").classes(
                                                                "text-body2 text-grey-7"
                                                            ).style("font-weight: 500;")
                                                        ui.separator().classes("q-my-xs")
                                                        with ui.column().classes(
                                                            "gap-1 w-full items-center"
                                                        ):
                                                            ui.label(
                                                                f"{issue_count} issues"
                                                            ).classes(
                                                                "text-body2 text-grey-7"
                                                            ).style(
                                                                "font-weight: 500;"
                                                            )
                                                            ui.label(
                                                                f"{line_count:,} lines"
                                                            ).classes("text-caption text-grey-6")

                                    # Pagination controls
                                    if total_pages > 1:
                                        with ui.row().classes(
                                            "w-full justify-center items-center gap-2 q-mt-md"
                                        ):
                                            ui.button(
                                                icon="chevron_left",
                                                on_click=lambda: apply_filters_impl(
                                                    max(1, page - 1)
                                                ),
                                            ).props("flat round").set_enabled(page > 1)

                                            ui.label(f"Page {page} of {total_pages}").classes(
                                                "text-caption"
                                            )

                                            ui.button(
                                                icon="chevron_right",
                                                on_click=lambda: apply_filters_impl(
                                                    min(total_pages, page + 1)
                                                ),
                                            ).props("flat round").set_enabled(page < total_pages)

                                    # Show total count in sidebar
                                    with stats_container:
                                        ui.label("Statistics").classes("text-subtitle2 q-mb-xs")
                                        ui.label(f"Page {page} of {total_pages}").classes(
                                            "text-body2 text-grey-8"
                                        )
                                        ui.label(f"{len(result)} months on this page").classes(
                                            "text-caption text-grey-7"
                                        )
                                        ui.label(f"{total_lines:,} lines on this page").classes(
                                            "text-caption text-grey-7"
                                        )

                                else:  # Issue
                                    # Get date range from data for defaults
                                    stats = state.data_loader.get_stats()
                                    min_year = 1900
                                    max_year = 2000
                                    if stats["min_date"] != "N/A" and stats["max_date"] != "N/A":
                                        min_year = datetime.strptime(
                                            stats["min_date"], "%Y-%m-%d"
                                        ).year
                                        max_year = datetime.strptime(
                                            stats["max_date"], "%Y-%m-%d"
                                        ).year

                                    # Query for individual issues with pagination
                                    # Filter by year and optionally month if drilling down
                                    sort_dir = (
                                        "DESC"
                                        if sort_order and sort_order.value == "Newest First"
                                        else "ASC"
                                    )

                                    # Build WHERE clause based on drill-down filters
                                    if current_filters["year"] is not None:
                                        year_filter_min = current_filters["year"]
                                        year_filter_max = current_filters["year"]
                                    elif year_from is not None and year_to is not None:
                                        year_filter_min = int(year_from.value)
                                        year_filter_max = int(year_to.value)
                                    else:
                                        year_filter_min = min_year
                                        year_filter_max = max_year

                                    where_clauses = [
                                        f"year >= {year_filter_min}",
                                        f"year <= {year_filter_max}",
                                    ]

                                    if current_filters["month"] is not None:
                                        where_clauses.append(f"month = {current_filters['month']}")

                                    where_clause = " AND ".join(where_clauses)

                                    # Get total count first
                                    count_result = db.execute(
                                        f"""
                                        SELECT COUNT(DISTINCT issue_id)
                                        FROM read_parquet('{parquet_path}')
                                        WHERE {where_clause}
                                    """
                                    ).fetchone()
                                    total_count = count_result[0] if count_result else 0
                                    total_pages = (
                                        total_count + items_per_page - 1
                                    ) // items_per_page

                                    offset = (page - 1) * items_per_page
                                    result = db.execute(
                                        f"""
                                        SELECT 
                                            issue_id,
                                            MIN(date) as date,
                                            MIN(newspaper_title) as title,
                                            MIN(year_volume) as year_volume,
                                            COUNT(*) as line_count,
                                            COUNT(DISTINCT text_block_id) as block_count,
                                            MIN(page_count) as page_count
                                        FROM read_parquet('{parquet_path}')
                                        WHERE {where_clause}
                                        GROUP BY issue_id
                                        ORDER BY MIN(date) {sort_dir}, issue_id {sort_dir}
                                        LIMIT {items_per_page} OFFSET {offset}
                                    """
                                    ).fetchall()

                                    if not result:
                                        with ui.card().classes("w-full"):
                                            with ui.card_section():
                                                ui.label(
                                                    "No results found for the selected filters"
                                                ).classes("text-grey-7")
                                        return

                                    # Get first page image for each issue (if available)
                                    issue_images = {}
                                    if state.image_indexer:
                                        index = state.image_indexer.load_index()
                                        if index is not None and len(index) > 0:
                                            for row in result:
                                                issue_id = row[0]

                                                # Match by issue_id now that we have proper format
                                                issue_imgs = index.filter(
                                                    (pl.col("issue_id") == issue_id)
                                                    & (pl.col("page_number") == 1)
                                                )
                                                if len(issue_imgs) > 0:
                                                    issue_images[issue_id] = issue_imgs[
                                                        "image_path"
                                                    ][0]

                                    # Display issue list in grid
                                    with ui.grid(columns=3).classes("w-full gap-4"):
                                        for row in result:
                                            issue_id = row[0]
                                            date = row[1]
                                            title = row[2] or "Unknown"
                                            year_volume = row[3] or "N/A"
                                            line_count = row[4]
                                            block_count = row[5]
                                            page_count = row[6] or 0

                                            with (
                                                ui.card()
                                                .classes("hover:shadow-lg cursor-pointer")
                                                .props("flat bordered")
                                                .style(
                                                    "border: 1px solid #e0e0e0; border-radius: 4px; overflow: hidden;"
                                                )
                                            ):

                                                def make_click_handler(i, d):
                                                    return lambda: render_issue_view(i, d)

                                                with (
                                                    ui.row()
                                                    .classes("w-full")
                                                    .style("height: 200px;")
                                                    .on("click", make_click_handler(issue_id, date))
                                                ):
                                                    # Left column - issue info
                                                    with (
                                                        ui.column()
                                                        .classes("justify-center q-pa-md")
                                                        .style("flex: 1; height: 100%;")
                                                    ):
                                                        # Extract daily issue count from issue_id
                                                        # Format: {source}_{YYYY-MM-DD}_{issue:03d}_{daily}
                                                        issue_parts = issue_id.split("_")
                                                        daily_count = (
                                                            issue_parts[-1]
                                                            if len(issue_parts) >= 4
                                                            else ""
                                                        )

                                                        # Format date with weekday as title
                                                        if date:
                                                            weekday = date.strftime("%A")
                                                            date_str = date.strftime("%d.%m.%Y")
                                                            # Add daily count to weekday if available
                                                            weekday_label = (
                                                                f"{weekday} #{daily_count}"
                                                                if daily_count
                                                                else weekday
                                                            )
                                                            ui.label(weekday_label).classes(
                                                                "text-overline text-grey-6"
                                                            )
                                                            ui.label(f"{date_str}").classes(
                                                                "text-h6 text-grey-9"
                                                            ).style(
                                                                "line-height: 1.2; font-weight: 500;"
                                                            )
                                                        else:
                                                            ui.label("N/A").classes(
                                                                "text-h6 text-grey-9"
                                                            ).style(
                                                                "line-height: 1.2; font-weight: 500;"
                                                            )

                                                        ui.separator().classes("q-my-xs").style(
                                                            "width: 80%;"
                                                        )
                                                        with ui.column().classes("gap-1"):
                                                            # Use issue_num from issue_id (already split above)
                                                            issue_num = (
                                                                issue_parts[-2]
                                                                if len(issue_parts) >= 2
                                                                else "N/A"
                                                            )
                                                            ui.label(f"Issue: {issue_num}").classes(
                                                                "text-body2 text-grey-7"
                                                            )
                                                            with ui.row().classes("gap-2 q-mt-sm"):
                                                                ui.badge(
                                                                    f"{page_count} pages",
                                                                    color="grey-7",
                                                                ).props("outline")
                                                                ui.badge(
                                                                    f"{block_count} blocks",
                                                                    color="grey-7",
                                                                ).props("outline")

                                                    # Right column - image
                                                    with ui.element("div").style(
                                                        "flex: 1; background: #e0e0e0; position: relative; overflow: hidden; height: 100%;"
                                                    ):
                                                        img_path = issue_images.get(issue_id)
                                                        if img_path:
                                                            ui.image(
                                                                f"/static/{state.selected_source}/images/{img_path}"
                                                            ).style(
                                                                "width: 100%; height: 100%; object-fit: cover; object-position: center;"
                                                            )
                                                        else:
                                                            pass

                                    # Pagination controls
                                    if total_pages > 1:
                                        with ui.row().classes(
                                            "w-full justify-center items-center gap-2 q-mt-md"
                                        ):
                                            ui.button(
                                                icon="chevron_left",
                                                on_click=lambda: apply_filters_impl(
                                                    max(1, page - 1)
                                                ),
                                            ).props("flat round").set_enabled(page > 1)

                                            ui.label(f"Page {page} of {total_pages}").classes(
                                                "text-caption"
                                            )

                                            ui.button(
                                                icon="chevron_right",
                                                on_click=lambda: apply_filters_impl(
                                                    min(total_pages, page + 1)
                                                ),
                                            ).props("flat round").set_enabled(page < total_pages)

                                    # Show total count in sidebar
                                    with stats_container:
                                        ui.label("Statistics").classes("text-subtitle2 q-mb-xs")
                                        ui.label(f"Page {page} of {total_pages}").classes(
                                            "text-body2 text-grey-8"
                                        )
                                        ui.label(f"{total_count} total issues").classes(
                                            "text-caption text-grey-7"
                                        )

                            except Exception as e:
                                with ui.card().classes("w-full"):
                                    with ui.card_section():
                                        ui.label(f"Error loading browse data: {e}").classes(
                                            "text-red"
                                        )
                                        import traceback

                                        ui.label(traceback.format_exc()).classes("text-caption")

                    # Assign to module-level variable for access by navigation functions
                    apply_filters = apply_filters_impl

                    # Initial load
                    ui.timer(0.1, lambda: apply_filters(1), once=True)  # type: ignore

    def render_issue_view(issue_id: str, date):
        """Render a single issue view"""
        main_container.clear()

        # State for page navigation
        current_page_num = {"value": 1}

        with main_container:
            # Back button and header
            with ui.row().classes("w-full items-center gap-4 q-mb-md"):
                ui.button(icon="arrow_back", on_click=lambda: render_main_view()).props(
                    "flat round"
                )
                ui.label("Issue Reader").classes("text-h4")

            if not state.data_loader or not state.data_loader.parquet_exists():
                with ui.card().classes("w-full"):
                    with ui.card_section():
                        ui.label("No data available").classes("text-grey-7")
                return

            # Load issue data
            parquet_path = state.data_loader.parquet_path
            db = state.data_loader.db

            try:
                # Get issue metadata
                metadata = db.execute(
                    f"""
                    SELECT 
                        MIN(newspaper_title) as title,
                        MIN(date) as date,
                        MIN(year_volume) as year_volume,
                        MIN(issue_number) as issue_number,
                        COUNT(DISTINCT page_number) as page_count,
                        COUNT(DISTINCT text_block_id) as block_count,
                        COUNT(*) as line_count
                    FROM read_parquet('{parquet_path}')
                    WHERE issue_id = '{issue_id}'
                """
                ).fetchone()

                if not metadata:
                    with ui.card().classes("w-full"):
                        with ui.card_section():
                            ui.label("Issue not found").classes("text-grey-7")
                    return

                title = metadata[0] or "Unknown"
                date_obj = metadata[1]
                year_volume = metadata[2] or "N/A"
                issue_number = metadata[3] or "N/A"
                page_count = metadata[4] or 0
                block_count = metadata[5] or 0
                line_count = metadata[6] or 0

                # Compact issue metadata card (single card with all info)
                with (
                    ui.card()
                    .classes("w-full q-mb-md")
                    .style("border: 1px solid #e0e0e0; box-shadow: 0 1px 2px rgba(0,0,0,0.05);")
                ):
                    with ui.card_section().classes("q-pa-sm"):
                        # Title row
                        with ui.row().classes("w-full items-center gap-2 q-mb-xs"):
                            ui.icon("newspaper", size="1.5em").classes("text-grey-7")
                            ui.label(f"{title}").classes("text-h5 text-grey-9")

                        # Metadata in compact rows
                        with ui.row().classes("w-full gap-4 wrap items-center"):
                            with ui.row().classes("items-center gap-1"):
                                ui.icon("calendar_today", size="1em").classes("text-grey-6")
                                ui.label(
                                    f"{date_obj.strftime('%A, %B %d, %Y') if date_obj else 'N/A'}"
                                ).classes("text-body2 text-grey-8")

                            with ui.row().classes("items-center gap-1"):
                                ui.icon("library_books", size="1em").classes("text-grey-6")
                                # Clean up year_volume to avoid redundancy (e.g., "Jahrgang 1901" not "Vol. Jahrgang 1901")
                                if year_volume and "Jahrgang" in year_volume:
                                    volume_text = year_volume  # Already has "Jahrgang"
                                elif year_volume:
                                    volume_text = f"Vol. {year_volume}"
                                else:
                                    volume_text = "N/A"
                                ui.label(f"{volume_text}, Issue {issue_number}").classes(
                                    "text-body2 text-grey-8"
                                )

                            with ui.row().classes("items-center gap-1"):
                                ui.icon("description", size="1em").classes("text-grey-6")
                                ui.label(f"{page_count} pages").classes("text-body2 text-grey-8")

                            with ui.row().classes("items-center gap-1"):
                                ui.icon("view_module", size="1em").classes("text-grey-6")
                                ui.label(f"{block_count} text blocks").classes(
                                    "text-body2 text-grey-8"
                                )

                            with ui.row().classes("items-center gap-1"):
                                ui.icon("format_align_left", size="1em").classes("text-grey-6")
                                ui.label(f"{line_count:,} lines").classes("text-body2 text-grey-8")

                # Container for page content (will be updated on navigation)
                page_content_container = ui.column().classes("w-full")

                def render_page(page_num: int):
                    """Render a specific page of the issue"""
                    current_page_num["value"] = page_num
                    page_content_container.clear()

                    with page_content_container:
                        # Page navigation header - centered and cleaner
                        with (
                            ui.row()
                            .classes("w-full items-center justify-center q-mb-md q-pa-sm")
                            .style("background: #f5f5f5; border-radius: 4px;")
                        ):
                            ui.button(
                                icon="chevron_left",
                                on_click=lambda: render_page(max(1, page_num - 1)),
                            ).props("flat round").set_enabled(page_num > 1)

                            ui.label(f"Page {page_num} of {page_count}").classes(
                                "text-subtitle1 text-grey-9 q-mx-md"
                            )

                            ui.button(
                                icon="chevron_right",
                                on_click=lambda: render_page(min(page_count, page_num + 1)),
                            ).props("flat round").set_enabled(page_num < page_count)

                        # Get image for this page if available
                        img_path = None
                        if state.image_indexer:
                            try:
                                index = state.image_indexer.load_index()
                                if index is not None and len(index) > 0:
                                    # Filter by issue_id and page_number
                                    page_imgs = index.filter(
                                        (pl.col("issue_id") == issue_id)
                                        & (pl.col("page_number") == page_num)
                                    )
                                    if len(page_imgs) > 0:
                                        img_path = page_imgs["image_path"][0]
                            except Exception as e:
                                # Silently handle image loading errors
                                import traceback

                                traceback.print_exc()

                        # Get text blocks for this page
                        text_data = db.execute(
                            f"""
                            SELECT 
                                text_block_id,
                                STRING_AGG(text, ' ') as full_text,
                                COUNT(*) as line_count,
                                MIN(y) as min_y,
                                MIN(x) as min_x
                            FROM read_parquet('{parquet_path}')
                            WHERE issue_id = '{issue_id}' AND page_number = {page_num}
                            GROUP BY text_block_id
                            ORDER BY min_y, min_x
                        """
                        ).fetchall()

                        # Two-column layout: image on left, text on right
                        with ui.row().classes("w-full gap-4").style("align-items: flex-start;"):
                            # Left column - Image
                            with ui.column().classes("flex-1"):
                                if img_path:
                                    img_url = f"/static/{state.selected_source}/images/{img_path}"
                                    with (
                                        ui.card()
                                        .classes("w-full")
                                        .style(
                                            "box-shadow: none; border: 1px solid #e0e0e0; overflow: hidden;"
                                        )
                                    ):
                                        ui.image(img_url).style(
                                            "width: 100%; height: auto; display: block;"
                                        )
                                else:
                                    with (
                                        ui.card()
                                        .classes("w-full")
                                        .style(
                                            "box-shadow: none; border: 1px solid #e0e0e0; min-height: 600px;"
                                        )
                                    ):
                                        with ui.element("div").style(
                                            "height: 600px; display: flex; align-items: center; justify-content: center; background: #fafafa;"
                                        ):
                                            with ui.column().classes("items-center gap-2"):
                                                ui.icon("image_not_supported", size="48px").classes(
                                                    "text-grey-4"
                                                )
                                                ui.label("No image available").classes(
                                                    "text-body2 text-grey-6"
                                                )

                            # Right column - Text content
                            with ui.column().classes("flex-1"):
                                if text_data:
                                    ui.label(f"Text Blocks ({len(text_data)}):").classes(
                                        "text-subtitle2 q-mb-md"
                                    )

                                    for row in text_data:
                                        block_id = row[0]
                                        full_text = row[1] or ""
                                        lines = row[2] or 0

                                        with (
                                            ui.card()
                                            .classes("w-full q-mb-sm")
                                            .style("box-shadow: none; border: 1px solid #e0e0e0;")
                                        ):
                                            with ui.card_section().classes("q-pa-md"):
                                                with ui.row().classes(
                                                    "w-full items-start justify-between q-mb-sm"
                                                ):
                                                    ui.badge(
                                                        f"{lines} lines", color="grey-7"
                                                    ).props("outline")

                                                # Text content
                                                ui.label(full_text).classes("text-body2").style(
                                                    "white-space: pre-wrap; line-height: 1.6;"
                                                )
                                else:
                                    with (
                                        ui.card()
                                        .classes("w-full")
                                        .style("box-shadow: none; border: 1px solid #e0e0e0;")
                                    ):
                                        with ui.card_section():
                                            ui.label("No text content found for this page").classes(
                                                "text-grey-7"
                                            )

                # Initial page render
                render_page(current_page_num["value"])

            except Exception as e:
                with ui.card().classes("w-full"):
                    with ui.card_section():
                        ui.label(f"Error loading issue: {e}").classes("text-red")
                        import traceback

                        ui.label(traceback.format_exc()).classes("text-caption")

    def render_main_view():
        """Render the main browse list view"""
        main_container.clear()
        with main_container:
            ui.timer(0.1, lambda: render_browse_list(), once=True)

    # Main container for switching between views
    main_container = ui.column().classes("w-full")

    with main_container:
        render_browse_list()
