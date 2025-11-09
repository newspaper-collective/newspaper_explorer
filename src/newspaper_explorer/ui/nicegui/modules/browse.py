"""
Browse tab for the NiceGUI interface - Browse newspaper collection chronologically
"""

from nicegui import ui
from pathlib import Path
from datetime import datetime


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

    def navigate_to_year(year):
        """Navigate to a specific year or back to all years"""
        current_filters["year"] = year
        current_filters["month"] = None
        current_page["value"] = 1
        render_main_view()

    def navigate_to_month(year, month):
        """Navigate to a specific month (or all months in a year)"""
        current_filters["year"] = year
        current_filters["month"] = month
        current_page["value"] = 1
        render_main_view()

    def show_issues_for_month(year, month):
        """Show individual issues for a specific month"""
        current_filters["year"] = year
        current_filters["month"] = month
        current_page["value"] = 1
        render_main_view()

    def render_browse_list():
        """Render the main browse list view"""
        with ui.column().classes("w-full"):
            # Get source title for display
            source_title = "Collection"
            if state.source_config and state.source_config.metadata:
                source_title = state.source_config.metadata.newspaper_title

            # Breadcrumb navigation
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
                        ui.button(
                            f"{current_filters['year']}",
                            on_click=lambda: navigate_to_month(current_filters["year"], None),
                        ).props("flat").classes("text-h6")

                    if current_filters["month"] is not None:
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
                        ui.label("›").classes("text-h5 text-grey-5")
                        ui.label(
                            f"{month_names[current_filters['month'] - 1]} {current_filters['year']}"
                        ).classes("text-h5")

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
                with ui.column().classes("w-64 flex-shrink-0"):
                    with ui.card().classes("w-full"):
                        with ui.card_section():
                            ui.label("Browse Filters").classes("text-h6 q-mb-md")

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

                            # Always create year_from and year_to for the apply_filters function
                            year_from = None
                            year_to = None

                            if current_filters["year"] is not None:
                                # Show selected year as read-only when drilling down
                                with ui.row().classes("w-full items-center gap-2 q-mb-md"):
                                    ui.input(
                                        label="Year", value=str(current_filters["year"])
                                    ).props("readonly outlined dense").classes("col")
                            else:
                                # Show year range selector
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

                            ui.separator()

                            # Group by options - only show when not drilling down
                            if current_filters["year"] is None:
                                ui.label("Group By").classes("text-subtitle2 q-mb-xs q-mt-md")
                                group_by = ui.select(
                                    label="Grouping",
                                    options=["Year", "Month", "Issue"],
                                    value="Year",
                                ).classes("w-full q-mb-md")
                                group_by_value = group_by.value
                            else:
                                # When drilling down, group_by is not shown
                                group_by = None
                                group_by_value = (
                                    "Year"  # Doesn't matter, will be overridden by show_view
                                )

                            ui.separator()

                            # Sort order
                            ui.label("Sort Order").classes("text-subtitle2 q-mb-xs q-mt-md")
                            sort_order = ui.select(
                                label="Order",
                                options=["Newest First", "Oldest First"],
                                value="Oldest First",
                            ).classes("w-full q-mb-md")

                            # Apply button - only show when not drilling down
                            apply_btn = None
                            if current_filters["year"] is None:
                                apply_btn = (
                                    ui.button("Apply Filters", icon="filter_list")
                                    .props("color=primary")
                                    .classes("w-full q-mt-md")
                                )

                    # Stats container in sidebar (will be updated by apply_filters)
                    with ui.card().classes("w-full q-mt-sm"):
                        with ui.card_section().classes("q-pa-sm"):
                            stats_container = ui.column().classes("w-full gap-1")

                # Right content area - Browse results
                with ui.column().classes("flex-1"):
                    # Results container
                    result_container = ui.column().classes("w-full gap-4")

                    async def apply_filters(page=1):
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
                                    # Show whatever the user selected
                                    show_view = group_by.value if group_by is not None else "Year"

                                if show_view == "Year":
                                    # Query for yearly grouping
                                    sort_dir = (
                                        "DESC" if sort_order.value == "Newest First" else "ASC"
                                    )

                                    # Get year range from filters or UI
                                    query_year_from = filter_year_from
                                    query_year_to = filter_year_to
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
                                        columns="repeat(auto-fill, minmax(200px, 1fr))"
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
                                                    "border: 1px solid #e0e0e0; border-radius: 4px;"
                                                )
                                            ):
                                                with (
                                                    ui.card_section()
                                                    .classes("q-pa-sm")
                                                    .on(
                                                        "click",
                                                        lambda y=year: navigate_to_month(y, None),
                                                    )
                                                ):
                                                    with ui.column().classes(
                                                        "gap-1 items-center text-center w-full"
                                                    ):
                                                        ui.label(f"{year}").classes(
                                                            "text-h4 text-grey-9"
                                                        ).style("line-height: 1;")
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
                                    # Query for monthly grouping with pagination
                                    # If drill-down year is set, only show months for that year
                                    sort_dir = (
                                        "DESC" if sort_order.value == "Newest First" else "ASC"
                                    )

                                    # Build WHERE clause based on filters
                                    if current_filters["year"] is not None:
                                        year_filter_min = current_filters["year"]
                                        year_filter_max = current_filters["year"]
                                    elif year_from is not None and year_to is not None:
                                        year_filter_min = int(year_from.value)
                                        year_filter_max = int(year_to.value)
                                    else:
                                        year_filter_min = filter_year_from
                                        year_filter_max = filter_year_to

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
                                                    with ui.column().classes(
                                                        "gap-1 items-center text-center w-full"
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
                                                on_click=lambda: apply_filters(max(1, page - 1)),
                                            ).props("flat round").set_enabled(page > 1)

                                            ui.label(f"Page {page} of {total_pages}").classes(
                                                "text-caption"
                                            )

                                            ui.button(
                                                icon="chevron_right",
                                                on_click=lambda: apply_filters(
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
                                    # Query for individual issues with pagination
                                    # Filter by year and optionally month if drilling down
                                    sort_dir = (
                                        "DESC" if sort_order.value == "Newest First" else "ASC"
                                    )

                                    # Build WHERE clause based on drill-down filters
                                    if current_filters["year"] is not None:
                                        year_filter_min = current_filters["year"]
                                        year_filter_max = current_filters["year"]
                                    elif year_from is not None and year_to is not None:
                                        year_filter_min = int(year_from.value)
                                        year_filter_max = int(year_to.value)
                                    else:
                                        year_filter_min = filter_year_from
                                        year_filter_max = filter_year_to

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
                                        ORDER BY MIN(date) {sort_dir}
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

                                    # Display issue list
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
                                            .classes("w-full hover:bg-grey-1")
                                            .props("flat bordered")
                                            .style("border: 1px solid #e0e0e0; border-radius: 4px;")
                                        ):

                                            def make_click_handler(i, d):
                                                return lambda: render_issue_view(i, d)

                                            with (
                                                ui.card_section()
                                                .classes("cursor-pointer")
                                                .on("click", make_click_handler(issue_id, date))
                                            ):
                                                with ui.row().classes(
                                                    "w-full items-center justify-between"
                                                ):
                                                    with ui.column().classes("gap-1"):
                                                        ui.label(f"📰 {title}").classes(
                                                            "text-h6 text-grey-9"
                                                        )
                                                        ui.label(
                                                            f"{date.strftime('%Y-%m-%d') if date else 'N/A'} • {year_volume}"
                                                        ).classes("text-caption text-grey-6")
                                                    with ui.row().classes("gap-2 items-center"):
                                                        ui.badge(
                                                            f"{page_count} pages", color="grey-7"
                                                        ).props("outline")
                                                        ui.badge(
                                                            f"{block_count} blocks", color="grey-7"
                                                        ).props("outline")
                                                        ui.icon("chevron_right").classes(
                                                            "text-grey-5"
                                                        )

                                    # Pagination controls
                                    if total_pages > 1:
                                        with ui.row().classes(
                                            "w-full justify-center items-center gap-2 q-mt-md"
                                        ):
                                            ui.button(
                                                icon="chevron_left",
                                                on_click=lambda: apply_filters(max(1, page - 1)),
                                            ).props("flat round").set_enabled(page > 1)

                                            ui.label(f"Page {page} of {total_pages}").classes(
                                                "text-caption"
                                            )

                                            ui.button(
                                                icon="chevron_right",
                                                on_click=lambda: apply_filters(
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

                    # Connect apply button (only if it exists)
                    if apply_btn is not None:
                        apply_btn.on("click", lambda: apply_filters(1))

                    # Initial load
                    ui.timer(0.1, lambda: apply_filters(1), once=True)

    def render_issue_view(issue_id: str, date):
        """Render a single issue view"""
        main_container.clear()

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

                # Issue metadata card
                with ui.card().classes("w-full q-mb-md"):
                    with ui.card_section():
                        ui.label(f"📰 {title}").classes("text-h5 q-mb-md")
                        with ui.row().classes("gap-4 wrap"):
                            ui.label(
                                f"📅 Date: {date_obj.strftime('%Y-%m-%d') if date_obj else 'N/A'}"
                            ).classes("text-body2")
                            ui.label(f"📖 Volume: {year_volume}").classes("text-body2")
                            ui.label(f"🔢 Issue: {issue_number}").classes("text-body2")
                            ui.label(f"📄 Pages: {page_count}").classes("text-body2")
                            ui.label(f"📦 Text blocks: {block_count}").classes("text-body2")

                # Get text blocks for this issue
                text_data = db.execute(
                    f"""
                    SELECT 
                        text_block_id,
                        page_number,
                        STRING_AGG(text, ' ') as full_text,
                        COUNT(*) as line_count
                    FROM read_parquet('{parquet_path}')
                    WHERE issue_id = '{issue_id}'
                    GROUP BY text_block_id, page_number
                    ORDER BY page_number, text_block_id
                    LIMIT 100
                """
                ).fetchall()

                if text_data:
                    ui.label(f"Content ({len(text_data)} text blocks):").classes("text-h6 q-mb-md")

                    for row in text_data:
                        block_id = row[0]
                        page_num = row[1] or "N/A"
                        full_text = row[2] or ""
                        lines = row[3] or 0

                        with ui.card().classes("w-full q-mb-sm").props("flat bordered"):
                            with ui.card_section():
                                with ui.row().classes("w-full items-start justify-between q-mb-sm"):
                                    ui.label(f"Page {page_num}").classes("text-caption text-grey-7")
                                    ui.badge(f"{lines} lines", color="grey").props("outline")

                                # Text content
                                ui.label(
                                    full_text[:500] + ("..." if len(full_text) > 500 else "")
                                ).classes("text-body2").style("white-space: pre-wrap;")
                else:
                    with ui.card().classes("w-full"):
                        with ui.card_section():
                            ui.label("No text content found for this issue").classes("text-grey-7")

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
