import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.colors as pc
from datetime import datetime
import json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import io
import sys

# Optional wordcloud imports - handled gracefully if package is missing
try:
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
except Exception:
    WordCloud = None
    plt = None

# Page config
st.set_page_config(
    page_title="Historical Newspaper Explorer", page_icon="📰", layout="wide"
)

# Initialize session state
if "layout_data" not in st.session_state:
    st.session_state.layout_data = {}
if "image_data" not in st.session_state:
    st.session_state.image_data = {}
if "selected_image" not in st.session_state:
    st.session_state.selected_image = None
if "current_page" not in st.session_state:
    st.session_state.current_page = 1


# Helper functions
def create_placeholder_image(
    width=300,
    height=300,
    text="No Image",
    bg_color=(204, 204, 204),
    text_color=(102, 102, 102),
):
    """Create a placeholder image with text"""
    img = Image.new("RGB", (width, height), color=bg_color)
    draw = ImageDraw.Draw(img)

    # Calculate text position (centered)
    bbox = draw.textbbox((0, 0), text)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    position = ((width - text_width) // 2, (height - text_height) // 2)

    draw.text(position, text, fill=text_color)
    return img


def parse_date_from_filename(filename):
    """Extract YYYY-MM-DD from filename like 3074409X_1912-02-11_000_35_H_2_005"""
    try:
        parts = filename.split("_")
        if len(parts) >= 2:
            date_str = parts[1]  # Expected format: YYYY-MM-DD
            return pd.to_datetime(date_str)
    except:
        pass
    return None


@st.cache_data
def load_image_pairs_batch(base_dir):
    """Load all image-caption pair JSONs from directory"""
    base_path = Path(base_dir)
    all_pairs = []

    # Find all JSON files in picture_caption_pairs
    json_files = list(base_path.glob("**/*.json"))

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Parse date from fulltext_basename
            parsed_date = parse_date_from_filename(data.get("fulltext_basename", ""))

            # Add metadata
            data["parsed_date"] = parsed_date
            data["json_path"] = str(json_file)
            all_pairs.append(data)
        except Exception as e:
            continue

    return sorted(
        all_pairs, key=lambda x: x["parsed_date"] if x["parsed_date"] else datetime.min
    )


@st.cache_data
def load_layout_json(json_path):
    """Load layout segmentation JSON"""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def filter_pairs_by_date(pairs, start_date, end_date):
    """Filter image pairs by date range"""
    filtered = []
    for pair in pairs:
        if not pair or not isinstance(pair, dict):
            continue
        parsed_date = pair.get("parsed_date")
        if parsed_date:
            if start_date <= parsed_date <= end_date:
                filtered.append(pair)
    return filtered


def draw_bboxes_on_image(img_path, detections, selected_classes=None):
    """Draw bounding boxes on image"""
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img, "RGBA")

    colors = {
        "Text": (0, 0, 255, 80),
        "Picture": (0, 255, 0, 80),
        "Caption": (255, 165, 0, 80),
        "Section-header": (255, 0, 255, 80),
    }

    for det in detections:
        if selected_classes and det["class"] not in selected_classes:
            continue

        bbox = det["bbox"]
        color = colors.get(det["class"], (128, 128, 128, 80))

        # Draw filled rectangle
        draw.rectangle(
            [(bbox["x1"], bbox["y1"]), (bbox["x2"], bbox["y2"])],
            outline=tuple(c if i < 3 else 255 for i, c in enumerate(color)),
            fill=color,
            width=3,
        )

        # Draw label
        label = f"{det['class']} ({det['confidence']:.2f})"
        draw.text((bbox["x1"], bbox["y1"] - 15), label, fill="red")

    return img


# Sidebar
st.sidebar.title("📰 Newspaper Archive")
st.sidebar.markdown("---")

# Data loading
st.sidebar.subheader("Data Sources")
data_dir = st.sidebar.text_input("Data directory", value="./data")
neo4j_uri = st.sidebar.text_input("Neo4j URI", value="bolt://localhost:7687")
neo4j_user = st.sidebar.text_input("Neo4j User", value="neo4j")
neo4j_pass = st.sidebar.text_input("Neo4j Password", type="password")

# Filters
st.sidebar.markdown("---")
st.sidebar.subheader("Filters")

# Date range
col1, col2 = st.sidebar.columns(2)
start_date = col1.date_input("Start", value=datetime(1901, 1, 1))
end_date = col2.date_input("End", value=datetime(1920, 12, 31))

# Emotion filter
emotions = ["all", "sadness", "joy", "anger", "fear", "surprise", "neutral"]
selected_emotion = st.sidebar.selectbox("Emotion", emotions)

# Entity type filter
entity_types = st.sidebar.multiselect(
    "Entity Types",
    ["PERSON", "ORG", "LOCATION", "EVENT"],
    default=["PERSON", "ORG", "LOCATION", "EVENT"],
)

# Entity search
entity_search = st.sidebar.text_input("Search entities")

# Stats
st.sidebar.markdown("---")
st.sidebar.subheader("Summary")
st.sidebar.metric("Total Pages", "1,247")
st.sidebar.metric("Entities Found", "3,456")
st.sidebar.metric("Date Range", "1901-1905")

# Main content
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Entities", "Knowledge Graph", "Images", "Search", "Topics", "Emotions"]
)

# Tab 1: Timeline
with tab1:
    # st.title("Temporal Analysis")

    # Load entities CSV (if present)
    @st.cache_data
    def load_entities_csv(path: str):
        return pd.read_csv(path)

    # Try the data directory first, then fall back to repo root filename
    csv_path = Path(data_dir) / "test.csv"
    if not csv_path.exists():
        csv_path = Path("test.csv")

    df_entities = None
    try:
        df_entities = load_entities_csv(str(csv_path))
        st.success(f"✅ Loaded entities data: {len(df_entities)} records")
    except Exception as e:
        st.warning(f"Could not load entities CSV at {csv_path}: {e}")
        st.info("Using mock data for demonstration")

    # Initialize date_filtered
    date_filtered = pd.DataFrame()

    # If we have real entity data, use it; otherwise use mock data
    if (
        df_entities is not None
        and not df_entities.empty
        and "Date" in df_entities.columns
    ):
        # Convert date column to datetime if needed
        df_entities["date"] = pd.to_datetime(df_entities["Date"], errors="coerce")
        df_entities["type"] = df_entities["Type"]

        # Filter by date range
        date_filtered = df_entities[
            (df_entities["date"] >= pd.Timestamp(start_date))
            & (df_entities["date"] <= pd.Timestamp(end_date))
        ]

        # Entity frequency over time
        st.subheader("Entity Mentions Over Time")

        # Group by date and entity type
        entity_counts = (
            date_filtered.groupby(["date", "type"]).size().reset_index(name="Count")
        )

        if not entity_counts.empty:
            fig2 = px.bar(
                entity_counts,
                x="date",
                y="Count",
                color="type",
                title="Entity Type Distribution Over Time",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No entity data in selected date range")

    else:
        # Fallback to mock data
        st.subheader("Entity Mentions Over Time (Mock Data)")
        dates = pd.date_range(start="1901-01-01", end="1901-12-31", freq="M")
        entity_data = pd.DataFrame(
            {
                "Date": dates,
                "PERSON": np.random.randint(10, 100, len(dates)),
                "ORG": np.random.randint(5, 80, len(dates)),
                "LOCATION": np.random.randint(10, 90, len(dates)),
                "EVENT": np.random.randint(1, 50, len(dates)),
            }
        )

        fig2 = px.bar(
            entity_data.melt(
                id_vars="Date", var_name="Entity Type", value_name="Count"
            ),
            x="Date",
            y="Count",
            color="Entity Type",
            title="Entity Type Distribution Over Time",
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Entity analysis
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Entity Type Distribution")
        if (
            df_entities is not None
            and not df_entities.empty
            and "type" in df_entities.columns
        ):
            type_counts = date_filtered["type"].value_counts()
            fig3 = px.pie(
                values=type_counts.values,
                names=type_counts.index,
                title="Distribution of Entity Types",
            )
            st.plotly_chart(fig3, use_container_width=True)
        else:
            # Mock pie chart
            st.info("No entity data available")

    with col2:
        st.subheader("Top Entities This Period")
        if (
            df_entities is not None
            and not df_entities.empty
            and "Entity" in df_entities.columns
        ):
            top_entities = (
                date_filtered.groupby(["Entity", "type"])
                .size()
                .reset_index(name="Count")
            )
            top_entities = top_entities.sort_values("Count", ascending=False).head(10)
            st.dataframe(top_entities, use_container_width=True, hide_index=True)
        else:
            # Mock data
            top_entities = pd.DataFrame(
                {
                    "Entity": [
                        "Kaiser Wilhelm II",
                        "Berlin",
                        "Reichstag",
                        "Deutschland",
                        "Frankreich",
                        "London",
                        "Paris",
                    ],
                    "Count": [145, 132, 98, 156, 89, 67, 54],
                    "Type": [
                        "PERSON",
                        "LOCATION",
                        "ORG",
                        "LOCATION",
                        "LOCATION",
                        "LOCATION",
                        "LOCATION",
                    ],
                }
            )
            st.dataframe(top_entities, use_container_width=True, hide_index=True)

# Tab 2: Network Graph
with tab2:
    # st.title("Entity Relationship Network")

    # Embed the interactive graph
    st.components.v1.iframe(
        "https://dertag.proto-khora.pages.dev/", height=800, scrolling=True
    )

# Tab 3: Images
with tab3:
    st.title("Image Explorer")

    # Load image pairs data
    layout_base = Path("../../layout/enriched/picture_caption_pairs")
    data_base = Path("../../data")

    # Load enrichment summary for statistics
    @st.cache_data
    def load_enrichment_summary():
        summary_path = Path("../../layout/enriched/enrichment_summary.json")
        if summary_path.exists():
            with open(summary_path, "r") as f:
                return json.load(f)
        return None

    enrichment_summary = load_enrichment_summary()

    # Display statistics at the top
    if enrichment_summary:
        st.markdown("### 📊 Collection Statistics")
        col1, col2, col3, col4 = st.columns(4)

        pic_cap_data = enrichment_summary.get("picture_caption_pairs", {})
        total_pics = pic_cap_data.get("total_pictures", 0)
        with_caps = pic_cap_data.get("pictures_with_captions", 0)
        without_caps = pic_cap_data.get("pictures_without_captions", 0)

        with col1:
            st.metric("Total Pictures", f"{total_pics:,}")
        with col2:
            st.metric(
                "With Captions",
                f"{with_caps:,}",
                delta=f"{(with_caps/total_pics*100):.1f}%" if total_pics > 0 else "0%",
            )
        with col3:
            st.metric(
                "Without Captions",
                f"{without_caps:,}",
                delta=(
                    f"{(without_caps/total_pics*100):.1f}%" if total_pics > 0 else "0%"
                ),
            )
        with col4:
            detections_with_text = enrichment_summary.get("detections_with_text", 0)
            total_detections = enrichment_summary.get("total_detections", 0)
            st.metric(
                "Text Enriched",
                (
                    f"{(detections_with_text/total_detections*100):.1f}%"
                    if total_detections > 0
                    else "0%"
                ),
            )

        # Visualizations
        col1, col2 = st.columns([1, 1])

        with col1:
            # Pie chart for caption coverage
            if with_caps > 0 or without_caps > 0:
                fig_pie = go.Figure(
                    data=[
                        go.Pie(
                            labels=["With Captions", "Without Captions"],
                            values=[with_caps, without_caps],
                            hole=0.4,
                            marker_colors=["#00cc96", "#ef553b"],
                        )
                    ]
                )
                fig_pie.update_layout(
                    title="Picture-Caption Pair Coverage",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=20),
                )
                st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar chart for class match rates
            class_match = enrichment_summary.get("class_match_rates", {})
            if class_match:
                classes = []
                match_percentages = []
                for cls, data in class_match.items():
                    matched = data.get("matched", 0)
                    total = data.get("total", 1)
                    if total > 0:
                        classes.append(cls)
                        match_percentages.append((matched / total) * 100)

                fig_bar = go.Figure(
                    data=[
                        go.Bar(
                            x=classes,
                            y=match_percentages,
                            marker_color="#636efa",
                            text=[f"{p:.1f}%" for p in match_percentages],
                            textposition="outside",
                        )
                    ]
                )
                fig_bar.update_layout(
                    title="Text Extraction Success Rate by Class",
                    yaxis_title="Success Rate (%)",
                    xaxis_title="Detection Class",
                    height=300,
                    margin=dict(l=20, r=20, t=40, b=60),
                    yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown("---")

    if st.session_state.selected_image is None:
        # Grid view with filters
        st.subheader("Picture-Caption Gallery")

        # Filter options
        col1, col2, col3 = st.columns(3)
        with col1:
            caption_filter = st.selectbox(
                "Caption Filter", ["All", "Only with captions", "Only without captions"]
            )
        with col2:
            min_confidence = st.slider(
                "Min Picture Confidence",
                min_value=0.0,
                max_value=1.0,
                value=0.0,
                step=0.05,
            )
        with col3:
            sort_option = st.selectbox(
                "Sort by",
                [
                    "Date (newest)",
                    "Date (oldest)",
                    "Confidence (high)",
                    "Confidence (low)",
                ],
            )

        # Load and filter data
        with st.spinner("Loading image pairs..."):
            all_pairs = load_image_pairs_batch(layout_base)

        if all_pairs:
            # Apply date filter
            filtered_pairs = filter_pairs_by_date(
                all_pairs, pd.Timestamp(start_date), pd.Timestamp(end_date)
            )

            # Apply caption filter
            if caption_filter == "Only with captions":
                filtered_pairs = [
                    p
                    for p in filtered_pairs
                    if p.get("caption", {}).get("text_content")
                ]
            elif caption_filter == "Only without captions":
                filtered_pairs = [
                    p
                    for p in filtered_pairs
                    if not p.get("caption", {}).get("text_content")
                ]

            # Apply confidence filter
            filtered_pairs = [
                p
                for p in filtered_pairs
                if p.get("picture", {}).get("confidence", 0) >= min_confidence
            ]

            # Apply sorting
            if sort_option == "Date (newest)":
                filtered_pairs = sorted(
                    filtered_pairs,
                    key=lambda x: x["parsed_date"] or datetime.min,
                    reverse=True,
                )
            elif sort_option == "Date (oldest)":
                filtered_pairs = sorted(
                    filtered_pairs, key=lambda x: x["parsed_date"] or datetime.min
                )
            elif sort_option == "Confidence (high)":
                filtered_pairs = sorted(
                    filtered_pairs,
                    key=lambda x: x.get("picture", {}).get("confidence", 0),
                    reverse=True,
                )
            elif sort_option == "Confidence (low)":
                filtered_pairs = sorted(
                    filtered_pairs,
                    key=lambda x: x.get("picture", {}).get("confidence", 0),
                )

            st.info(f"Found {len(filtered_pairs)} image pairs matching filters")

            # Pagination controls
            items_per_page = 12
            total_pages = max(1, (len(filtered_pairs) - 1) // items_per_page + 1)

            col1, col2, col3 = st.columns([1, 3, 1])
            with col2:
                page = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=st.session_state.current_page,
                    key="page_selector",
                )
                st.session_state.current_page = page

            # Calculate slice
            start_idx = (page - 1) * items_per_page
            end_idx = min(start_idx + items_per_page, len(filtered_pairs))
            page_pairs = filtered_pairs[start_idx:end_idx]

            # Display grid (4 columns)
            cols = st.columns(4)
            for i, pair in enumerate(page_pairs):
                if not pair or not isinstance(pair, dict):
                    continue

                with cols[i % 4]:
                    # Try to load crop image, fallback to source image
                    picture = pair.get("picture") or {}
                    crop_path = data_base / picture.get("crop_file", "")

                    image_loaded = False

                    # Try crop first
                    if crop_path.exists():
                        try:
                            img = Image.open(crop_path)
                            st.image(img, use_container_width=True)
                            image_loaded = True
                        except:
                            pass

                    # Fallback to source image
                    if not image_loaded:
                        source_img_path = data_base / pair.get("source_image", "")
                        if source_img_path.exists():
                            try:
                                img = Image.open(source_img_path)
                                # Crop to picture bbox if available
                                bbox = picture.get("bbox", {})
                                if bbox:
                                    x1 = int(bbox.get("x1", 0))
                                    y1 = int(bbox.get("y1", 0))
                                    x2 = int(bbox.get("x2", img.width))
                                    y2 = int(bbox.get("y2", img.height))
                                    img = img.crop((x1, y1, x2, y2))
                                st.image(img, use_container_width=True)
                                image_loaded = True
                            except Exception as e:
                                pass

                    # Final fallback to placeholder
                    if not image_loaded:
                        placeholder = create_placeholder_image(
                            300, 300, "Image Not Found"
                        )
                        st.image(placeholder, use_container_width=True)

                    # Display metadata
                    parsed_date = pair.get("parsed_date")
                    date_str = (
                        parsed_date.strftime("%Y-%m-%d") if parsed_date else "Unknown"
                    )
                    st.caption(f"📅 {date_str}")

                    # Caption preview (first 100 chars)
                    caption = pair.get("caption") or {}
                    caption_text = caption.get("text_content", None)
                    if caption_text:
                        if len(caption_text) > 80:
                            preview = caption_text[:80] + "..."
                        else:
                            preview = caption_text
                        st.caption(f"💬 {preview}")
                    else:
                        st.caption("💬 ⚠️ _No caption text_")

                    # Confidence and bbox size
                    conf = picture.get("confidence", 0)
                    bbox = picture.get("bbox", {})
                    size_info = (
                        f"{bbox.get('width', 0):.0f}×{bbox.get('height', 0):.0f}"
                    )
                    st.caption(f"🎯 {conf:.2f} | 📏 {size_info}")

                    # View details button
                    if st.button("View Details", key=f"view_{i}_{page}"):
                        st.session_state.selected_image = pair
                        st.rerun()
        else:
            st.warning("No image pairs found. Please check the data directory.")

    else:
        # Detail view
        pair = st.session_state.selected_image

        if not pair or not isinstance(pair, dict):
            st.error("Invalid image data. Returning to gallery.")
            st.session_state.selected_image = None
            st.rerun()

        if st.button("⬅️ Back to Gallery"):
            st.session_state.selected_image = None
            st.rerun()

        st.markdown("---")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("Picture & Caption")

            # Display picture crop
            picture = pair.get("picture") or {}
            crop_path = data_base / picture.get("crop_file", "")

            picture_loaded = False
            if crop_path.exists():
                try:
                    img = Image.open(crop_path)
                    st.image(img, caption="Cropped Picture")
                    picture_loaded = True
                except Exception as e:
                    pass

            # Fallback to source image with bbox crop
            if not picture_loaded:
                source_img_path = data_base / pair.get("source_image", "")
                if source_img_path.exists():
                    try:
                        img = Image.open(source_img_path)
                        bbox = picture.get("bbox", {})
                        if bbox:
                            x1 = int(bbox.get("x1", 0))
                            y1 = int(bbox.get("y1", 0))
                            x2 = int(bbox.get("x2", img.width))
                            y2 = int(bbox.get("y2", img.height))
                            img = img.crop((x1, y1, x2, y2))
                        st.image(img, caption="Picture (from source)")
                        picture_loaded = True
                    except Exception as e:
                        st.error(f"Could not load image: {e}")

            if not picture_loaded:
                st.warning("Picture image not available")

            # Display caption crop if available
            caption = pair.get("caption") or {}
            caption_crop = caption.get("crop_file", "")

            caption_loaded = False
            if caption_crop:
                caption_crop_path = data_base / caption_crop
                if caption_crop_path.exists():
                    try:
                        caption_img = Image.open(caption_crop_path)
                        st.image(caption_img, caption="Caption Region")
                        caption_loaded = True
                    except:
                        pass

            # Fallback to source image with caption bbox crop
            if not caption_loaded and caption:
                source_img_path = data_base / pair.get("source_image", "")
                if source_img_path.exists():
                    try:
                        img = Image.open(source_img_path)
                        cap_bbox = caption.get("bbox", {})
                        if cap_bbox:
                            x1 = int(cap_bbox.get("x1", 0))
                            y1 = int(cap_bbox.get("y1", 0))
                            x2 = int(cap_bbox.get("x2", img.width))
                            y2 = int(cap_bbox.get("y2", img.height))
                            img = img.crop((x1, y1, x2, y2))
                            st.image(img, caption="Caption Region (from source)")
                            caption_loaded = True
                    except:
                        pass

        with col2:
            st.subheader("Metadata")

            # Date
            parsed_date = pair.get("parsed_date")
            date_str = parsed_date.strftime("%Y-%m-%d") if parsed_date else "Unknown"
            st.metric("Date", date_str)

            # Basename
            st.text(f"File: {pair.get('fulltext_basename', 'N/A')}")

            # Source paths
            with st.expander("📂 Source Files"):
                st.code(pair.get("source_image", "N/A"), language="text")
                st.code(pair.get("source_fulltext", "N/A"), language="text")

            st.markdown("**Picture Detection**")
            pic = picture  # Already defined above
            st.write(f"- Confidence: {pic.get('confidence', 0):.3f}")
            bbox = pic.get("bbox", {})
            st.write(
                f"- BBox: ({bbox.get('x1', 0):.0f}, {bbox.get('y1', 0):.0f}) → "
                f"({bbox.get('x2', 0):.0f}, {bbox.get('y2', 0):.0f})"
            )
            st.write(
                f"- Size: {bbox.get('width', 0):.0f} × {bbox.get('height', 0):.0f} px"
            )
            area = bbox.get("width", 0) * bbox.get("height", 0)
            st.write(f"- Area: {area:,.0f} px²")

            st.markdown("**Caption Detection**")
            cap = caption  # Already defined above
            if cap:
                st.write(f"- Confidence: {cap.get('confidence', 0):.3f}")
                cap_bbox = cap.get("bbox", {})
                if cap_bbox:
                    st.write(
                        f"- Size: {cap_bbox.get('width', 0):.0f} × "
                        f"{cap_bbox.get('height', 0):.0f} px"
                    )
            else:
                st.write("- No caption detected")

            # Full caption text
            caption_text = cap.get("text_content", None) if cap else None
            st.markdown("**Caption Text:**")
            if caption_text:
                st.info(caption_text)
                st.caption(f"Character count: {len(caption_text)}")
            else:
                st.warning("No text extracted")

            # Spatial relationship
            spatial = pair.get("spatial_relationship", {})
            if spatial:
                st.markdown("**Spatial Relationship:**")
                st.write(f"- Euclidean distance: {spatial.get('distance', 0):.1f} px")
                vertical_dist = spatial.get("vertical_distance", 0)
                st.write(f"- Vertical distance: {vertical_dist:.1f} px")
                if vertical_dist < 0:
                    st.caption("↑ Caption is above picture")
                else:
                    st.caption("↓ Caption is below picture")

            # Download options
            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            with col_dl1:
                st.download_button(
                    "📥 JSON",
                    data=json.dumps(pair, indent=2, default=str),
                    file_name=f"{pair.get('fulltext_basename', 'pair')}.json",
                    mime="application/json",
                    use_container_width=True,
                )
            with col_dl2:
                if caption_text:
                    st.download_button(
                        "📝 Text",
                        data=caption_text,
                        file_name=f"{pair.get('fulltext_basename', 'caption')}.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )

# Tab 4: Search
with tab4:
    st.title("Caption Search")

    # Search box
    search_query = st.text_input("🔍 Search", placeholder="Search in image captions...")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.caption("Searches through extracted caption text from picture-caption pairs")

    with col2:
        sort_by = st.selectbox(
            "Sort by:",
            ["Date (newest)", "Date (oldest)", "Relevance"],
        )

    if search_query:
        st.markdown("---")

        # Load data and search
        layout_base = Path("../../layout/enriched/picture_caption_pairs")
        with st.spinner("Searching..."):
            all_pairs = load_image_pairs_batch(layout_base)

        # Filter by date range
        date_filtered = filter_pairs_by_date(
            all_pairs, pd.Timestamp(start_date), pd.Timestamp(end_date)
        )

        # Search in captions (case-insensitive)
        search_lower = search_query.lower()
        results = []
        for pair in date_filtered:
            if not pair or not isinstance(pair, dict):
                continue
            caption = pair.get("caption") or {}
            caption_text = caption.get("text_content", "")
            if caption_text and search_lower in caption_text.lower():
                results.append(pair)

        # Sort results
        if sort_by == "Date (newest)":
            results = sorted(
                results,
                key=lambda x: x.get("parsed_date") or datetime.min,
                reverse=True,
            )
        elif sort_by == "Date (oldest)":
            results = sorted(
                results, key=lambda x: x.get("parsed_date") or datetime.min
            )

        st.subheader(f"Found {len(results)} results for: '{search_query}'")

        if results:
            for i, pair in enumerate(results[:50]):  # Limit to first 50 results
                if not pair or not isinstance(pair, dict):
                    continue

                with st.container():
                    col1, col2 = st.columns([4, 1])

                    with col1:
                        parsed_date = pair.get("parsed_date")
                        date_str = (
                            parsed_date.strftime("%Y-%m-%d")
                            if parsed_date
                            else "Unknown"
                        )
                        st.markdown(f"### 📅 {date_str}")

                        caption = pair.get("caption") or {}
                        caption_text = caption.get("text_content", "No caption")

                        # Highlight search term
                        if caption_text:
                            # Simple highlighting
                            highlighted = caption_text.replace(
                                search_query, f"**{search_query}**"
                            )
                            st.markdown(highlighted)

                        st.caption(f"Basename: {pair.get('fulltext_basename', 'N/A')}")

                    with col2:
                        picture = pair.get("picture") or {}
                        conf = picture.get("confidence", 0)
                        st.metric("Conf.", f"{conf:.2f}")

                        if st.button("View", key=f"search_view_{i}"):
                            st.session_state.selected_image = pair
                            st.rerun()

                    st.markdown("---")
        else:
            st.info("No results found. Try a different search term or date range.")

    # ----------------------
    # Word cloud by entity type
    # ----------------------
    st.markdown("---")
    st.subheader("Entity Word Cloud")

    # Load CSV here as well (cached)
    csv_path_wc = Path(data_dir) / "test.csv"
    if not csv_path_wc.exists():
        csv_path_wc = Path("test.csv")

    try:
        df_entities_wc = load_entities_csv(str(csv_path_wc))
    except Exception as e:
        st.error(f"Could not load entities CSV for word cloud: {e}")
        df_entities_wc = pd.DataFrame(columns=["Date", "Type", "Entity"])

    # Ensure expected columns exist
    if "Type" not in df_entities_wc.columns or "Entity" not in df_entities_wc.columns:
        st.info(
            'CSV must contain columns named "Type" and "Entity" to build the word cloud.'
        )
    else:
        # Debug info: number of rows and sample
        st.caption(f"Loaded rows: {len(df_entities_wc)}")
        with st.expander("Show sample data"):
            st.write(df_entities_wc.head(10))

        types = sorted(df_entities_wc["Type"].dropna().unique())
        st.caption(f"Available entity types: {', '.join(types[:10])}")

        if not types:
            st.info("No entity types found in CSV.")
        else:
            selected_type = st.selectbox("Select entity type", options=types)

            # Filter entities and compute frequencies
            subset = df_entities_wc[
                df_entities_wc["Type"].astype(str).str.lower()
                == str(selected_type).lower()
            ]
            freqs = subset["Entity"].astype(str).value_counts().to_dict()

            if not freqs:
                st.info(f"No entities found for type: {selected_type}")
            else:
                st.caption(f"Subset rows for type '{selected_type}': {len(subset)}")
                # show top 10 freqs for debugging
                with st.expander("Show top entities"):
                    top10 = list(freqs.items())[:10]
                    st.write(pd.DataFrame(top10, columns=["Entity", "Count"]))

                # Show debug info so users can see which python/venv Streamlit runs under
                st.caption(f"App Python: {sys.executable}")
                st.caption(
                    f"wordcloud available: {'yes' if WordCloud is not None else 'no'}"
                )

                # max words control
                max_words = st.slider(
                    "Max words", min_value=20, max_value=1000, value=200
                )

                if WordCloud is None:
                    st.warning(
                        'The Python package "wordcloud" is not available in the running interpreter. Showing a bar chart fallback.'
                    )
                    st.info(
                        "To enable the graphical word cloud, install it in the Python interpreter running this app:"
                    )
                    st.code(
                        f"{sys.executable} -m pip install wordcloud matplotlib",
                        language="bash",
                    )

                    # Fallback: show top entities as a bar chart
                    top_n = min(max_words, 50)
                    top_items = list(freqs.items())[:top_n]
                    ent_names = [k for k, v in top_items]
                    ent_counts = [v for k, v in top_items]
                    fig_bar = px.bar(
                        x=ent_names,
                        y=ent_counts,
                        labels={"x": "Entity", "y": "Count"},
                        title=f"Top entities for type: {selected_type}",
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    # Build the wordcloud from frequencies
                    try:
                        wc = WordCloud(
                            width=900,
                            height=400,
                            background_color="white",
                            max_words=max_words,
                            colormap="viridis",
                        )
                        wc.generate_from_frequencies(freqs)
                        img = wc.to_image()
                        st.image(
                            img,
                            use_column_width=True,
                            caption=f"Word cloud for type: {selected_type}",
                        )
                    except Exception as e:
                        st.error(f"Failed to generate word cloud: {e}")

# Tab 5: Topics
with tab5:
    st.title("Topic Analysis")

    # Load topic data
    @st.cache_data
    def load_topics_data():
        """Load topics CSV data"""
        csv_path = Path("../../issues_topics_llm.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            # Create date column
            df["date"] = pd.to_datetime(df[["year", "month", "day"]])
            # Expand topics (split by pipe)
            expanded = []
            for _, row in df.iterrows():
                if pd.notna(row["topics"]):
                    topics = row["topics"].split("|")
                    for topic in topics:
                        expanded.append(
                            {
                                "date": row["date"],
                                "year": row["year"],
                                "month": row["month"],
                                "day": row["day"],
                                "topic": topic.strip(),
                            }
                        )
            return pd.DataFrame(expanded)
        return None

    topics_df = load_topics_data()

    if topics_df is not None:
        # Filters
        st.sidebar.markdown("---")
        st.sidebar.subheader("Topic Filters")

        min_mentions = st.sidebar.slider("Min total mentions", 1, 50, 1)
        top_n = st.sidebar.slider("Show top N topics", 10, 100, 40)

        # Calculate topic frequencies
        topic_totals = topics_df.groupby("topic").size().sort_values(ascending=False)
        filtered_topics = topic_totals[topic_totals >= min_mentions].head(top_n)

        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Topics", len(topic_totals))
        with col2:
            st.metric("Filtered Topics", len(filtered_topics))
        with col3:
            st.metric(
                "Date Range", f"{topics_df['year'].min()}-{topics_df['year'].max()}"
            )

        st.markdown("---")

        # Prepare plot data
        summary_plot = topics_df[topics_df["topic"].isin(filtered_topics.index)].copy()

        if not summary_plot.empty:
            # Count occurrences per topic per date
            summary = (
                summary_plot.groupby(["date", "topic"]).size().reset_index(name="count")
            )

            # Create topic ordering and mapping
            topics_ordered = list(filtered_topics.index)
            topic_to_y = {t: i for i, t in enumerate(reversed(topics_ordered), start=1)}
            topic_to_idx = {t: i for i, t in enumerate(topics_ordered)}

            summary["y"] = summary["topic"].map(topic_to_y)
            summary["color_idx"] = summary["topic"].map(topic_to_idx)

            # Create scatter plot with Plotly
            fig = go.Figure()

            # Generate distinct colors
            num_topics = len(topics_ordered)
            import plotly.colors as pc

            if num_topics <= 10:
                colors = pc.qualitative.Plotly
            elif num_topics <= 24:
                colors = pc.qualitative.Dark24
            else:
                colors = pc.sample_colorscale(
                    "turbo", [i / (num_topics - 1) for i in range(num_topics)]
                )

            # Add scatter for each topic
            for i, topic in enumerate(topics_ordered):
                topic_data = summary[summary["topic"] == topic]
                color_idx = i % len(colors) if isinstance(colors, list) else i
                color = colors[color_idx] if isinstance(colors, list) else colors[i]

                fig.add_trace(
                    go.Scatter(
                        x=topic_data["date"],
                        y=topic_data["y"],
                        mode="markers",
                        name=topic[:50] + "..." if len(topic) > 50 else topic,
                        marker=dict(
                            size=topic_data["count"] * 3 + 5,
                            color=color,
                            opacity=0.7,
                            line=dict(width=0),
                        ),
                        text=topic_data.apply(
                            lambda row: f"{row['topic']}<br>Count: {row['count']}",
                            axis=1,
                        ),
                        hovertemplate="<b>%{text}</b><br>Date: %{x}<extra></extra>",
                    )
                )

            # Update layout
            inv_map = {v: k for k, v in topic_to_y.items()}
            y_ticks = sorted(set(summary["y"]))

            fig.update_layout(
                title="Topic Mentions Over Time (Size = Daily Mentions)",
                xaxis_title="Date",
                yaxis_title="Topic",
                yaxis=dict(
                    tickmode="array",
                    tickvals=y_ticks,
                    ticktext=[
                        inv_map[i][:60] + "..." if len(inv_map[i]) > 60 else inv_map[i]
                        for i in y_ticks
                    ],
                ),
                height=max(600, len(topics_ordered) * 25),
                showlegend=True,
                legend=dict(yanchor="top", y=1, xanchor="left", x=1.02),
                hovermode="closest",
            )

            st.plotly_chart(fig, use_container_width=True)

            # Topic frequency table
            st.markdown("---")
            st.subheader("Topic Frequency Table")

            freq_table = filtered_topics.reset_index()
            freq_table.columns = ["Topic", "Total Mentions"]
            st.dataframe(freq_table, use_container_width=True, hide_index=True)

            # Time series for selected topic
            st.markdown("---")
            st.subheader("Topic Time Series")

            selected_topic = st.selectbox("Select a topic", topics_ordered)

            if selected_topic:
                topic_time = summary[summary["topic"] == selected_topic].copy()

                fig2 = px.line(
                    topic_time,
                    x="date",
                    y="count",
                    title=f"Mentions over time: {selected_topic}",
                    markers=True,
                )
                fig2.update_traces(line_color="#FF6B6B", marker=dict(size=8))
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No topics match the current filters.")
    else:
        st.warning(
            "Topics data not found. Please check that `issues_topics_llm.csv` exists in the project root."
        )

# Tab 6: Emotions
with tab6:
    st.title("📊 Emotion Analysis")

    # Load emotion data
    @st.cache_data
    def load_emotion_data():
        """Load emotion analysis data"""
        # Get path relative to this script
        script_dir = Path(__file__).parent
        base_path = script_dir / "em"

        # Also try alternative paths
        alt_path = Path("./src/ui/em")

        # Try to load the main dataset
        data = {}

        # Try primary path first, then alternative
        for path_base in [base_path, alt_path]:
            means_path = path_base / "ALL_emotions_means_by_id.tsv"
            outliers_path = path_base / "ALL_emotions_means_by_id.outliers_topN.tsv"

            if means_path.exists() and "means" not in data:
                data["means"] = pd.read_csv(means_path, sep="\t")
            if outliers_path.exists() and "outliers" not in data:
                data["outliers"] = pd.read_csv(outliers_path, sep="\t")

            # If we found both files, stop searching
            if "means" in data and "outliers" in data:
                break

        return data

    emotion_data = load_emotion_data()

    if "means" in emotion_data and not emotion_data["means"].empty:
        df_means = emotion_data["means"]

        # Emotion configuration
        EMOTIONS = ["Joy", "Love", "Fear", "Anger", "Sadness", "Agitation"]
        emotion_cols = [f"mean_{e}" for e in EMOTIONS]

        # Positive/negative color mapping
        emotion_colors = {
            "Joy": "#2ca25f",  # green
            "Love": "#66c2a4",  # light green
            "Anger": "#b2182b",  # dark red
            "Fear": "#d6604d",  # reddish
            "Sadness": "#f4a582",  # salmon
            "Agitation": "#d7301f",  # orange-red
        }

        # === 1. Overall Emotion Distribution ===
        st.subheader("Overall Emotion Distribution")

        # Calculate average for each emotion across all records
        emotion_means = {}
        for emo in EMOTIONS:
            col = f"mean_{emo}"
            if col in df_means.columns:
                emotion_means[emo] = df_means[col].mean()

        # Calculate shares
        total = sum(emotion_means.values())
        emotion_shares = {
            k: (v / total if total > 0 else 0) for k, v in emotion_means.items()
        }

        # Create bar chart
        fig1 = go.Figure(
            go.Bar(
                x=list(emotion_shares.keys()),
                y=list(emotion_shares.values()),
                text=[f"{v:.1%}" for v in emotion_shares.values()],
                textposition="auto",
                marker_color=[emotion_colors[e] for e in emotion_shares.keys()],
            )
        )

        fig1.update_layout(
            title="Emotion Distribution (Relative Shares)",
            xaxis_title="Emotion",
            yaxis_title="Share",
            yaxis_tickformat=".0%",
            template="plotly_white",
            height=500,
        )

        st.plotly_chart(fig1, use_container_width=True)

        # === 2. Emotions Over Time ===
        st.subheader("Emotions Over Time")

        if "year" in df_means.columns:
            # Group by year and calculate means
            yearly_means = df_means.groupby("year")[emotion_cols].mean().reset_index()

            # Create line chart
            fig2 = go.Figure()

            for emo in EMOTIONS:
                col = f"mean_{emo}"
                if col in yearly_means.columns:
                    fig2.add_trace(
                        go.Scatter(
                            x=yearly_means["year"],
                            y=yearly_means[col],
                            mode="lines+markers",
                            name=emo,
                            line=dict(width=2, color=emotion_colors[emo]),
                            marker=dict(size=6, color=emotion_colors[emo]),
                        )
                    )

            fig2.update_layout(
                title="Emotion Trends Over Years",
                xaxis_title="Year",
                yaxis_title="Mean Value",
                template="plotly_white",
                height=500,
                hovermode="x unified",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
            )

            st.plotly_chart(fig2, use_container_width=True)

        # === 2b. Stream Chart (Relative Shares Over Time) ===
        st.subheader("Emotion Share Distribution Over Time")

        if "id" in df_means.columns:
            # Extract year-month from id column (format: 3074409X_YYYY-MM-DD_...)
            import re

            date_pattern = re.compile(r"(\d{4})-(\d{2})-(\d{2})")

            def extract_year_month(id_str):
                match = date_pattern.search(str(id_str))
                if match:
                    year, month = match.group(1), match.group(2)
                    return f"{year}-{month}"
                return None

            # Add year-month column
            df_means_copy = df_means.copy()
            df_means_copy["year_month"] = df_means_copy["id"].apply(extract_year_month)

            # Filter out rows without valid dates
            df_means_copy = df_means_copy[df_means_copy["year_month"].notna()]

            if not df_means_copy.empty:
                # Calculate relative shares per month
                monthly_sums = df_means_copy.groupby("year_month")[emotion_cols].sum()
                monthly_totals = monthly_sums.sum(axis=1)
                monthly_shares = monthly_sums.div(monthly_totals, axis=0).reset_index()

                # Convert year_month to datetime for proper plotting
                monthly_shares["date"] = pd.to_datetime(
                    monthly_shares["year_month"] + "-01"
                )
                monthly_shares = monthly_shares.sort_values("date")

                # Create stream chart
                fig_stream = go.Figure()

                # Plot order: negative emotions first, then positive
                plot_order = ["Agitation", "Anger", "Fear", "Sadness", "Love", "Joy"]

                for emo in plot_order:
                    col = f"mean_{emo}"
                    if col in monthly_shares.columns:
                        fig_stream.add_trace(
                            go.Scatter(
                                x=monthly_shares["date"],
                                y=monthly_shares[col],
                                name=emo,
                                mode="lines",
                                line=dict(width=0.5, color=emotion_colors[emo]),
                                stackgroup="one",
                                groupnorm="fraction",  # Normalize to show shares
                                hovertemplate=f"{emo}: %{{y:.1%}}<extra></extra>",
                            )
                        )

                fig_stream.update_layout(
                    title="Relative Emotion Shares Over Time (Stream Chart - Monthly)",
                    template="plotly_white",
                    height=500,
                    hovermode="x unified",
                    yaxis=dict(tickformat=".0%", title="Share"),
                    xaxis_title="Month",
                    xaxis=dict(tickformat="%Y-%m", dtick="M2"),  # Show every 2 months
                    legend=dict(
                        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                    ),
                )

                st.plotly_chart(fig_stream, use_container_width=True)

                st.caption(
                    "💡 This stream chart shows the relative proportions of emotions over time at monthly granularity, normalized to 100% for each month."
                )

        # === 3. Emotions by Era ===
        st.subheader("Emotions by Historical Era")

        if "era1" in df_means.columns:
            era_order = ["pre war", "war", "post war"]
            era_colors_map = {
                "pre war": "#2b8cbe",  # blue
                "war": "#f46d43",  # orange/red
                "post war": "#74add1",  # light blue
            }

            # Create box plots for each emotion
            col1, col2 = st.columns(2)

            for idx, emo in enumerate(EMOTIONS):
                col = f"mean_{emo}"
                if col not in df_means.columns:
                    continue

                # Filter data by era
                fig3 = go.Figure()

                for era in era_order:
                    era_data = df_means[df_means["era1"] == era]
                    if not era_data.empty:
                        fig3.add_trace(
                            go.Box(
                                y=era_data[col],
                                name=era,
                                marker_color=era_colors_map.get(era, "#999"),
                                boxpoints="outliers",
                            )
                        )

                fig3.update_layout(
                    title=f"{emo} by Era",
                    yaxis_title="Value",
                    template="plotly_white",
                    height=350,
                    showlegend=True,
                    boxmode="group",
                )

                if idx % 2 == 0:
                    col1.plotly_chart(fig3, use_container_width=True)
                else:
                    col2.plotly_chart(fig3, use_container_width=True)

        # === 4. Top Outliers ===
        if "outliers" in emotion_data and not emotion_data["outliers"].empty:
            st.subheader("Notable Emotion Peaks")

            df_outliers = emotion_data["outliers"]

            # Select emotion to view outliers
            selected_emo = st.selectbox("Select emotion to view peaks", EMOTIONS)

            if selected_emo:
                emo_outliers = df_outliers[df_outliers["emotion"] == selected_emo].head(
                    10
                )

                if not emo_outliers.empty:
                    # Display table
                    display_cols = ["date", "era1", "value", "z_mad"]
                    available_cols = [
                        c for c in display_cols if c in emo_outliers.columns
                    ]

                    st.dataframe(
                        emo_outliers[available_cols].style.format(
                            {"value": "{:.4f}", "z_mad": "{:.2f}"}
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # Plot outliers over time
                    if (
                        "date" in emo_outliers.columns
                        and "value" in emo_outliers.columns
                    ):
                        # Create a copy and add absolute z_mad for sizing (must be positive)
                        plot_data = emo_outliers.copy()
                        if "z_mad" in plot_data.columns:
                            plot_data["abs_z_mad"] = plot_data["z_mad"].abs()
                            size_col = "abs_z_mad"
                        else:
                            size_col = None

                        fig4 = px.scatter(
                            plot_data,
                            x="date",
                            y="value",
                            size=size_col,
                            color="era1" if "era1" in plot_data.columns else None,
                            title=f"Top {selected_emo} Peaks Over Time",
                            hover_data=(
                                ["era1", "z_mad"]
                                if "era1" in plot_data.columns
                                and "z_mad" in plot_data.columns
                                else (
                                    ["z_mad"]
                                    if "z_mad" in plot_data.columns
                                    else (
                                        ["era1"]
                                        if "era1" in plot_data.columns
                                        else None
                                    )
                                )
                            ),
                        )
                        fig4.update_layout(template="plotly_white", height=400)
                        st.plotly_chart(fig4, use_container_width=True)
                else:
                    st.info(f"No outlier data available for {selected_emo}")

        # === 5. Summary Statistics ===
        with st.expander("📈 Summary Statistics"):
            st.markdown("### Emotion Statistics by Era")

            if "era1" in df_means.columns:
                for era in ["pre war", "war", "post war"]:
                    st.markdown(f"**{era.upper()}**")
                    era_data = df_means[df_means["era1"] == era]

                    if not era_data.empty:
                        stats = {}
                        for emo in EMOTIONS:
                            col = f"mean_{emo}"
                            if col in era_data.columns:
                                stats[emo] = {
                                    "Mean": era_data[col].mean(),
                                    "Std": era_data[col].std(),
                                    "Min": era_data[col].min(),
                                    "Max": era_data[col].max(),
                                }

                        stats_df = pd.DataFrame(stats).T
                        st.dataframe(
                            stats_df.style.format("{:.4f}"), use_container_width=True
                        )
                    st.markdown("---")
    else:
        st.warning(
            "Emotion data not found. Please ensure emotion analysis files exist in `src/ui/em/` directory."
        )
        st.info(
            "Expected files:\n"
            "- ALL_emotions_means_by_id.tsv\n"
            "- ALL_emotions_means_by_id.outliers_topN.tsv"
        )

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
**Quick Tips:**
- Use date filters to narrow temporal scope
- Combine entity + emotion filters for targeted analysis
- Click on graph nodes to see related pages
- Export results via the menu
"""
)
