"""
Tests for layout detection functionality.

Unit tests use mocked YOLO models for fast execution.
Integration tests (marked with @pytest.mark.integration) use real images and models.
"""

from pathlib import Path
from unittest.mock import Mock, MagicMock, patch
import pytest
import numpy as np

from newspaper_explorer.analyze.layout.detection import LayoutDetector, DOCLAYNET_CLASSES
from newspaper_explorer.analyze.layout.schemas import Detection, BoundingBox, PageLayout


# Fixtures
@pytest.fixture
def sample_image_path():
    """Path to sample newspaper image for testing."""
    fixtures_dir = Path(__file__).parent.parent.parent / "fixtures" / "images"
    # Use one of our fixture images
    sample_path = fixtures_dir / "page_1900_01_02_01.jpg"

    if sample_path.exists():
        return sample_path
    else:
        pytest.skip("Sample image not found. Fixture images should exist in tests/fixtures/images/")


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for unit tests."""
    with patch("newspaper_explorer.analyze.layout.detection.YOLO") as mock_yolo:
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance
        yield mock_model_instance


@pytest.fixture
def mock_detection_result():
    """Create a mock YOLO detection result."""
    mock_result = Mock()
    mock_boxes = Mock()
    mock_boxes.__len__ = Mock(return_value=3)
    mock_result.boxes = mock_boxes
    mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
        [
            [10.0, 20.0, 100.0, 200.0],  # Title
            [150.0, 50.0, 300.0, 150.0],  # Picture
            [150.0, 160.0, 300.0, 180.0],  # Caption
        ]
    )
    mock_boxes.cls.cpu.return_value.numpy.return_value = np.array(
        [10, 6, 0]
    )  # Title, Picture, Caption
    mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.95, 0.87, 0.82])
    mock_result.names = {0: "Caption", 6: "Picture", 10: "Title"}
    return mock_result


class TestDocLayNetClasses:
    """Test DOCLAYNET_CLASSES constant"""

    def test_all_classes_present(self):
        """Test that all 11 DocLayNet classes are defined"""
        assert len(DOCLAYNET_CLASSES) == 11
        expected = [
            "Caption",
            "Footnote",
            "Formula",
            "List-item",
            "Page-footer",
            "Page-header",
            "Picture",
            "Section-header",
            "Table",
            "Text",
            "Title",
        ]
        assert DOCLAYNET_CLASSES == expected


class TestLayoutDetector:
    """Test LayoutDetector class"""

    @patch("huggingface_hub.hf_hub_download")
    @patch("newspaper_explorer.analyze.layout.detection.YOLO")
    def test_init_download_from_huggingface(self, mock_yolo, mock_hf_download):
        """Test initialization with HuggingFace download"""
        mock_hf_download.return_value = "/downloaded/model.pt"

        detector = LayoutDetector(model_size="small", device="cuda:0")

        mock_hf_download.assert_called_once()
        call_kwargs = mock_hf_download.call_args[1]
        assert call_kwargs["repo_id"] == "Armaggheddon/yolo11-document-layout"
        assert call_kwargs["filename"] == "yolo11s_doc_layout.pt"

        mock_yolo.assert_called_once_with("/downloaded/model.pt")

    @patch("newspaper_explorer.analyzeayout.detection.YOLO")
    def test_init_invalid_model_size(self, mock_yolo):
        """Test initialization with invalid model size"""
        with pytest.raises(RuntimeError, match="Failed to load model from HuggingFace"):
            LayoutDetector(model_size="invalid")

    @patch("huggingface_hub.hf_hub_download")
    @patch("newspaper_explorer.analyze.layout.detection.YOLO")
    def test_extract_detections(self, mock_yolo, mock_hf_download):
        """Test _extract_detections method"""
        mock_hf_download.return_value = "/fake/model.pt"
        detector = LayoutDetector(model_size="medium", device="cpu")

        # Mock YOLO detection result
        mock_result = Mock()
        # Make mock_result.boxes have a length
        mock_boxes = Mock()
        mock_boxes.__len__ = Mock(return_value=2)
        mock_result.boxes = mock_boxes

        # Create fake detection data
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
            [
                [10.0, 20.0, 100.0, 200.0],  # Box 1
                [150.0, 50.0, 300.0, 150.0],  # Box 2
            ]
        )
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([0, 6])  # Caption, Picture
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.95, 0.87])
        mock_result.names = {0: "Caption", 6: "Picture"}

        detections = detector._extract_detections(mock_result, "test_page_001")

        assert len(detections) == 2

        # Check first detection
        det1 = detections[0]
        # Detection ID should be UUID-based now, just check format and uniqueness
        assert det1.detection_id.startswith("test_page_001_caption_")
        assert len(det1.detection_id.split("_")) >= 4  # page_id + class + uuid
        assert det1.class_name == "Caption"
        assert det1.confidence == 0.95
        assert det1.bbox.x1 == 10.0
        assert det1.bbox.y1 == 20.0
        assert det1.bbox.x2 == 100.0
        assert det1.bbox.y2 == 200.0
        assert det1.page_id == "test_page_001"

        # Check second detection
        det2 = detections[1]
        assert det2.detection_id.startswith("test_page_001_picture_")
        assert det2.class_name == "Picture"
        assert det2.confidence == 0.87

        # Ensure detection IDs are unique
        assert det1.detection_id != det2.detection_id

    @patch("huggingface_hub.hf_hub_download")
    @patch("newspaper_explorer.analyze.layout.detection.YOLO")
    def test_extract_detections_empty(self, mock_yolo, mock_hf_download):
        """Test _extract_detections with no detections"""
        mock_hf_download.return_value = "/fake/model.pt"
        detector = LayoutDetector(model_size="medium", device="cpu")

        # Mock empty result
        mock_result = Mock()
        mock_result.boxes = None

        detections = detector._extract_detections(mock_result, "empty_page")

        assert len(detections) == 0
        assert detections == []

    @patch("huggingface_hub.hf_hub_download")
    @patch("newspaper_explorer.analyzelayout.detection.YOLO")
    def test_extract_detections_with_foreign_keys(self, mock_yolo, mock_hf_download):
        """Test _extract_detections extracts foreign keys from page_id"""
        mock_hf_download.return_value = "/fake/model.pt"
        detector = LayoutDetector(model_size="medium", device="cpu")

        # Mock YOLO detection result
        mock_result = Mock()
        mock_boxes = Mock()
        mock_boxes.__len__ = Mock(return_value=1)
        mock_result.boxes = mock_boxes

        # Create fake detection data
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10.0, 20.0, 100.0, 200.0]])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([10])  # Title
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.95])
        mock_result.names = {10: "Title"}

        # Use real page_id format with ZDB source ID
        page_id = "3074409-X_1901-01-08_006_1_001"
        detections = detector._extract_detections(mock_result, page_id)

        assert len(detections) == 1
        det = detections[0]

        # Check foreign keys are extracted
        assert det.source_id == "3074409-X"
        assert det.issue_id == "3074409-X_1901-01-08_006_1"
        assert det.page_id == page_id

        # Check detection_id format
        assert det.detection_id.startswith(f"{page_id}_title_")
        assert len(det.detection_id) > len(page_id) + 7  # page_id + "_title_" + uuid

    @patch("newspaper_explorer.analyzeayout.detection.YOLO")
    def test_detect_page(self, mock_yolo):
        """Test detect_page method"""
        # Setup mock model
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Mock prediction result
        mock_result = Mock()
        mock_boxes = Mock()
        mock_boxes.__len__ = Mock(return_value=1)
        mock_result.boxes = mock_boxes
        mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array([[10, 20, 100, 200]])
        mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([10])  # Title
        mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.92])
        mock_result.names = {10: "Title"}

        mock_model_instance.predict.return_value = [mock_result]

        with patch("huggingface_hub.hf_hub_download", return_value="/fake/model.pt"):
            detector = LayoutDetector(model_size="medium", device="cpu")

        # Test detection
        with patch("pathlib.Path.exists", return_value=True):
            layout = detector.detect_page("/fake/image.jpg", page_id="page_001")

        assert isinstance(layout, PageLayout)
        assert layout.page_id == "page_001"
        assert layout.image_path == "/fake/image.jpg"
        assert len(layout.detections) == 1
        assert layout.detections[0].class_name == "Title"
        assert layout.total_detections == 1

    @patch("newspaper_explorer.analyzet.detection.YOLO")
    def test_detect_page_auto_page_id(self, mock_yolo):
        """Test detect_page with automatic page_id from filename"""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Mock empty result for simplicity
        mock_result = Mock()
        mock_result.boxes = None
        mock_model_instance.predict.return_value = [mock_result]

        with patch("huggingface_hub.hf_hub_download", return_value="/fake/model.pt"):
            detector = LayoutDetector(model_size="medium", device="cpu")

        with patch("pathlib.Path.exists", return_value=True):
            layout = detector.detect_page("/images/1902_01_01_001.jpg")

        # Page ID should be extracted from filename stem
        assert layout.page_id == "1902_01_01_001"

    @patch("newspaper_explorer.analyzeetection.YOLO")
    def test_detect_batch(self, mock_yolo):
        """Test detect_batch method with real images"""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Mock results for 3 pages
        mock_results = []
        for i in range(3):
            mock_result = Mock()
            mock_boxes = Mock()
            mock_boxes.__len__ = Mock(return_value=1)
            mock_result.boxes = mock_boxes
            mock_boxes.xyxy.cpu.return_value.numpy.return_value = np.array(
                [[10 + i * 10, 20, 100, 200]]
            )
            mock_boxes.cls.cpu.return_value.numpy.return_value = np.array([6])  # Picture
            mock_boxes.conf.cpu.return_value.numpy.return_value = np.array([0.8 + i * 0.05])
            mock_result.names = {6: "Picture"}
            mock_results.append(mock_result)

        mock_model_instance.predict.return_value = mock_results

        with patch("huggingface_hub.hf_hub_download", return_value="/fake/model.pt"):
            detector = LayoutDetector(model_size="medium", device="cpu", batch_size=2)

        # Use real fixture images
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures" / "images"
        image_paths = [
            str(fixtures_dir / "page_1900_01_02_01.jpg"),
            str(fixtures_dir / "page_1900_01_02_05.jpg"),
            str(fixtures_dir / "page_1920_03_03_01.jpg"),
        ]
        page_ids = [f"page_{i:03d}" for i in range(3)]

        layouts = detector.detect_batch(image_paths, page_ids)

        assert len(layouts) == 3
        for i, layout in enumerate(layouts):
            assert layout.page_id == f"page_{i:03d}"
            assert len(layout.detections) == 1
            assert layout.detections[0].class_name == "Picture"

    @patch("newspaper_explorer.analyzelayout.detection.YOLO")
    def test_detect_batch_auto_page_ids(self, mock_yolo):
        """Test detect_batch with automatic page_id generation using real images"""
        mock_model_instance = MagicMock()
        mock_yolo.return_value = mock_model_instance

        # Mock empty results
        mock_result = Mock()
        mock_result.boxes = None
        mock_model_instance.predict.return_value = [mock_result, mock_result]

        with patch("huggingface_hub.hf_hub_download", return_value="/fake/model.pt"):
            detector = LayoutDetector(model_size="medium", device="cpu")

        # Use real fixture images
        fixtures_dir = Path(__file__).parent.parent.parent / "fixtures" / "images"
        image_paths = [
            str(fixtures_dir / "page_1920_03_03_01.jpg"),
            str(fixtures_dir / "page_1920_03_03_04.jpg"),
        ]

        layouts = detector.detect_batch(image_paths)

        assert len(layouts) == 2
        assert layouts[0].page_id == "page_1920_03_03_01"
        assert layouts[1].page_id == "page_1920_03_03_04"

    @patch("newspaper_explorer.analyze.layout.detection.YOLO")
    def test_detect_batch_mismatched_lengths(self, mock_yolo):
        """Test detect_batch raises error with mismatched lengths"""
        with patch("huggingface_hub.hf_hub_download", return_value="/fake/model.pt"):
            detector = LayoutDetector(model_size="medium", device="cpu")

        image_paths = ["/img1.jpg", "/img2.jpg"]
        page_ids = ["page1"]  # Wrong length

        with pytest.raises(ValueError, match="page_ids must have same length"):
            detector.detect_batch(image_paths, page_ids)


class TestPageLayout:
    """Test PageLayout schema"""

    def test_page_layout_creation(self):
        """Test creating a PageLayout object"""
        det1 = Detection(
            detection_id="det_1",
            class_name="Title",
            confidence=0.95,
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=50),
            page_id="page_001",
        )

        det2 = Detection(
            detection_id="det_2",
            class_name="Picture",
            confidence=0.88,
            bbox=BoundingBox(x1=50, y1=100, x2=200, y2=300),
            page_id="page_001",
        )

        layout = PageLayout(
            page_id="page_001",
            image_path="/images/page_001.jpg",
            detections=[det1, det2],
        )

        assert layout.page_id == "page_001"
        assert layout.total_detections == 2
        assert len(layout.detections) == 2

    def test_page_layout_counts(self):
        """Test counts property"""
        detections = [
            Detection(
                detection_id=f"det_{i}",
                class_name=cls,
                confidence=0.9,
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=50),
                page_id="page_001",
            )
            for i, cls in enumerate(
                ["Title", "Picture", "Picture", "Caption", "Text", "Text", "Text"]
            )
        ]

        layout = PageLayout(
            page_id="page_001",
            image_path="/images/page_001.jpg",
            detections=detections,
        )

        counts = layout.counts
        assert counts["Title"] == 1
        assert counts["Picture"] == 2
        assert counts["Caption"] == 1
        assert counts["Text"] == 3

    def test_page_layout_filter_by_class(self):
        """Test filter_by_class method"""
        detections = [
            Detection(
                detection_id=f"det_{i}",
                class_name=cls,
                confidence=0.9,
                bbox=BoundingBox(x1=10, y1=20, x2=100, y2=50),
                page_id="page_001",
            )
            for i, cls in enumerate(["Title", "Picture", "Caption", "Picture", "Text"])
        ]

        layout = PageLayout(
            page_id="page_001",
            image_path="/images/page_001.jpg",
            detections=detections,
        )

        # Filter single class
        pictures = layout.filter_by_class("Picture")
        assert len(pictures) == 2
        assert all(d.class_name == "Picture" for d in pictures)

        # Filter multiple classes
        headlines = layout.filter_by_class(["Title", "Section-header", "Page-header"])
        assert len(headlines) == 1
        assert headlines[0].class_name == "Title"

    def test_page_layout_empty_detections(self):
        """Test PageLayout with no detections"""
        layout = PageLayout(
            page_id="empty_page",
            image_path="/images/empty.jpg",
            detections=[],
        )

        assert layout.total_detections == 0
        assert layout.counts == {}
        assert layout.filter_by_class("Picture") == []


class TestDetection:
    """Test Detection schema"""

    def test_detection_creation(self):
        """Test creating a Detection object"""
        bbox = BoundingBox(x1=10.5, y1=20.5, x2=100.5, y2=200.5)

        det = Detection(
            detection_id="page_001_det_5",
            class_name="Picture",
            confidence=0.92,
            bbox=bbox,
            page_id="page_001",
        )

        assert det.detection_id == "page_001_det_5"
        assert det.class_name == "Picture"
        assert det.confidence == 0.92
        assert det.page_id == "page_001"
        assert det.bbox.x1 == 10.5

    def test_detection_with_optional_fields(self):
        """Test Detection with optional fields"""
        det = Detection(
            detection_id="det_1",
            class_name="Picture",
            confidence=0.95,
            bbox=BoundingBox(x1=10, y1=20, x2=100, y2=200),
            page_id="page_001",
            image_path="/images/page_001.jpg",
            text_content="Image description",
            caption_text="Figure 1: Example",
        )

        assert det.image_path == "/images/page_001.jpg"
        assert det.text_content == "Image description"
        assert det.caption_text == "Figure 1: Example"


class TestBoundingBox:
    """Test BoundingBox schema"""

    def test_bbox_properties(self):
        """Test BoundingBox computed properties"""
        bbox = BoundingBox(x1=10, y1=20, x2=110, y2=120)

        assert bbox.width == 100
        assert bbox.height == 100
        assert bbox.center_x == 60
        assert bbox.center_y == 70
        assert bbox.area == 10000

    def test_bbox_iou_no_overlap(self):
        """Test IoU with no overlap"""
        bbox1 = BoundingBox(x1=0, y1=0, x2=10, y2=10)
        bbox2 = BoundingBox(x1=20, y1=20, x2=30, y2=30)

        assert bbox1.iou(bbox2) == 0.0

    def test_bbox_iou_full_overlap(self):
        """Test IoU with full overlap"""
        bbox1 = BoundingBox(x1=10, y1=10, x2=20, y2=20)
        bbox2 = BoundingBox(x1=10, y1=10, x2=20, y2=20)

        assert bbox1.iou(bbox2) == 1.0

    def test_bbox_iou_partial_overlap(self):
        """Test IoU with partial overlap"""
        bbox1 = BoundingBox(x1=0, y1=0, x2=10, y2=10)  # Area = 100
        bbox2 = BoundingBox(x1=5, y1=5, x2=15, y2=15)  # Area = 100

        # Intersection = 5x5 = 25
        # Union = 100 + 100 - 25 = 175
        # IoU = 25/175 ≈ 0.143
        iou = bbox1.iou(bbox2)
        assert 0.14 < iou < 0.15


# Integration Tests (slower, use real models and images)
@pytest.mark.integration
class TestLayoutDetectorIntegration:
    """Integration tests with real images and models."""

    def test_detect_real_image_with_mock_model(
        self, sample_image_path, mock_yolo_model, mock_detection_result
    ):
        """Test detection on real image with mocked model (validates I/O)."""
        mock_yolo_model.predict.return_value = [mock_detection_result]

        with patch("huggingface_hub.hf_hub_download", return_value="/fake/model.pt"):
            detector = LayoutDetector(
                model_size="medium",
                device="cpu",
                conf_threshold=0.2,
            )

        layout = detector.detect_page(sample_image_path)

        # Page ID is extracted from filename
        assert layout.page_id == "page_1900_01_02_01"
        assert layout.image_path == str(sample_image_path)
        assert len(layout.detections) == 3
        assert layout.counts["Title"] == 1
        assert layout.counts["Picture"] == 1
        assert layout.counts["Caption"] == 1

    @pytest.mark.slow
    def test_detect_real_image_real_model(self, sample_image_path):
        """
        Test detection with real model (downloads from HuggingFace).

        This test is marked @pytest.mark.slow because it:
        - Downloads ~100MB model on first run
        - Runs actual GPU/CPU inference (~30s runtime)

        Run with: pytest -m slow
        Skip with: pytest -m "not slow" (default)
        """
        detector = LayoutDetector(
            model_size="nano",  # Smallest model for faster testing
            device="cpu",
            conf_threshold=0.2,
            batch_size=1,
        )

        layout = detector.detect_page(sample_image_path)

        # Basic sanity checks
        assert layout.page_id == "page_1900_01_02_01"
        assert layout.total_detections > 0

        # Should detect at least some common elements
        assert len(layout.detections) > 0

        # All detections should be valid DocLayNet classes
        for det in layout.detections:
            assert det.class_name in DOCLAYNET_CLASSES
            assert 0.0 <= det.confidence <= 1.0
            assert det.bbox.width > 0
            assert det.bbox.height > 0

            # Check foreign keys are populated
            assert det.page_id == "page_1900_01_02_01"
            # source_id and issue_id might be None due to non-standard page_id format
            # but detection_id should exist and follow UUID format
            assert det.detection_id.startswith("page_1900_01_02_01_")
            assert len(det.detection_id.split("_")) >= 6  # page parts + class + uuid
