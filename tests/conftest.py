import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


needs_tesseract = pytest.mark.skipif(
    shutil.which("tesseract") is None,
    reason="tesseract binary not installed",
)


@pytest.fixture(scope="session")
def image1_path():
    return ROOT / "Images" / "image1.JPG"


@pytest.fixture(scope="session")
def image2_path():
    return ROOT / "Images" / "image2.JPG"


@pytest.fixture(scope="session")
def image1_bgr(image1_path):
    import cv2
    img = cv2.imread(str(image1_path))
    assert img is not None, f"could not load {image1_path}"
    return img


@pytest.fixture(scope="session")
def image2_bgr(image2_path):
    import cv2
    img = cv2.imread(str(image2_path))
    assert img is not None, f"could not load {image2_path}"
    return img
