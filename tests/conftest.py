from pathlib import Path

import numpy as np
import pytest
import rasterio as rio

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "example"
TEST_TIF1_PATH = EXAMPLE_DIR / "singapore_org.tif"
TEST_TIF2_PATH = EXAMPLE_DIR / "singapore_pred.tif"


@pytest.fixture
def test_array1():
    with rio.open(TEST_TIF1_PATH) as tif:
        img_array = tif.read()
        # transposing to move the no. of channels as third dimension
        return np.transpose(img_array, (1, 2, 0))


@pytest.fixture
def test_array2():
    with rio.open(TEST_TIF2_PATH) as tif:
        img_array = tif.read()
        # transposing to move the no. of channels as third dimension
        return np.transpose(img_array, (1, 2, 0))
