import pytest
import rasterio as rio
import numpy as np

TEST_TIF1_PATH = "example/singapore_org.tif"
TEST_TIF2_PATH = "example/singapore_pred.tif"

@pytest.fixture
def test_array1():
    with rio.open(TEST_TIF1_PATH) as tif:
        img_array =  tif.read()
        # transposing to move the no. of channels as third dimension
        return np.transpose(img_array, (1,2,0))

@pytest.fixture
def test_array2():
    with rio.open(TEST_TIF2_PATH) as tif:
        img_array =  tif.read()
        # transposing to move the no. of channels as third dimension
        return np.transpose(img_array, (1,2,0))
