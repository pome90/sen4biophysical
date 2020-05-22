import pytest
import pandas
import numpy as np

import sys
sys.path.append('../sen4biophysical')

from sen4biophysical import LAI, CAB, FAPAR

@pytest.fixture
def lai_xy():
    df = pandas.read_csv('./tests/auxdata/lai_test_cases.txt', header=None)
    arr = df.to_numpy(dtype="float32")
    x = arr[..., :11]
    y = arr[..., 11:]
    return x, y


@pytest.fixture
def cab_xy():
    df = pandas.read_csv('./tests/auxdata/cab_test_cases.txt', header=None)
    arr = df.to_numpy(dtype="float32")
    x = arr[..., :11]
    y = arr[..., 11:]
    return x, y


@pytest.fixture
def fapar_xy():
    df = pandas.read_csv('./tests/auxdata/fapar_test_cases.txt', header=None)
    arr = df.to_numpy(dtype="float32")
    x = arr[..., :11]
    y = arr[..., 11:]
    return x, y


def _assert_equal(out, y, mask):
    out[mask] = np.nan
    y[mask] = np.nan
    out[np.isnan(y)] = np.nan
    np.testing.assert_array_almost_equal(out, y, decimal=2)


def test_lai(lai_xy):
    x, y = lai_xy
    out, mask = LAI()(x)
    _assert_equal(out, y, mask)


def test_cab(cab_xy):
    x, y = cab_xy
    out, mask = CAB()(x)
    _assert_equal(out, y, mask)


def test_fapar(fapar_xy):
    x, y = fapar_xy
    out, mask = FAPAR()(x)
    _assert_equal(out, y, mask)