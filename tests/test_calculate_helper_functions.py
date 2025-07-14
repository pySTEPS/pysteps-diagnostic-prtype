# python
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal

# Test the calculate_precip_type function with different scenarios:
    # PT=0  no precip
    # PT=1  rain
    # PT=2  rain/snow mix
    # PT=3  snow
    # PT=4  freezing rain
def test_calculate_precip_type_dry():
    # Test with no precipitation
    precip = np.zeros((2, 3, 4))
    snow_level = np.zeros((2, 3, 4))
    topography = np.zeros((3, 4))
    ground_temp = np.zeros((2, 3, 4))
    temp = np.ones((2, 3, 4))
    result = calculate_precip_type(precip, snow_level, topography, ground_temp, temp)
    expected_result = np.zeros((2, 3, 4))
    assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_rain():
    # Test with rain
    precip = np.ones((2, 3, 4))
    snow_level = np.ones((2, 3, 4))* 200
    topography = np.zeros((3, 4))
    ground_temp = np.ones((2, 3, 4))*5
    temp = np.ones((2, 3, 4)) * 10
    result = calculate_precip_type(precip, snow_level, topography, ground_temp, temp)
    expected_result = np.ones((2, 3, 4))
    assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_snow():
    # Test with snow
    precip = np.ones((2, 3, 4))
    snow_level = np.ones((2, 3, 4)) * 200
    topography = np.ones((3, 4)) * 100
    ground_temp = np.ones((2, 3, 4)) * -10
    temp = np.ones((2, 3, 4)) * -5
    result = calculate_precip_type(precip, snow_level, topography, ground_temp, temp)
    expected_result = np.ones((2, 3, 4)) * 3
    assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_freezing_rain():
    # Test with freezing rain
    # T2m < TTO and TG < TG0
    precip = np.ones((2, 3, 4))
    snow_level = np.ones((2, 3, 4)) * 200
    topography = np.ones((3, 4)) * 100
    ground_temp = np.ones((2, 3, 4)) * -10
    temp = np.ones((2, 3, 4)) * 1
    result = calculate_precip_type(precip, snow_level, topography, ground_temp, temp)
    expected_result = np.ones((2, 3, 4)) * 4
    assert_array_almost_equal(result, expected_result)

    # T2m < TGO
    precip = np.ones((2, 3, 4))
    snow_level = np.ones((2, 3, 4)) * 200
    topography = np.ones((3, 4)) * 100
    ground_temp = np.ones((2, 3, 4)) * 1
    temp = np.ones((2, 3, 4)) * -5
    result = calculate_precip_type(precip, snow_level, topography, ground_temp, temp)
    expected_result = np.ones((2, 3, 4)) * 4
    assert_array_almost_equal(result, expected_result)

# def test_calculate_precip_type_rain_snow_mix():