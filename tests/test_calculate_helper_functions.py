# python
import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from pysteps.postprocessing.diagnostics import calculate_precip_type
# calculate_precip_type(snow_level_grid_m, temperature_grid_degC, ground_temperature_grid_degCerature_grid_degC, precipitation_intensity_grid_mmph, topography_grid_m, melting_layer_thickness_m=100., freezing_rain_2m_temperature_with_frozen_ground_degC=2., freezing_rain_temperature_threshold_degC=0., minimum_precipitation_threshold_mmph=0)
# Test the calculate_precip_type function with different scenarios:
    # PT=0  no precip
    # PT=1  rain
    # PT=2  rain/snow mix
    # PT=3  snow
    # PT=4  freezing rain
def test_calculate_precip_type_dry():
    # Test with no precipitation
    precipitation_intensity_grid_mmph = np.zeros((3, 4))
    snow_level_grid_m = np.zeros((3, 4))
    topography_grid_m = np.zeros((3, 4))
    ground_temperature_grid_degC = np.zeros((3, 4))
    temperature_grid_degC = np.ones((3, 4))
    result = calculate_precip_type(snow_level_grid_m, temperature_grid_degC, ground_temperature_grid_degC, precipitation_intensity_grid_mmph, topography_grid_m)
    expected_result = np.zeros((3, 4))
    assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_rain():
    # Test with rain
    precipitation_intensity_grid_mmph = np.ones((3, 4))
    snow_level_grid_m = np.ones((3, 4))* 200
    topography_grid_m = np.zeros((3, 4))
    ground_temperature_grid_degC = np.ones((3, 4))*5
    temperature_grid_degC = np.ones((3, 4)) * 10
    result = calculate_precip_type(snow_level_grid_m, temperature_grid_degC, ground_temperature_grid_degC, precipitation_intensity_grid_mmph, topography_grid_m)
    expected_result = np.ones((3, 4))
    assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_snow_topohigh():
    # Test with snow and high topography above snow level (at least 1.5 times melting layer that is 100m default)
    precipitation_intensity_grid_mmph = np.ones((3, 4)) # precipitation of 1 mm/h
    snow_level_grid_m = np.ones((3, 4)) * 100 # snow level at 100 m
    topography_grid_m = np.ones((3, 4)) * 300 # topography at 300 m
    ground_temperature_grid_degC = np.ones((3, 4)) * -10
    temperature_grid_degC = np.ones((3, 4)) * -5
    result = calculate_precip_type(snow_level_grid_m, temperature_grid_degC, ground_temperature_grid_degC, precipitation_intensity_grid_mmph, topography_grid_m)
    expected_result = np.ones((3, 4)) * 3
    assert_array_almost_equal(result, expected_result)

### The following will never give snow since topography < snow level: should be freezing rain case T2m < TTO and TG < TG0
# def test_calculate_precip_type_snow_topolow():
#     # Test with snow but topography is lower than snow level
#     precipitation_intensity_grid_mmph = np.ones((3, 4)) # precipitation of 1 mm/h
#     snow_level_grid_m = np.ones((3, 4)) * 200 # snow level at 200 m
#     topography_grid_m = np.ones((3, 4)) * 100 # topography at 100 m
#     ground_temperature_grid_degC = np.ones((3, 4)) * -10
#     temperature_grid_degC = np.ones((3, 4)) * -5
#     result = calculate_precip_type(snow_level_grid_m, temperature_grid_degC, ground_temperature_grid_degC, precipitation_intensity_grid_mmph, topography_grid_m)
#     expected_result = np.ones((3, 4)) * 3
#     assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_snow_topolow():
    # Test with snow and topography above snow level but difference less than 1.5 times melting layer higher
    # Snow only in lower evelation where topo <= 1.5 melting layer (100m) = 150m
    precipitation_intensity_grid_mmph = np.ones((3, 4)) # precipitation of 1 mm/h
    snow_level_grid_m = np.ones((3, 4)) * 100 # snow level at 100 m
    topography_grid_m = np.ones((3, 4)) * 140 # topography at 140 m
    ground_temperature_grid_degC = np.ones((3, 4)) * -10
    temperature_grid_degC = np.ones((3, 4)) * -5
    result = calculate_precip_type(snow_level_grid_m, temperature_grid_degC, ground_temperature_grid_degC, precipitation_intensity_grid_mmph, topography_grid_m)
    expected_result = np.ones((3, 4)) * 3
    assert_array_almost_equal(result, expected_result)

def test_calculate_precip_type_freezing_rain():
    # Test with freezing rain
    # T2m < TTO and TG < TG0
    precipitation_intensity_grid_mmph = np.ones((3, 4))
    snow_level_grid_m = np.ones((3, 4)) * 200
    topography_grid_m = np.ones((3, 4)) * 100
    ground_temperature_grid_degC = np.ones((3, 4)) * -10
    temperature_grid_degC = np.ones((3, 4)) * 1
    result = calculate_precip_type(snow_level_grid_m, temperature_grid_degC, ground_temperature_grid_degC, precipitation_intensity_grid_mmph, topography_grid_m)
    expected_result = np.ones((3, 4)) * 4
    assert_array_almost_equal(result, expected_result)

    # T2m < TG0
    precipitation_intensity_grid_mmph = np.ones((3, 4))
    snow_level_grid_m = np.ones((3, 4)) * 200
    topography_grid_m = np.ones((3, 4)) * 100
    ground_temperature_grid_degC = np.ones((3, 4)) * 1
    temperature_grid_degC = np.ones((3, 4)) * -2
    result = calculate_precip_type(snow_level_grid_m, temperature_grid_degC, ground_temperature_grid_degC, precipitation_intensity_grid_mmph, topography_grid_m)
    expected_result = np.ones((3, 4)) * 4
    assert_array_almost_equal(result, expected_result)

# def test_calculate_precip_type_rain_snow_mix():