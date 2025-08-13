#!/usr/bin/env python

"""Tests for `pysteps_diagnostic_prtype` package with artificial data."""

def test_plugins_discovery():
    """It is recommended to at least test that the plugin modules provided by the plugin are
    correctly detected by pysteps. For this, the tests should be ran on the installed
    version of the plugin (and not against the plugin sources).
    """
    # plugin exists as interface method
    from pysteps.postprocessing import interface as pp_interface
    assert 'diagnostic_prtype' in pp_interface._diagnostics_methods
    # plugin exists as module
    import importlib
    available_module_methods = [
            attr
            for attr in dir(importlib.import_module('pysteps.postprocessing.diagnostics'))
        ]
    assert 'diagnostic_prtype' in available_module_methods

def test_prtype_function():
    """Additionally, you can test that your plugin correctly reads the corresponding
    some example data.
    """
    import pandas as pd
    import numpy as np
    from pysteps.postprocessing.diagnostics import diagnostic_prtype
    # load function with 8 required arguments:
    #    'precipitation_intensity_mmph'
    #    'precipitation_metadata_dict'
    #    'startdate'
    #    'model_snow_level_m'
    #    'model_temperature_degC'
    #    'model_ground_temperature_degC'
    #    'model_metadata_dict'
    #    'topography_data_m'
    #    'topography_metadata_dict'
    
    ### load the test data (artificial)
    startdate = "204002291545"
    
    # use projection and dimension as in pysteps output with RADQPE (RMI) input
    # precipitation data
    precipitation_intensity_mmph = np.ones((24,700,700))*10 # to fix and control the quantity
    # precipitation_intensity_mmph = np.random.random((24,700,700))*10 # add some variability
    projection = "+proj=lcc +lat_1=49.83333333333334 +lat_2=51.16666666666666 +lat_0=50.797815 +lon_0=4.359215833333333 +x_0=649328 +y_0=665262 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs "
    precipitation_metadata_dict = {
        'projection':projection,    
        'xpixelsize': 1000.0,
        'ypixelsize': 1000.0,
        'cartesian_unit':'m',
        'x1': 300000.0,
        'y1': 300000.0,
        'x2': 1000000.0,
        'y2': 1000000.0,
        'yorigin': 'upper',
        'accutime': 5.0,
        'unit': 'mm/h',
        'transform': None,
        'zerovalue': 0.0,
        'threshold': 0.10000015050172806,
        'timestamps':pd.date_range(start=pd.to_datetime(startdate,format='%Y%m%d%H%M')+pd.Timedelta('5min'),
                                   #pysteps first timestamp is the first nowcast of startdate + 5min
                                   periods=24,freq='5min')
        }
    
    # mimick the INCA basic fields transformed to a 3D array with
    # dimension (timestep,x,y)
    # timestep: 13 (analysis + 12h forecast, hourly)
    # x: 600, y: 590
    size_x = 600
    size_y = 590
    # create artifical snow level as array [m]
    model_snow_level_m = np.ones((13,size_x,size_y))*300
    # create artifical temperature field as array [K]
    model_temperature_degC = np.ones((13,size_x,size_y))*280-273.15    
    # create artifical surface temperature field as array [K]
    model_ground_temperature_degC = np.ones((13,size_x,size_y))*270-273.15   
    # Model metadata is defined in pysteps.io.importers
    ## EPSG: 3812 (Belgian Lambert 2008) projection string
    projection='+proj=lcc +lat_1=49.83333333333334 +lat_2=51.16666666666666 +lat_0=50.797815 +lon_0=4.359215833333333 +x_0=649328 +y_0=665262 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs '
    model_metadata_dict = {
        'projection':projection,
        'x1':360000,
        'y1':350000,
        'x2':360000+size_x*1000,
        'y2':350000+size_y*1000,
        'xpixelsize':1000,
        'ypixelsize':1000,
        'cartesian_unit':'m',
        'yorigin':'upper',
        }
    
    # create artifical topo data (or read Belgium data)
    ### need to change function input: from tope filename to tope array (to match other input)
    topography_data_m = np.zeros((size_x,size_y))
    topography_metadata_dict = model_metadata_dict.copy()    
    # test the prtype function
    prtype_list = diagnostic_prtype(precipitation_intensity_mmph,
                          precipitation_metadata_dict,
                          startdate,
                          model_snow_level_m,
                          model_temperature_degC,
                          model_ground_temperature_degC,
                          model_metadata_dict,
                          topography_data_m,
                          topography_metadata_dict)

    # test the shape
    assert prtype_list.shape == precipitation_intensity_mmph.shape
    
    # modify the snow level and temperature data to force rain, snow, freezing rain
    #### check that all pixels are 0 (no precip) or of expected type
    #### 0: no precip, 1: rain, 2: melting snow, 3: snow, 4: freezing rain, 5: hail, 6: severe hail
    # rain everywhere
    model_snow_level_m = np.ones((13,size_x,size_y))*1000.
    model_temperature_degC = np.ones((13,size_x,size_y))*300.-273.15    
    model_ground_temperature_degC = np.ones((13,size_x,size_y))*295.-273.15    
    # test the precip type
    prtype_list = diagnostic_prtype(precipitation_intensity_mmph,
                          precipitation_metadata_dict,
                          startdate,
                          model_snow_level_m,
                          model_temperature_degC,
                          model_ground_temperature_degC,
                          model_metadata_dict,
                          topography_data_m,
                          topography_metadata_dict)
    assert np.all(prtype_list[np.isfinite(prtype_list)] == 1)    
    
    # snow everywhere
    model_snow_level_m = np.ones((13,size_x,size_y))*50.
    model_temperature_degC = np.ones((13,size_x,size_y))*270.-273.15   
    model_ground_temperature_degC = np.ones((13,size_x,size_y))*270.-273.15
    # test the precip type
    prtype_list = diagnostic_prtype(precipitation_intensity_mmph,
                          precipitation_metadata_dict,
                          startdate,
                          model_snow_level_m,
                          model_temperature_degC,
                          model_ground_temperature_degC,
                          model_metadata_dict,
                          topography_data_m+300.,
                          topography_metadata_dict)
    assert np.all(prtype_list[np.isfinite(prtype_list)] == 3)   
    
    # freezing rain everywhere
    model_snow_level_m = np.ones((13,size_x,size_y))*300.
    model_temperature_degC = np.ones((13,size_x,size_y))*275.-273.15 
    model_ground_temperature_degC = np.ones((13,size_x,size_y))*260.-273.15
    # test the precip type
    prtype_list = diagnostic_prtype(precipitation_intensity_mmph,
                          precipitation_metadata_dict,
                          startdate,
                          model_snow_level_m,
                          model_temperature_degC,
                          model_ground_temperature_degC,
                          model_metadata_dict,
                          topography_data_m,
                          topography_metadata_dict)
    assert np.all(prtype_list[np.isfinite(prtype_list)] == 4)
    
    print('-- finished tests --')
    
    
