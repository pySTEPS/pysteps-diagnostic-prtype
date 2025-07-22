# -*- coding: utf-8 -*-
"""
pysteps.postprocessing.diagnostics.diagnostic_prtype
======================
Precipitation Type calculator.

This plugin allows a user to calculate the precipitation types of the hydro-meteors detected in a pysteps blended
nowcast through the use of both the nowcast data and snow level, temperature, and ground temperature data taken from
another weather model, such as INCA or Cosmo.
"""

import os
import numpy as np
import datetime

import matplotlib.pyplot as plt
import matplotlib.pylab as plb
from matplotlib import colors

from pysteps.utils.reprojection import reproject_grids
from pysteps.io import import_netcdf_pysteps
from pysteps.visualization import get_geogrid, get_basemap_axis


def diagnostic_prtype(precipitation_intensity_mmph,
                      precipitation_metadata_dict,
                      startdate,
                      model_snow_level_m,
                      model_temperature_degC,
                      groundmodel_temperature_degC,
                      model_metadata_dict,
                      topography_data_m,
                      topography_metadata_dict,
                      model_timestep_min=None,
                      precipitation_timestep_min=None,
                      **kwargs):
    """
    Calculate the precipitation types for ensemble data at particular time steps from a combination of a pysteps nowcast
    and external model data (such as from INCA or COSMO).

    Parameters
    ----------

    precipitation_intensity_mmph : 2D, 3D or 4D Array
      The precipitation field as observation (2d, [X-coord,Y-coord]), as deterministic nowcast (3D, [time step,X-coord,Y-coord]),
      or as probabilistic nowcast (4D, [member,time step,X-coord,Y-coord])
      
    precipitation_metadata_dict: dict
      A dictionary containing the metadata for the precipitation input.
      Must contain the key "timestamps".
      See pysteps.io.importers for metadata doc.

    startdate : str
      The time and date of the model files in the format "%Y%m%d%H%M"
      e.g. 202305010000 for midnight on the 1st of May 2023.

    model_snow_level_m: 3D Array
      Data should be in the form of a 3D matrix. [time step, X-coord, Y-coord]

    model_temperature_degC: 3D Array
      Data should be in the form of a 3D matrix. [time step, X-coord, Y-coord]

    groundmodel_temperature_degC: 3D Array
      Data should be in the form of a 3D matrix. [time step, X-coord, Y-coord]

    model_metadata_dict: dict
      A dictionary containing the metadata for the snow level, temperature, and ground temperature data.
      See pysteps.io.importers for metadata doc.

    topography_data_m: 2D Array
      Data should be in the form of a 2D matrix. [X-coord, Y-coord]

    topography_metadata_dict: dict
      A dictionary containing the metadata for the topography data.
      See pysteps.io.importers for metadata doc.

    model_timestep_min : int
      The time step between two consecutive model fields as source. (dt in min)

    precipitation_timestep_min : int
      The time step  between two consecutive precipitation nowcast as target. (dt in min)

    {extra_kwargs_doc}

    Returns
    -------
    output:
        A 2D or 3D array containing the precipitation type mask for every pixel
        where rain was observed/predicted.
        (a rainy pixel is a pixel with rain for any member if probabilistic input -> use the mask to apply for individual members afterwards)
        Output arrays take the form [time step, precipitation_intensity_mmph X-coord, precipitation_intensity_mmph Y-coord].
        Or [precipitation_intensity_mmph X-coord, precipitation_intensity_mmph Y-coord] in the case of observation as input.
    """

    # Run checks to ensure correct input parameters
    if not isinstance(precipitation_metadata_dict, dict):
        raise TypeError(
            "precipitation_metadata_dict must be a dictionary containing the metadata about the precipitation field")
    if not "timestamps" in precipitation_metadata_dict.keys():
        raise KeyError("precipitation field metadata must contain the key 'timestamps'")

    if not isinstance(model_metadata_dict, dict):
        raise TypeError(
            "model_metadata_dict must be a dictionary containing the metadata about the snow level, temperature, "
            "and ground temperature files")

    if not isinstance(topography_metadata_dict, dict):
        raise TypeError("topography_metadata_dict must be a dictionary containing metadata about the topography")

    if model_timestep_min is not None and (not isinstance(model_timestep_min, int) or not model_timestep_min > 0):
        raise TypeError("model_timestep_min must be a positive integer")

    if precipitation_timestep_min is not None and (not isinstance(precipitation_timestep_min, int) or not precipitation_timestep_min > 0):
        raise TypeError("precipitation_timestep_min must be a positive integer")

    ####################################################################################

    # Define default parameter values
    if model_timestep_min is None:
        model_timestep_min = 60
    if precipitation_timestep_min is None:
        precipitation_timestep_min = 5
        
    # Convert startdate to datetime
    startdate = datetime.datetime.strptime(startdate, "%Y%m%d%H%M")
    
    # Match precipitation_intensity_mmph dimension to expected shape ([member,time step,X-coord,Y-coord])
    if len(precipitation_intensity_mmph.shape) == 2:
        precipitation_intensity_mmph = precipitation_intensity_mmph[np.newaxis,:]
    if len(precipitation_intensity_mmph.shape) == 3:
        precipitation_intensity_mmph = precipitation_intensity_mmph[np.newaxis,:]

    # ---------------------------------------------------------------------------

    # Reproject
    
    #     topography over model grid
    # # reproject function expects a 3D array -> add a time axis to topography_data_m
    # topo_grid, _ = reproject_grids(topography_data_m[np.newaxis, :], model_snow_level_m[0, :, :], topography_metadata_dict, model_metadata_dict)
    # topo_grid = topo_grid[0]#back to 2D array
    # print('Re-projection of topography grid done')
    #     projection over pySTEPS grid
    model_snow_level_m, meta = reproject_grids(model_snow_level_m, precipitation_intensity_mmph[0, 0, :, :], model_metadata_dict, precipitation_metadata_dict)
    model_temperature_degC, _ = reproject_grids(model_temperature_degC, precipitation_intensity_mmph[0, 0, :, :], model_metadata_dict, precipitation_metadata_dict)
    groundmodel_temperature_degC, _ = reproject_grids(groundmodel_temperature_degC, precipitation_intensity_mmph[0, 0, :, :], model_metadata_dict, precipitation_metadata_dict)
    topo_grid, _ = reproject_grids(topography_data_m[np.newaxis, :], model_snow_level_m[0, :, :], topography_metadata_dict, meta)
    topo_grid = topo_grid[0]#back to 2D array
    print('Re-projection on precip grid done')
    # --------------------------------------------------------------------------

    # Calculate temporal interpolation matrices

    # Calculate temporal interpolations values for matching timestamps between model and pySTEPS
    interpolations_ZS, timestamps_idxs = generate_interpolations(model_snow_level_m, precipitation_metadata_dict['timestamps'],
                                                                 startdate, precipitation_timestep_min, model_timestep_min)
    interpolations_TT, _ = generate_interpolations(model_temperature_degC, precipitation_metadata_dict['timestamps'], startdate, precipitation_timestep_min,
                                                   model_timestep_min)
    interpolations_TG, _ = generate_interpolations(groundmodel_temperature_degC, precipitation_metadata_dict['timestamps'], startdate, precipitation_timestep_min,
                                                   model_timestep_min)
    print("Interpolation in time done!")

    # Clean (After interpolation, we don't need the reprojected data anymore)
    del model_snow_level_m, model_temperature_degC, groundmodel_temperature_degC, topography_data_m

    # --------------------------------------------------------------------------

    # Diagnose precipitation type per member over time, using mean mask

    # WARNING (1): The grids have been sub-scripted to the model size. This requires the model metadata to
    # be used for plotting. If the original PYSTEPS grid size is used (700x700) for plotting, the pysteps precipitation_metadata_dict
    # should be used instead.
    #
    # WARNING (2): Topography does not need to be re-projected if it matches the grid size of the model.

    print("Calculate precipitation type as mask for all members over time...")

    ######
    # I think the next line is what we'd like to avoid - we do not want to use the indexes
    # but rather return NaNs for pixel where one or the other grid does not exist
    ########
    # # Find subscript indexes for model grid
    # x1, x2, y1, y2 = get_reprojected_indexes(interpolations_ZS[0])

    # Result list
    # ptype_list = np.zeros((precipitation_intensity_mmph.shape[1], x2 - x1, y2 - y1))
    ptype_list = np.zeros((precipitation_intensity_mmph.shape[1], precipitation_intensity_mmph.shape[2], precipitation_intensity_mmph.shape[3]))

    # loop over timestamps
    for ts in range(len(timestamps_idxs)):
        print("Calculating precipitation types at: ", str(timestamps_idxs[ts]))
        
        # Members mean for this timestamp
        # precipitation_intensity_mmph_mean = np.mean(precipitation_intensity_mmph[:, ts, x1:x2, y1:y2],axis=0)
        precipitation_intensity_mmph_mean = np.mean(precipitation_intensity_mmph[:, ts, :, :],axis=0)
        
        # Calculate precipitation type result with members mean
        # ptype_mean = calculate_precip_type(Znow=interpolations_ZS[ts, x1:x2, y1:y2],
        #                                    Temp=interpolations_TT[ts, x1:x2, y1:y2],
        #                                    GroundTemp=interpolations_TG[ts, x1:x2, y1:y2],
        #                                    precipGrid=precipitation_intensity_mmph_mean,
        #                                    topographyGrid=topo_grid[x1:x2, y1:y2])
        ptype_mean = calculate_precip_type(Znow=interpolations_ZS[ts],
                                           Temp=interpolations_TT[ts],
                                           GroundTemp=interpolations_TG[ts],
                                           precipGrid=precipitation_intensity_mmph_mean,
                                           topographyGrid=topo_grid)
        
        # Add mean result to output
        ptype_list[ts, :, :] = ptype_mean

    print("--Script finished--")
    return ptype_list


def plot_precipType_field(
        precipType,
        ax=None,
        geodata=None,
        bbox=None,
        colorscale="pysteps",
        title=None,
        colorbar=True,
        cBarLabel="",
        categoryNr=4,
        axis="on",
        cax=None,
        map_kwargs=None,
):
    """
    Function to plot a precipitation types field with a colorbar.

    .. _Axes: https://matplotlib.org/api/axes_api.html#matplotlib.axes.Axes

    .. _SubplotSpec: https://matplotlib.org/api/_as_gen/matplotlib.gridspec.SubplotSpec.html

    Parameters
    ----------
    precipType: array-like
        Two-dimensional array containing the input precipitation types.
    ax: fig Axes_
        Axes for the basemap.
    geodata: dictionary or None, optional
        Optional dictionary containing geographical information about
        the field. Required is map is not None.

        If geodata is not None, it must contain the following key-value pairs:

        .. tabularcolumns:: |p{1.5cm}|L|

        +-----------------+---------------------------------------------------+
        |        Key      |                  Value                            |
        +=================+===================================================+
        |    projection   | PROJ.4-compatible projection definition           |
        +-----------------+---------------------------------------------------+
        |    x1           | x-coordinate of the lower-left corner of the data |
        |                 | raster                                            |
        +-----------------+---------------------------------------------------+
        |    y1           | y-coordinate of the lower-left corner of the data |
        |                 | raster                                            |
        +-----------------+---------------------------------------------------+
        |    x2           | x-coordinate of the upper-right corner of the     |
        |                 | data raster                                       |
        +-----------------+---------------------------------------------------+
        |    y2           | y-coordinate of the upper-right corner of the     |
        |                 | data raster                                       |
        +-----------------+---------------------------------------------------+
        |    yorigin      | a string specifying the location of the first     |
        |                 | element in the data raster w.r.t. y-axis:         |
        |                 | 'upper' = upper border, 'lower' = lower border    |
        +-----------------+---------------------------------------------------+
    bbox : tuple, optional
        Four-element tuple specifying the coordinates of the bounding box. Use
        this for plotting a subdomain inside the input grid. The coordinates are
        of the form (lower left x, lower left y ,upper right x, upper right y).
        If 'geodata' is not None, the bbox is in map coordinates, otherwise
        it represents image pixels.
    colorscale : {'pysteps', 'STEPS-BE', 'STEPS-NL', 'BOM-RF3'}, optional
        Which colorscale to use. TO BE DEFINED
    title : str, optional
        If not None, print the title on top of the plot.
    colorbar : bool, optional
        If set to True, add a colorbar on the right side of the plot.
    cBarLabel :
        Set color bar label.
    categoryNr :
        Number of categories to be plotted (2 to 6)
    axis : {'off','on'}, optional
        Whether to turn off or on the x and y axis.
    cax : Axes_ object, optional
        Axes into which the colorbar will be drawn. If no axes is provided
        the colorbar axes are created next to the plot.

    Other parameters
    ----------------
    map_kwargs: dict
        Optional parameters that need to be passed to
        :py:func:`pysteps.visualization.basemaps.plot_geography`.

    Returns
    -------
    ax : fig Axes_
        Figure axes. Needed if one wants to add e.g. text inside the plot.
    """

    if map_kwargs is None:
        map_kwargs = {}

    if len(precipType.shape) != 2:
        raise ValueError("The input is not two-dimensional array")

    # Assumes the input dimensions are lat/lon
    nlat, nlon = precipType.shape

    x_grid, y_grid, extent, regular_grid, origin = get_geogrid(
        nlat, nlon, geodata=geodata
    )

    ax = get_basemap_axis(extent, ax=ax, geodata=geodata, map_kwargs=map_kwargs)

    precipType = np.ma.masked_invalid(precipType)
    # plot rainfield
    if regular_grid:
        im = _plot_field(precipType, ax, colorscale, categoryNr, extent, origin=origin)
    else:
        im = _plot_field(
            precipType, ax, colorscale, categoryNr, extent, x_grid=x_grid, y_grid=y_grid
        )

    plb.title(title, loc='center', fontsize=25)

    # add colorbar
    cbar = None
    if colorbar:
        # get colormap and color levels
        _, _, clevs, clevs_str = get_colormap(colorscale, categoryNr)
        cbar = plb.colorbar(
            im, ticks=clevs, spacing="uniform", extend="neither", shrink=0.8, cax=cax, drawedges=False
        )
        if clevs_str is not None:
            cbar.ax.set_yticklabels('')
            cbar.ax.tick_params(size=0)
            cbar.ax.set_yticks([i + .5 for i in clevs][:-1], minor=True)
            cbar.ax.set_yticklabels(clevs_str[:-1], minor=True, fontsize=15)
    cbar.set_label(cBarLabel)

    if geodata is None or axis == "off":
        ax.xaxis.set_ticks([])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticks([])
        ax.yaxis.set_ticklabels([])

    if bbox is not None:
        ax.set_xlim(bbox[0], bbox[2])
        ax.set_ylim(bbox[1], bbox[3])

    return ax


def _plot_field(precipType, ax, colorscale, categoryNr, extent, origin=None, x_grid=None, y_grid=None):
    precipType = precipType.copy()

    # Get colormap and color levels
    cmap, norm, _, _ = get_colormap(colorscale, categoryNr)

    if (x_grid is None) or (y_grid is None):
        im = ax.imshow(
            precipType,
            cmap=cmap,
            norm=norm,
            extent=extent,
            interpolation="nearest",
            origin=origin,
            zorder=10,
        )
    else:
        im = ax.pcolormesh(
            x_grid,
            y_grid,
            precipType,
            cmap=cmap,
            norm=norm,
            zorder=10,
        )

    return im


def get_colormap(colorscale="pysteps", categoryNr=4):
    """
    Function to generate a colormap (cmap) and norm.

    Parameters
    ----------

    colorscale : {'pysteps', 'STEPS-BE', 'STEPS-NL', 'BOM-RF3'}, optional
      Which colorscale to use. Applicable if units is 'mm/h', 'mm' or 'dBZ'.

    Returns
    -------
    cmap : Colormap instance
      colormap
    norm : colors.Normalize object
      Colors norm
    clevs: list(float)
      List of precipitation values defining the color limits.
    clevs_str: list(str)
      List of precipitation values defining the color limits (with correct
      number of decimals).
      :param categoryNr:
    """
    # Get list of colors
    color_list, clevs, clevs_str = _get_colorlist(colorscale, categoryNr)
    cmap = colors.LinearSegmentedColormap.from_list(
        "cmap", color_list, len(clevs) - 1
    )
    cmap.set_over("darkred", 1)
    cmap.set_bad("gray", alpha=0.5)
    cmap.set_under("none")
    norm = colors.BoundaryNorm(clevs, cmap.N)

    return cmap, norm, clevs, clevs_str


def _get_colorlist(colorscale="pysteps", categoryNr=4):
    """
    Function to get a list of colors to generate the colormap.

    Parameters
    ----------
    colorscale : str
        Which colorscale to use (BOM-RF3, pysteps, STEPS-BE, STEPS-NL)
    categoryNr  :
        How many categories should be plotted

    Returns
    -------
    color_list : list(str)
        List of color strings.

    clevs : list(float)
        List of precipitation values defining the color limits.

    clevs_str : list(str)
        List of precipitation type names
    """

    if categoryNr < 1 or categoryNr > 6:
        raise ValueError("Invalid category index [1 to 6] " + str(categoryNr))

    if colorscale == "pysteps":
        color_list = ["#ffe38f", "#ceda86", "#009489", "#3897ed", "#b0a0dc", "#ec623b"]
    # elif colorscale == 'other color scale': ... [6 colors]
    else:
        print("Invalid colorscale", colorscale)
        raise ValueError("Invalid colorscale " + colorscale)

    # Ticks and labels
    clevs = [1, 2, 3, 4, 5, 6, 7]
    clevs_str = ['Rain', 'Wet Snow', 'Snow', 'Freezing Rain', 'Hail', 'Severe Hail']

    # filter by category number
    color_list = color_list[0:categoryNr]
    clevs = clevs[0:(categoryNr + 1)]
    clevs_str = clevs_str[0:categoryNr] + ['']

    return color_list, clevs, clevs_str


def plot_ptype(ptype_grid, metadata, i, date_time, dir_gif, categoryNr=4):
    title = 'Precipitation type ' + date_time.strftime("%Y-%m-%d %H:%M")
    fig = plt.figure(figsize=(15, 15))
    # fig.add_subplot(1, 1, 1)
    plot_precipType_field(ptype_grid, geodata=metadata, title=title, colorscale="pysteps", categoryNr=categoryNr)
    # plt.suptitle('Precipitation Type', fontsize=30)
    plt.tight_layout()
    filename = f'{i}.png'
    #  filenames.append(filename)
    plt.savefig(os.path.join(dir_gif, filename), dpi=72)
    plt.close()
    return filename


def calculate_precip_type(snow_level_grid_m, temperature_grid_degC, ground_temperature_grid_degCerature_grid_degC, precipitation_intensity_grid_mmph, topography_grid_m, melting_layer_thickness_m=100., freezing_rain_2m_temperature_with_frozen_ground_degC=2., freezing_rain_temperature_threshold_degC=0.,
                          minimum_precipitation_threshold_mmph=0):
    """Precipitation type algorithm, returns a 2D matrix with categorical values:
    # PT=0  no precip
    # PT=1  rain
    # PT=2  rain/snow mix
    # PT=3  snow
    # PT=4  freezing rain

    snow_level_grid_m:
        snow level 2D grid
    temperature_grid_degC:
        2m temperature_grid_degCerature 2D grid
    ground_temperature_grid_degCerature_grid_degC:
        ground temperature_grid_degCerature 2D grid
    precipitation_intensity_grid_mmph:
        Precipitation (netCDF PYSTEPS) 2D grid
    topography_grid_m:
        Topography grid 2D
    melting_layer_thickness_m:
        thickness of the melting layer (default 100m)
    freezing_rain_2m_temperature_with_frozen_ground_degC:
        2m temperature_grid_degCerature threshold below which rain will freeze when it hits a freezing ground (default 2C)
    freezing_rain_temperature_threshold_degC:
        temperature_grid_degCerature threshold for freezing rain, either for the 2m temperature_grid_degCerature or for the ground temperature_grid_degCerature
    minimum_precipitation_threshold_mmph:
        minimum precipitation threshold (default 0mm/h)

    returns:
        2D matrix with categorical data for each type
    """

    # Result grid
    result = np.zeros((precipitation_intensity_grid_mmph.shape[0], precipitation_intensity_grid_mmph.shape[1]))
    topoZSDiffGrid = (snow_level_grid_m - topography_grid_m)  # dzs -> higher means we are lower wrt the snow level
    precipMask = (precipitation_intensity_grid_mmph > minimum_precipitation_threshold_mmph)

    # SNOW ((dzs<-1.5*melting_layer_thickness_m) || ( (ZH[i][j] <= 1.5*melting_layer_thickness_m) && (dzs<=0)))
    snowMask = (topoZSDiffGrid < (-1.5 * melting_layer_thickness_m)) | ((topography_grid_m <= (1.5 * melting_layer_thickness_m)) & (topoZSDiffGrid <= 0))
    result[snowMask & precipMask] = 3

    # RAIN+SNOW DIAGNOSIS (dzs < 0.5 * melting_layer_thickness_m) = 2
    rainSnowMask = ~snowMask & (topoZSDiffGrid < (0.5 * melting_layer_thickness_m))
    result[rainSnowMask & precipMask] = 2

    # RAIN
    rainMask = ~snowMask & ~rainSnowMask
    result[rainMask & precipMask] = 1

    # FREEZING RAIN DIAGNOSIS 4
    # if ((PT[i][j]==1) && ( (tg_<freezing_rain_temperature_threshold_degC && TT[i][j]<freezing_rain_2m_temperature_with_frozen_ground_degC) || TT[i][j]<freezing_rain_temperature_threshold_degC))
    freezingMask = (result == 1) & (((ground_temperature_grid_degCerature_grid_degC < freezing_rain_temperature_threshold_degC) & (temperature_grid_degC < freezing_rain_2m_temperature_with_frozen_ground_degC)) | (temperature_grid_degC < freezing_rain_temperature_threshold_degC))
    result[freezingMask] = 4

    return result


def get_reprojected_indexes(reprojectedGrid):
    """Reprojected model grids contains a frame of NAN values, this function returns the start and end indexes
    of the model grid over the reprojected grid

    reprojectedGrid:
        model reprojected Grid

    ---
    Returns:
        x y indexes of model reprojected grid over pysteps dimensions
    """

    x_start = np.where(~np.isnan(reprojectedGrid))[0][0]
    x_end = np.where(~np.isnan(reprojectedGrid))[0][-1] + 1
    y_start = np.where(~np.isnan(reprojectedGrid))[-1][0]
    y_end = np.where(~np.isnan(reprojectedGrid))[-1][-1] + 1

    return x_start, x_end, y_start, y_end


def grid_interpolation(numpyGridStart, numpyGridEnd, interpolation_timestep_min=5, input_timestep_min=60):
    """ Time interpolation between 2 2D grids

    numpyGridStart:
        Numpy 2-D grid of start values
    numpyGridEnd:
        Numpy 2-D grid of end values
    interpolation_timestep_min:
        Time step for interpolation target grid in minutes
    input_timestep_min:
        Time of input grid in minutes
    applyOver:
        Array with sub-indexes to calculate interpolation (inner grid)
    ----

    Return:
        Returns a list of 3D numpy interpolation matrix
    """
    if numpyGridStart.shape != numpyGridEnd.shape:
        raise ValueError("ERROR: Grids have different dimensions")

    interPoints = np.arange(0, (input_timestep_min + interpolation_timestep_min), interpolation_timestep_min)
    interpolationGrid = np.zeros((len(interPoints), numpyGridStart.shape[0], numpyGridStart.shape[1]))
    interpolationGrid[:, :, :] = np.nan

    # print('Calculating linear interpolation..', end=' ')
    for i in range(len(interPoints)):
        interpolationGrid[i, :, :] = numpyGridStart + ((numpyGridEnd - numpyGridStart) / interPoints[-1]) * interPoints[
            i]
    # print('Done')

    return interpolationGrid


def create_timestamp_indexing(nrOfModelMessages, startDateTime, interpolation_timestep_min=5, model_timestep_min=60):
    """create a timestamp array for model indexing

    nrOfModelMessages:
        Number of model available messages

    startDateTime:
        Start date and time

    interpolation_timestep_min:
        Time step for interpolation in minutes

    model_timestep_min:
        Time step of model input in minutes

    ___
    Return:
          Array of timestamps similar to pysteps timestamps
    """

    if nrOfModelMessages < 2:
        raise ValueError("Not enough interpolation messages, should be at least 2")

    result = []
    interPoints = np.arange(0, (model_timestep_min + interpolation_timestep_min), interpolation_timestep_min)

    for i in range(nrOfModelMessages - 1):
        for j in interPoints[:-1]:
            result.append(startDateTime)
            startDateTime = startDateTime + datetime.timedelta(minutes=interpolation_timestep_min)

    result.append(startDateTime)
    return np.array(result)


def generate_interpolations(model_reprojected_data, nwc_timestamps, startdate, interpolation_timestep_min=5, model_timestep_min=60,
                            dateFormat='%Y%m%d%H%M'):
    """Generate a sub-selection of the interpolation matrix for all messages available from model data

    model_reprojected_data:
        model reprojected data.

    nwc_timestamps:
        Timestamps of the precipitation field.
        E.g., array of timestamps available from PYSTEPS metadata ['timestamps']
    
    startdate:
        datetime object
        
    interpolation_timestep_min:
        Time step for interpolation in minutes

    model_timestep_min:
        Time step of model input in minutes
        
    dateFormat:
        use any datetime format string (used to ensure consient times, not related to data input)

    ----
    Return:
        3D matrix with depth equal to the common matching timestamps between the model and PYSTEPS.

    """

    # Create a timestamp index array for model interpolation matrix
    model_timestamps = create_timestamp_indexing(model_reprojected_data.shape[0], startdate, interpolation_timestep_min=interpolation_timestep_min,
                                                 model_timestep_min=model_timestep_min)
    # Convert precipitation_metadata_dict['timestamps'] to datetime
    nwc_ts = [datetime.datetime.strptime(ts.strftime(dateFormat), dateFormat) for ts in nwc_timestamps]

    model_start = np.where(model_timestamps == nwc_ts[0])[0][0]
    model_end = np.where(model_timestamps == nwc_ts[-1])[0][0] + 1
    timestamp_selection = model_timestamps[model_start:model_end]  # to be returned

    # interpolation indexes
    resultMatrix = np.zeros(
        (model_start + len(timestamp_selection), model_reprojected_data.shape[1], model_reprojected_data.shape[2]))
    result_idx = 0

    # loop over the messages
    for m in range(1, model_reprojected_data.shape[0]):
        if result_idx < resultMatrix.shape[0]:
            # calculate interpolations
            interpolationMatrix = grid_interpolation(model_reprojected_data[m - 1], model_reprojected_data[m],
                                                     interpolation_timestep_min=interpolation_timestep_min, model_timestep_min=model_timestep_min)
            interp_idx = 0
            # Add the interpolation values to the result matrix (this assignment can be done without looping...)
            while interp_idx < interpolationMatrix.shape[0] and (result_idx < resultMatrix.shape[0]):
                resultMatrix[result_idx, :, :] = interpolationMatrix[interp_idx, :, :]
                result_idx = result_idx + 1
                interp_idx = interp_idx + 1
            result_idx = result_idx - 1  # overwrite the last value

            return resultMatrix[model_start:], timestamp_selection