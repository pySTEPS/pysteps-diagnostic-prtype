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


def diagnostic_prtype(precip_field,
                      precipMetadataDictionary,
                      startdate,
                      snowLevelData,
                      temperatureData,
                      groundTemperatureData,
                      modelMetadataDictionary,
                      topographyData,
                      topoMetadataDictionary,
                      src_timestep=None,
                      trgt_timestep=None,
                      **kwargs):
    """
    Calculate the precipitation types for ensemble data at particular time steps from a combination of a pysteps nowcast
    and external model data (such as from INCA or COSMO).

    Parameters
    ----------

    precip_field : 2D, 3D or 4D Array
      The precipitation field as observation (2d, [X-coord,Y-coord]), as deterministic nowcast (3D, [time step,X-coord,Y-coord]),
      or as probabilistic nowcast (4D, [member,time step,X-coord,Y-coord])
      
    precipMetadataDictionary: dict
      A dictionary containing the metadata for the precipitation input.
      Must contain the key "timestamps".
      See pysteps.io.importers for metadata doc.

    startdate : str
      The time and date of the model files in the format "%Y%m%d%H%M"
      e.g. 202305010000 for midnight on the 1st of May 2023.

    snowLevelData: 3D Array
      Data should be in the form of a 3D matrix. [time step, X-coord, Y-coord]

    temperatureData: 3D Array
      Data should be in the form of a 3D matrix. [time step, X-coord, Y-coord]

    groundTemperatureData: 3D Array
      Data should be in the form of a 3D matrix. [time step, X-coord, Y-coord]

    modelMetadataDictionary: dict
      A dictionary containing the metadata for the snow level, temperature, and ground temperature data.
      See pysteps.io.importers for metadata doc.

    topographyData: 2D Array
      Data should be in the form of a 2D matrix. [X-coord, Y-coord]

    topoMetadataDictionary: dict
      A dictionary containing the metadata for the topography data.
      See pysteps.io.importers for metadata doc.

    src_timestep : int
      The time step between two consecutive model fields as source. (dt in min)

    trgt_timestep : int
      The time step  between two consecutive precipitation nowcast as target. (dt in min)

    {extra_kwargs_doc}

    Returns
    -------
    output:
        A 2D or 3D array containing the precipitation type mask for every pixel
        where rain was observed/predicted.
        (a rainy pixel is a pixel with rain for any member if probabilistic input -> use the mask to apply for individual members afterwards)
        Output arrays take the form [time step, precip_field X-coord, precip_field Y-coord].
        Or [precip_field X-coord, precip_field Y-coord] in the case of observation as input.
    """

    # Run checks to ensure correct input parameters
    if not isinstance(precipMetadataDictionary, dict):
        raise TypeError(
            "precipMetadataDictionary must be a dictionary containing the metadata about the precipitation field")
    if not "timestamps" in precipMetadataDictionary.keys():
        raise KeyError("precipitation field metadata must contain the key 'timestamps'")

    if not isinstance(modelMetadataDictionary, dict):
        raise TypeError(
            "modelMetadataDictionary must be a dictionary containing the metadata about the snow level, temperature, "
            "and ground temperature files")

    if not isinstance(topoMetadataDictionary, dict):
        raise TypeError("topoMetadataDictionary must be a dictionary containing metadata about the topography")

    if src_timestep is not None and (not isinstance(src_timestep, int) or not src_timestep > 0):
        raise TypeError("src_timestep must be a positive integer")

    if trgt_timestep is not None and (not isinstance(trgt_timestep, int) or not trgt_timestep > 0):
        raise TypeError("trgt_timestep must be a positive integer")

    ####################################################################################

    # Define default parameter values
    if src_timestep is None:
        src_timestep = 60
    if trgt_timestep is None:
        trgt_timestep = 5
        
    # Convert startdate to datetime
    startdate = datetime.datetime.strptime(startdate, "%Y%m%d%H%M")
    
    # Match precip_field dimension to expected shape ([member,time step,X-coord,Y-coord])
    if len(precip_field.shape) == 2:
        precip_field = precip_field[np.newaxis,:]
    if len(precip_field.shape) == 3:
        precip_field = precip_field[np.newaxis,:]

    # ---------------------------------------------------------------------------

    # Reproject
    
    #     topography over model grid
    # # reproject function expects a 3D array -> add a time axis to topographyData
    # topo_grid, _ = reproject_grids(topographyData[np.newaxis, :], snowLevelData[0, :, :], topoMetadataDictionary, modelMetadataDictionary)
    # topo_grid = topo_grid[0]#back to 2D array
    # print('Re-projection of topography grid done')
    #     projection over pySTEPS grid
    snowLevelData, meta = reproject_grids(snowLevelData, precip_field[0, 0, :, :], modelMetadataDictionary, precipMetadataDictionary)
    temperatureData, _ = reproject_grids(temperatureData, precip_field[0, 0, :, :], modelMetadataDictionary, precipMetadataDictionary)
    groundTemperatureData, _ = reproject_grids(groundTemperatureData, precip_field[0, 0, :, :], modelMetadataDictionary, precipMetadataDictionary)
    topo_grid, _ = reproject_grids(topographyData[np.newaxis, :], snowLevelData[0, :, :], topoMetadataDictionary, meta)
    topo_grid = topo_grid[0]#back to 2D array
    print('Re-projection on precip grid done')
    # --------------------------------------------------------------------------

    # Calculate temporal interpolation matrices

    # Calculate temporal interpolations values for matching timestamps between model and pySTEPS
    interpolations_ZS, timestamps_idxs = generate_interpolations(snowLevelData, precipMetadataDictionary['timestamps'],
                                                                 startdate, trgt_timestep, src_timestep)
    interpolations_TT, _ = generate_interpolations(temperatureData, precipMetadataDictionary['timestamps'], startdate, trgt_timestep,
                                                   src_timestep)
    interpolations_TG, _ = generate_interpolations(groundTemperatureData, precipMetadataDictionary['timestamps'], startdate, trgt_timestep,
                                                   src_timestep)
    print("Interpolation in time done!")

    # Clean (After interpolation, we don't need the reprojected data anymore)
    del snowLevelData, temperatureData, groundTemperatureData, topographyData

    # --------------------------------------------------------------------------

    # Diagnose precipitation type per member over time, using mean mask

    # WARNING (1): The grids have been sub-scripted to the model size. This requires the model metadata to
    # be used for plotting. If the original PYSTEPS grid size is used (700x700) for plotting, the pysteps precipMetadataDictionary
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
    # ptype_list = np.zeros((precip_field.shape[1], x2 - x1, y2 - y1))
    ptype_list = np.zeros((precip_field.shape[1], precip_field.shape[2], precip_field.shape[3]))

    # loop over timestamps
    for ts in range(len(timestamps_idxs)):
        print("Calculating precipitation types at: ", str(timestamps_idxs[ts]))
        
        # Members mean for this timestamp
        # precip_field_mean = np.mean(precip_field[:, ts, x1:x2, y1:y2],axis=0)
        precip_field_mean = np.mean(precip_field[:, ts, :, :],axis=0)
        
        # Calculate precipitation type result with members mean
        # ptype_mean = calculate_precip_type(Znow=interpolations_ZS[ts, x1:x2, y1:y2],
        #                                    Temp=interpolations_TT[ts, x1:x2, y1:y2],
        #                                    GroundTemp=interpolations_TG[ts, x1:x2, y1:y2],
        #                                    precipGrid=precip_field_mean,
        #                                    topographyGrid=topo_grid[x1:x2, y1:y2])
        ptype_mean = calculate_precip_type(Znow=interpolations_ZS[ts],
                                           Temp=interpolations_TT[ts],
                                           GroundTemp=interpolations_TG[ts],
                                           precipGrid=precip_field_mean,
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


def calculate_precip_type(Znow, Temp, GroundTemp, precipGrid, topographyGrid, DZML=100., TT0=2., TG0=0.,
                          RRMIN=0):
    """Precipitation type algorithm, returns a 2D matrix with categorical values:
    # PT=0  no precip
    # PT=1  rain
    # PT=2  rain/snow mix
    # PT=3  snow
    # PT=4  freezing rain

    Znow:
        snow level 2D grid
    Temp:
        2m temperature 2D grid
    GroundTemp:
        ground temperature 2D grid
    precipGrid:
        Precipitation (netCDF PYSTEPS) 2D grid
    topographyGrid:
        Topography grid 2D
    DZML:
        thickness of the melting layer (default 100m)
    TT0:
        2m temperature threshold below which rain will freeze when it hits a freezing ground (default 2C)
    TG0:
        temperature threshold for freezing rain, either for the 2m temperature or for the ground temperature
    RRMIN:
        minimum precipitation threshold (default 0mm/h)

    returns:
        2D matrix with categorical data for each type
    """

    # Result grid
    result = np.zeros((precipGrid.shape[0], precipGrid.shape[1]))
    topoZSDiffGrid = (Znow - topographyGrid)  # dzs -> higher means we are lower wrt the snow level
    precipMask = (precipGrid > RRMIN)

    # SNOW ((dzs<-1.5*DZML) || ( (ZH[i][j] <= 1.5*DZML) && (dzs<=0)))
    snowMask = (topoZSDiffGrid < (-1.5 * DZML)) | ((topographyGrid <= (1.5 * DZML)) & (topoZSDiffGrid <= 0))
    result[snowMask & precipMask] = 3

    # RAIN+SNOW DIAGNOSIS (dzs < 0.5 * DZML) = 2
    rainSnowMask = ~snowMask & (topoZSDiffGrid < (0.5 * DZML))
    result[rainSnowMask & precipMask] = 2

    # RAIN
    rainMask = ~snowMask & ~rainSnowMask
    result[rainMask & precipMask] = 1

    # FREEZING RAIN DIAGNOSIS 4
    # if ((PT[i][j]==1) && ( (tg_<TG0 && TT[i][j]<TT0) || TT[i][j]<TG0))
    freezingMask = (result == 1) & (((GroundTemp < TG0) & (Temp < TT0)) | (Temp < TG0))
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


def grid_interpolation(numpyGridStart, numpyGridEnd, trgt_timestep=5, src_timestep=60):
    """ Time interpolation between 2 2D grids

    numpyGridStart:
        Numpy 2-D grid of start values
    numpyGridEnd:
        Numpy 2-D grid of end values
    trgt_timestep:
        Time step for interpolation target grid in minutes
    src_timestep:
        Time of input grid in minutes
    applyOver:
        Array with sub-indexes to calculate interpolation (inner grid)
    ----

    Return:
        Returns a list of 3D numpy interpolation matrix
    """
    if numpyGridStart.shape != numpyGridEnd.shape:
        raise ValueError("ERROR: Grids have different dimensions")

    interPoints = np.arange(0, (src_timestep + trgt_timestep), trgt_timestep)
    interpolationGrid = np.zeros((len(interPoints), numpyGridStart.shape[0], numpyGridStart.shape[1]))
    interpolationGrid[:, :, :] = np.nan

    # print('Calculating linear interpolation..', end=' ')
    for i in range(len(interPoints)):
        interpolationGrid[i, :, :] = numpyGridStart + ((numpyGridEnd - numpyGridStart) / interPoints[-1]) * interPoints[
            i]
    # print('Done')

    return interpolationGrid


def create_timestamp_indexing(nrOfModelMessages, startDateTime, trgt_timestep=5, src_timestep=60):
    """create a timestamp array for model indexing

    nrOfModelMessages:
        Number of model available messages

    startDateTime:
        Start date and time

    trgt_timestep:
        Time step for interpolation in minutes

    src_timestep:
        Time step of model input in minutes

    ___
    Return:
          Array of timestamps similar to pysteps timestamps
    """

    if nrOfModelMessages < 2:
        raise ValueError("Not enough interpolation messages, should be at least 2")

    result = []
    interPoints = np.arange(0, (src_timestep + trgt_timestep), trgt_timestep)

    for i in range(nrOfModelMessages - 1):
        for j in interPoints[:-1]:
            result.append(startDateTime)
            startDateTime = startDateTime + datetime.timedelta(minutes=trgt_timestep)

    result.append(startDateTime)
    return np.array(result)


def generate_interpolations(model_reprojected_data, nwc_timestamps, startdate, trgt_timestep=5, src_timestep=60,
                            dateFormat='%Y%m%d%H%M'):
    """Generate a sub-selection of the interpolation matrix for all messages available from model data

    model_reprojected_data:
        model reprojected data.

    nwc_timestamps:
        Timestamps of the precipitation field.
        E.g., array of timestamps available from PYSTEPS metadata ['timestamps']
    
    startdate:
        datetime object
        
    trgt_timestep:
        Time step for interpolation in minutes

    src_timestep:
        Time step of model input in minutes
        
    dateFormat:
        use any datetime format string (used to ensure consient times, not related to data input)

    ----
    Return:
        3D matrix with depth equal to the common matching timestamps between the model and PYSTEPS.

    """

    # Create a timestamp index array for model interpolation matrix
    model_timestamps = create_timestamp_indexing(model_reprojected_data.shape[0], startdate, trgt_timestep=trgt_timestep,
                                                 src_timestep=src_timestep)
    # Convert precipMetadataDictionary['timestamps'] to datetime
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
                                                     trgt_timestep=trgt_timestep, src_timestep=src_timestep)
            interp_idx = 0
            # Add the interpolation values to the result matrix (this assignment can be done without looping...)
            while interp_idx < interpolationMatrix.shape[0] and (result_idx < resultMatrix.shape[0]):
                resultMatrix[result_idx, :, :] = interpolationMatrix[interp_idx, :, :]
                result_idx = result_idx + 1
                interp_idx = interp_idx + 1
            result_idx = result_idx - 1  # overwrite the last value

            return resultMatrix[model_start:], timestamp_selection