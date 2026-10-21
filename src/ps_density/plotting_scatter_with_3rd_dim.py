"""
This script contains functions that allow to plot scatter plots with a third dimension
"""

# Import modules
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.figure
from matplotlib.colors import LinearSegmentedColormap

from typing import Union
import warnings


def using_mpl_scatter_density(
    fig: matplotlib.figure.Figure,
    x: np.ndarray,
    y: np.ndarray,
    vmax: Union[float, None] = None,
    dpi: int = 75,
) -> Union[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    """
    The following function generates scatter plots
    that have denisty of points as a third dim
    Source of the function:
    https://stackoverflow.com/questions/20105364/how-can-i-make-a-scatter-plot-colored-by-density-in-matplotlib

    Arguments:
        fig (matplotlib.figure.Figure): figure
        x (np.ndarray): x data
        y (np.ndarray): y data
        vmax (Union[float, None]): the max value on the colour bar
        dpi (int): dpi

    Returns:
        Union[matplotlib.figure.Figure, matplotlib.axes.Axes]: figure and axes
    """

    # Function for plotted points density calculation
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")

    # Silence a warning that is a known bug in the external function used
    # https://github.com/astrofrog/mpl-scatter-density/issues/35
    # https://github.com/glue-viz/glue/issues/2027
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore", message="All-NaN slice encountered")
        density = ax.scatter_density(x, y, cmap=white_viridis, vmax=vmax, dpi=dpi)

    fig.colorbar(density, label="Number of points per pixel on the plot")

    return fig, ax


# "Viridis-like" colormap with white background
white_viridis = LinearSegmentedColormap.from_list(
    "white_viridis",
    [
        (0, "#ffffff"),
        (1e-20, "#440053"),
        (0.2, "#404388"),
        (0.4, "#2a788e"),
        (0.6, "#21a784"),
        (0.8, "#78d151"),
        (1, "#fde624"),
    ],
    N=256,
)


def plot_scatter_denisty(
    x: np.ndarray,
    y: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    x_limit: Union[float, None] = None,
    y_limit: Union[float, None] = None,
    vmax: Union[float, None] = None,
    x_limit_min: Union[float, None] = None,
    y_limit_min: Union[float, None] = None,
    dpi: int = 75,
    fig_size: list = [15, 7],
):
    """
    The following function plots scatter plots with density of points as a third dimension
    It uses using_mpl_scatter_denisty function

    Arguments:
        x (np.ndarray): x data
        y (np.ndarray): y data
        title (str): plot title
        xlabel (str): x label
        ylabel (str): y label
        x_limit (Union[float, None]): max value of x axis
        y_limit (Union[float, None]): max value of y axis
        vmax (Union[float, None]): max value on the colour bar
        x_limit_min (Union[float, None]): min value of x axis
        y_limit_min (Union[float, None]): min value of y axis
        dpi (int): dpi
        fig_size (list): figure size

    Returns:
        fig, ax: figure and axes
    """

    plt.rcParams.update({"font.size": 18})
    plt.rcParams["figure.figsize"] = fig_size

    fig = plt.figure()

    fig, ax = using_mpl_scatter_density(fig, x, y, vmax, dpi)

    fig.autofmt_xdate(rotation=45)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # If no min x given, assume it is 0; otherwise, use the variable
    if x_limit_min is None:
        x_limit_min = 0

    ax.set_xlim([x_limit_min, x_limit])
    ax.set_ylim([y_limit_min, y_limit])
    #     ax.set_ylim([0, y_limit])

    return fig, ax
