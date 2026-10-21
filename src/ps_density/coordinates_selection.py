"""
This module contains functions for handling coordinates selection and 
generating a list of coordinate pairs based on user input.

Functions:
- input_int_lat(message: str) -> int: Ensures that the user inputs 
an integer within the range of -90 to 90 for latitude.
- input_int_lon(message: str) -> int: Ensures that the user inputs 
an integer within the range of -180 to 180 for longitude.
- get_coordinates() -> List: Asks users to select how they want to define 
the Area of Interest (AOI) and generates a list of coordinate pairs representing 
the latitude and longitude values for the region based on the user's input.
- haversine_distance: Great-circle distance between two lat/lon points, in kilometres.

"""

import itertools
import math
from typing import List
from shapely.geometry import Polygon
import geopandas as gpd
import osmnx as ox
import matplotlib.pyplot as plt
import numpy as np

# Functions for reading input Lat and Lon
# and handling errors (if lat/lon is not an int)


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points, in kilometres.

    Arguments:
        lat1 (float): Latitude of the first point in degrees.
        lon1 (float): Longitude of the first point in degrees.
        lat2 (float): Latitude of the second point in degrees.
        lon2 (float): Longitude of the second point in degrees.

    Returns:
        float: Distance in kilometres.
    """
    radius_km = 6371.0
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2
    )
    return radius_km * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def input_int_lat(message: str) -> int:
    """
    The function ensures that the user inputs an integer within the range of -90 to 90.

    Parameters:
        message (str): The message to be displayed to the user.

    Returns:
        user_input (int): Lat value that is within the range of -90 to 90.
    """
    while True:
        try:
            user_input = int(input(message))
        except ValueError:
            print("This is not an integer! Try again.")
            continue
        else:
            if -90 <= user_input <= 90:
                return user_input
                # break
            else:
                print("Number not in range! Try again.")
                continue


def input_int_lon(message: str) -> int:
    """
    The function ensures that the user inputs an integer within the range of -180 to 180.

    Parameters:
        message (str): The message to be displayed to the user.

    Returns:
        user_input (int): Lon value that is within the range of -180 to 180.
    """
    while True:
        try:
            user_input = int(input(message))
        except ValueError:
            print("This is not an integer! Try again.")
            continue
        else:
            if -180 <= user_input <= 180:
                return user_input
                # break
            else:
                print("Number not in range! Try again.")
                continue


# Set up region to be downloaded
# Coordinates of the tile refer to the top left corner of the tile and the code accounts for that.
# Thus, if you want to cover region from (5,-2) to (7,1)
# set lat_min_max to [5,7] and lon_min_max to [-2,1]
# and the following tiles will be downloaded:
# N06W002, N06W001, N06E000, N07W002, N07W001, N07E000 covering your region of interest.
# lat_min_max = [-100,43] #both values from range included;
# must between 0 and (-)90; positive N, negative S
# lon_min_max = [11,12] #both values from range included;
# must between 0 and (-)180; positive E, negative W


def get_coordinates():
    """
    The function ask users to select how they want to define the AOI (Area of Interest) - either by
    providing the range of coordinates or by providing the name of the AOI as defined in OSM.
    If the user selects to provide the range of coordinates, the function asks for the range of
    latitude and longitude values.
    If the user selects to provide the name of the AOI, the function asks for the names of the
    places separated by a semicolon.
    The function then generates a list of coordinate pairs representing the latitude and longitude
    values for the region based on the user's input.

    Parameters:
        None

    Returns:
        List: A list of coordinate pairs representing the latitude and longitude values for the region


    """

    print("Select how do you want to define the AOI")
    print("1 - You can either provide the range of coordinates")
    print("2 - or you can provide name of the AOI (as it is defined in OSM)")

    while True:
        try:
            selected_mode = int(
                input(
                    "Type 1 (if you want to use range of coordinates)"
                    + "or  2 (if you prefer to select by name) \n "
                )
            )
        except ValueError:
            print("This is not an integer! Try again.")
            continue
        else:
            if 1 <= selected_mode <= 2:
                #             print(selected_mode)
                break
            else:
                print("Number not in range! Try again.")
                continue

    if selected_mode == 1:
        # Set up empty lists for storing lat and lon values
        lat_min_max: List[int] = []
        lon_min_max: List[int] = []

        print(
            "\n Provide range of lat as two integers. "
            + "Values must be between 0 and (-)90 (positive N, negative S) \n"
        )

        lat_min_max.insert(0, input_int_lat("Lat min "))
        lat_min_max.insert(1, input_int_lat("Lat max "))

        print(
            "\n Now, provide range of lon as two integers. "
            + "Values must be between 0 and (-)180 (positive E, negative W) \n"
        )

        lon_min_max.insert(0, input_int_lon("Lon min "))
        lon_min_max.insert(1, input_int_lon("Lon max "))

        # generate list of lat and lon coordinates to be downloaded
        lat_range = list(range(min(lat_min_max), max(lat_min_max)))
        lon_range = list(range(min(lon_min_max), max(lon_min_max)))

        coordinates_to_download = list(itertools.product(lon_range, lat_range))

        df_polygons = gpd.GeoDataFrame(coordinates_to_download, columns=["lon", "lat"])
        df_polygons["poly_geometry"] = ""

        for i in range(len(df_polygons)):
            lon = df_polygons.loc[i, "lon"]
            lat = df_polygons.loc[i, "lat"]
            poly = Polygon(
                [(lon, lat), (lon, lat + 1), (lon + 1, lat + 1), (lon + 1, lat)]
            )
            df_polygons.loc[i, "poly_geometry"] = poly

        df_polygons = df_polygons.set_geometry("poly_geometry")
        world_borders = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

        plt.rcParams["figure.figsize"] = [40, 20]
        _, ax = plt.subplots()
        world_borders.plot(color="lightgray", ax=ax)
        df_polygons.geometry.boundary.plot(
            color=None, edgecolor="k", linewidth=2, ax=ax
        )

    elif selected_mode == 2:
        while True:
            try:
                places_input = input(
                    "Pass names of places you want to consider separated by ; \n"
                    + "(e.g. Italy; Austria or USA or Bristol, UK; Bath, UK) \n"
                )
            except Exception as e:
                print(f"Something went wrong, try again. Error code: {e} \n")
                continue
            else:
                try:
                    places_input_splt = places_input.split("; ")
                    places_input_splt = [str(x) for x in places_input_splt]
                    gdf_places = ox.geocode_to_gdf(places_input_splt)
                    break
                except Exception as e:
                    print(
                        "Something went wrong, try again. Please check if the spelling of your input is correct. \n"
                        + "Error code: {} \n".format(e)
                    )
                    continue

        for i in range(len(gdf_places)):
            poly_place = gdf_places.loc[i, "geometry"]

            lon_min = np.floor(
                min([gdf_places.loc[i, "bbox_west"], gdf_places.loc[i, "bbox_east"]])
            )
            lon_max = np.ceil(
                max([gdf_places.loc[i, "bbox_west"], gdf_places.loc[i, "bbox_east"]])
            )
            lat_min = np.floor(
                min([gdf_places.loc[i, "bbox_south"], gdf_places.loc[i, "bbox_north"]])
            )
            lat_max = np.ceil(
                max([gdf_places.loc[i, "bbox_south"], gdf_places.loc[i, "bbox_north"]])
            )
            #         print(lon_min, lon_max, lat_min, lat_max)

            lon_range = list(range(int(lon_min), int(lon_max)))
            lat_range = list(range(int(lat_min), int(lat_max)))

            coordinates_to_download = list(itertools.product(lon_range, lat_range))

            df_polygons = gpd.GeoDataFrame(
                coordinates_to_download, columns=["lon", "lat"]
            )
            df_polygons["poly_geometry"] = ""
            df_polygons["to_be_used"] = ""

            for j in range(len(df_polygons)):
                lon = df_polygons.loc[j, "lon"]
                lat = df_polygons.loc[j, "lat"]
                poly = Polygon(
                    [(lon, lat), (lon, lat + 1), (lon + 1, lat + 1), (lon + 1, lat)]
                )
                df_polygons.loc[j, "poly_geometry"] = poly
                df_polygons.loc[j, "to_be_used"] = poly_place.intersects(poly)

            df_polygons = df_polygons.set_geometry("poly_geometry")

            if i == 0:
                df_polygons_ouput = df_polygons.copy()
            else:
                df_polygons_ouput = df_polygons_ouput.append(df_polygons).reset_index()

        world_borders = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

        plt.rcParams["figure.figsize"] = [40, 20]
        _, ax = plt.subplots()
        world_borders.plot(color="lightgray", ax=ax)
        gdf_places.plot(ax=ax)
        df_polygons_ouput[df_polygons_ouput["to_be_used"]].geometry.boundary.plot(
            color=None, edgecolor="k", linewidth=2, ax=ax
        )

        lon_to_be_used = df_polygons_ouput[df_polygons_ouput["to_be_used"]]["lon"]
        lat_to_be_used = df_polygons_ouput[df_polygons_ouput["to_be_used"]]["lat"]
        coordinates_to_download = list(zip(lon_to_be_used, lat_to_be_used))

    return coordinates_to_download
