"""
This script 
"""

# check Python version (3.9 needed)
# !python --version # jupyter
import platform

print(platform.python_version())

# Import modules
import os

import osmium

import shapely
import shapely.wkb as wkblib


class BuildRoadRailHandler(osmium.SimpleHandler):
    # credit: https://max-coding.medium.com/getting-administrative-boundaries-from-open-street-map-osm-using-pyosmium-9f108c34f86
    # https://medium.com/@max-coding/extracting-open-street-map-osm-street-data-from-data-files-using-pyosmium-afca6eaa5d00
    """
    This class is used to handle OSM data. It extends the SimpleHandler class from the osmium package.
    It extracts buildings, roads, and railways from the OSM data.

    Arguments:
        osmium.SimpleHandler: The parent class for handling OSM data

    Returns:
        None
    """

    def __init__(self):
        # Call the parent class's constructor
        osmium.SimpleHandler.__init__(self)

        # Initialize lists to store buildings, roads, and railways
        self.buildings = []
        self.roads = []
        self.railways = []

        # Initialize a WKBFactory, which will be used to create geometries
        self.wkbfab = osmium.geom.WKBFactory()

    def area(self, a):
        """
        This method is called for each area in the OSM data.
        If the area is a building, it is added to the buildings list.

        Arguments:
            a (osmium.osm.Area): An area in the OSM data

        Returns:
            None

        """
        if "building" in a.tags:
            try:
                # Create a multipolygon geometry for the area
                wkbshape = self.wkbfab.create_multipolygon(a)
                # Convert the WKB shape to a Shapely object
                shapely_obj = shapely.wkb.loads(wkbshape, hex=True)

                # Create a dictionary representing the building
                building = {"id": a.id, "geometry": shapely_obj}

                # Add the building to the buildings list
                self.buildings.append(building)
            except RuntimeError as e:
                print(f"An error occurred while processing area with id {a.id}: {e}")

    def way(self, w):
        """
        This method is called for each way in the OSM data.
        If the way is a road or a railway, it is added to the corresponding list.

        Arguments:
            w (osmium.osm.Way): A way in the OSM data

        Returns:
            None
        """
        if w.tags.get("highway") in ["motorway", "trunk", "primary", "secondary"]:
            try:
                # Create a linestring geometry for the way
                wkb = self.wkbfab.create_linestring(w)
                # Convert the WKB shape to a Shapely object
                geo = wkblib.loads(wkb, hex=True)

            except:
                # If an error occurs while creating the geometry, skip this way
                return

            # Create a dictionary representing the road
            row = {"id": w.id, "geometry": geo}

            # Add highway and bridge columns
            for key, value in w.tags:
                if key in ["highway", "bridge"]:
                    row[key] = value

            # Add the road to the roads list
            self.roads.append(row)

        elif w.tags.get("railway") == "rail":
            try:
                # Create a linestring geometry for the way
                wkb = self.wkbfab.create_linestring(w)
                # Convert the WKB shape to a Shapely object
                geo = wkblib.loads(wkb, hex=True)
                # Create a dictionary representing the railway
                row = {"id": w.id, "geometry": geo, "bridge": w.tags["bridge"]}

            except KeyError:
                # except when there is no info about bridge (it'll set bridge to NaN)
                row = {"id": w.id, "geometry": geo}

            except:
                # If an error occurs while creating the geometry, skip this way
                return

            # Add the railway to the railways list
            self.railways.append(row)


def convert_osm_osmconvert(input_osm_path, output_osm_path):
    """
    Convert an OSM file to a different format using osmconvert.

    Arguments:
        input_osm_path (str): The path to the input OSM file.
        output_osm_path (str): The path to the output OSM file

    Returns:
        None
    """
    print("{} started!".format(output_osm_path), flush=True)

    # Check if the output file already exists
    if not os.path.exists(output_osm_path):
        # Construct the command to convert the OSM file

        command = f"/mnt/d/ML_model_data_paper/trainingnn_results_paper_revisions/pixel_wise_regression_run_predictions/osmconvert64 {input_osm_path} -o={output_osm_path}"

        try:
            # Execute the command
            os.system(command)
            print(f"{output_osm_path} finished!")
        except Exception as e:
            print(f"{output_osm_path} did not finish! Error: {e}")


def map_value(row):
    """
    Map the 'type' column to a specific value.

    This function takes a row of a DataFrame and returns a value based on the 'type' column.
    If the 'type' is not recognized, it returns None.

    Parameters:
        row (pandas.Series): A row of a DataFrame.

    Returns:
        float or None: The mapped value, or None if the 'type' is not recognized.
    """
    # Define a dictionary to map the 'type' to a value
    type_to_value = {
        "rail": 0.86,
        "building": 0.88,
        "motorway": 0.73,
        "trunk": 0.55,
        "primary": 0.59,
        "secondary": 0.59,
    }

    # Use the dictionary to get the value for the 'type'
    # If the 'type' is not in the dictionary, get() will return None
    return type_to_value.get(row["type"])
