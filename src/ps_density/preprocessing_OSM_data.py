"""
This script 
"""

# check Python version (3.9 needed)
# !python --version # jupyter
import platform

print(platform.python_version())

# Import modules
import os
import subprocess
from pathlib import Path

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


def _run_osmconvert(osmconvert_bin, inputs, output_path, bbox=None):
    """Run osmconvert. The output is published only after a zero exit code."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f".{output_path.stem}.partial.o5m")
    command = [str(osmconvert_bin), *(str(path) for path in inputs)]
    if bbox is not None:
        west, south, east, north = bbox
        command.append(f"-b={west},{south},{east},{north}")
    command.append(f"-o={partial}")
    result = subprocess.run(command, check=False)
    if result.returncode != 0 or not partial.exists():
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"osmconvert failed (exit {result.returncode}) writing {output_path}"
        )
    partial.replace(output_path)


def convert_osm_osmconvert(
    input_osm_path, output_osm_path, osmconvert_bin=None, bbox=None
):
    """
    Convert one or more OSM files to a different format using osmconvert.

    osmconvert accepts only one ``.pbf`` input per call. Extra extracts are
    converted to ``.o5m`` once (cached beside the tile output) and then merged
    and clipped together. Border objects that appear in more than one extract
    are kept; rasterization paints the same value again.

    Arguments:
        input_osm_path (str or sequence): One OSM path, or several paths
            that together cover the clip.
        output_osm_path (str): The path to the output OSM file
        osmconvert_bin (str): osmconvert executable. Defaults to
            ``predict.osmconvert`` in config/paths.yaml (or ``osmconvert`` on PATH).
        bbox (tuple): Optional (west, south, east, north) clip in WGS84 degrees.

    Returns:
        None
    """
    print("{} started!".format(output_osm_path), flush=True)

    # Check if the output file already exists
    if os.path.exists(output_osm_path):
        return

    if isinstance(input_osm_path, (str, os.PathLike)):
        inputs = [input_osm_path]
    else:
        inputs = list(input_osm_path)
    if not inputs:
        raise ValueError("No OSM inputs were given to osmconvert")

    if osmconvert_bin is None:
        from ps_density.paths import load_paths

        osmconvert_bin = load_paths().predict.osmconvert

    pbf_inputs = [path for path in inputs if str(path).lower().endswith(".pbf")]
    if len(pbf_inputs) > 1:
        cache_dir = Path(output_osm_path).parent / "_pbf_as_o5m"
        merged_inputs = []
        for path in inputs:
            path = Path(path)
            if str(path).lower().endswith(".pbf"):
                cached = cache_dir / f"{path.name}.o5m"
                if not cached.exists():
                    print(
                        f"Converting {path.name} to o5m "
                        "(osmconvert allows only one .pbf input)",
                        flush=True,
                    )
                    _run_osmconvert(osmconvert_bin, [path], cached)
                merged_inputs.append(cached)
            else:
                merged_inputs.append(path)
        inputs = merged_inputs

    _run_osmconvert(osmconvert_bin, inputs, output_osm_path, bbox=bbox)
    print(f"{output_osm_path} finished!", flush=True)


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
