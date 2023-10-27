"""
    file_utils.py

    Summary:
        Contains general non-math related utility functions.

    Author: Trym Tengesdal
"""


from pathlib import Path
from typing import Optional, Tuple

import colav_simulator.common.map_functions as mapf
import colav_simulator.common.miscellaneous_helper_methods as mhm
import colav_simulator.common.vessel_data as vd
import openpyxl
import pandas as pd
import yaml


def read_yaml_into_dict(file_name: Path) -> dict:
    with file_name.open(mode="r", encoding="utf-8") as file:
        output_dict = yaml.safe_load(file)
    return output_dict


def write_to_excel_file(data_frame: pd.DataFrame, file_prefix: str, sheet_name: str) -> None:
    file_name = file_prefix + ".xlsx"
    try:
        book = openpyxl.load_workbook(file_name)
        writer = pd.ExcelWriter(file_name, mode="a", engine="openpyxl")  # pylint: disable=abstract-class-instantiated
        writer.book = book
        writer.sheets = dict((ws.title, ws) for ws in book.worksheets)
        data_frame.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()
    except FileNotFoundError as e:
        print(e)
        writer = pd.ExcelWriter(file_name, engine="openpyxl")  # pylint: disable=abstract-class-instantiated
        data_frame.to_excel(writer, sheet_name=sheet_name, index=False)
        writer.save()


def read_ais_data(
    ais_path: Path,
    ship_info_path: Optional[Path] = None,
    utm_zone: int = 33,
    map_origin_enu: Optional[Tuple[float, float]] = None,
    map_size: Optional[Tuple[float, float]] = None,
    sample_interval: float = 1.0,
) -> dict:
    """
    Reads the ais data file specified by the ais_path parameter and creates a list of VesselData instances for
    each vessel recorded in the data. The list of MMSI`s and map origin/reference (in ENU) are also returned.

    Args:
    - ais_path (Path): Path to the ais data file.
    - ship_info_path (Path): Path to the ship information data file.
    - utm_zone (int): UTM zone of the coordinate system.
    - map_origin_enu (Optional[Tuple[float, float]]): Origin of the coordinate system in ENU coordinates.
    - map_size: (Optional[Tuple[float, float]]): Size of the considered area, relative to the origin, ENU coordinates.
    - sample_interval (float): Sampling interval used for interpolation on the vessel data.

    Returns:
        dict: Dictionary containing:
        - List of VesselData instances
        - List of vessel MMSI
        - Reference/origin of the local coordinate system in ENU coordinates.
        - Timespan of the data.
        - Tuple of size_x, size_y of the map area (+ buffer) containing the data, referenced to the origin.
        - Extent of the map area (+ buffer) containing the data, in lat/lon coordinates ([lat_min, lat_max, lon_min, lon_max])
    """
    output = {}
    vessels = []
    mmsi_list = []
    ship_info_df = None
    if ais_path.is_file():
        ais_df = pd.read_csv(ais_path, sep=";", parse_dates=["date_time_utc"], infer_datetime_format=True)
    else:
        raise FileNotFoundError(f"AIS data file not found: {ais_path}")

    if ship_info_path is not None and ship_info_path.is_file():
        ship_info_df = pd.read_csv(ship_info_path, sep=";", dtype={"mmsi": "uint32", "length": "float16", "width": "float16"})

    origin_buffer = 0.01
    lat0 = min(ais_df.lat) - 0.01
    lon0 = min(ais_df.lon) - 0.01
    lat_max = max(ais_df.lat) + origin_buffer
    lon_max = max(ais_df.lon) + origin_buffer

    if map_origin_enu is None:
        map_origin_enu = mapf.latlon2local(lat0, lon0, utm_zone=utm_zone)

    size: list = [0.0, 0.0]
    if map_size is None:
        size[0] = mapf.dist_between_latlon_coords(lat0, lon0, lat0, lon_max)
        size[1] = mapf.dist_between_latlon_coords(lat0, lon0, lat_max, lon0)
    else:
        size = list(map_size)

    t_0 = ais_df.date_time_utc.min()
    t_end = ais_df.date_time_utc.max()

    count = 0
    ship_ais_df_list = mhm.get_ship_ais_df_list_from_ais_df(ais_df)
    for ship_ais_df in ship_ais_df_list:
        vessel = vd.VesselData.create_from_ais_data(
            t_0_global=t_0,
            t_end_global=t_end,
            identifier=count,
            ship_ais_df=ship_ais_df,
            ship_info_df=ship_info_df,
            utm_zone=utm_zone,
            sample_interval=sample_interval,
        )
        if vessel is not None and vessel.status != vd.Status.AtAnchor:
            vessels.append(vessel)
            mmsi_list.append(vessel.mmsi)
            # vessel.plot_trajectory()
            count += 1

    output["vessels"] = vessels
    output["mmsi_list"] = mmsi_list
    output["map_origin_enu"] = list(map_origin_enu)
    output["map_size"] = size
    output["timespan"] = [t_0, t_end]
    output["lla_extent"] = [lat0, lon0, lat_max, lon_max]
    return output
