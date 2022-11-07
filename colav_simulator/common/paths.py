"""
    paths.py

    Summary:
        Contains paths to default configuration files and schemas.

    Author: Trym Tengesdal
"""
import pathlib

root = pathlib.Path(__file__).parents[2]
config = root / "config"
package = root / "colav_simulator"
schemas = package / "schemas"

simulator_schema = schemas / "simulator.yaml"
ship_schema = schemas / "ship.yaml"
scenario_generation_schema = schemas / "scenario.yaml"

simulator_config = config / "simulator.yaml"
seacharts_config = config / "seacharts.yaml"
ships_config = config / "ships.yaml"
scenario_generation_config = config / "scenario.yaml"

map_data = root / "map_data"
