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
scenarios = root / "scenarios"
output = root / "output"

schemas = package / "schemas"
ship_schema = schemas / "ship.yaml"
simulator_schema = schemas / "simulator.yaml"
existing_scenario_schema = schemas / "existing_scenario.yaml"
new_scenario_schema = schemas / "new_scenario.yaml"
scenario_generator_schema = schemas / "scenario_generator.yaml"
visualizer_schema = schemas / "visualizer.yaml"

ship_config = config / "ships.yaml"
simulator_config = config / "simulator.yaml"
seacharts_config = config / "seacharts.yaml"
new_scenario_config = config / "new_scenario.yaml"
scenario_generator_config = config / "scenario_generator.yaml"
visualizer_config = config / "visualizer.yaml"

enc_data = root / "data" / "external"
ais_output = output / "ais"
animation_output = output / "animation"
