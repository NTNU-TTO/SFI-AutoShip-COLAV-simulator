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
simulator_schema = schemas / "simulator.yaml"
scenario_schema = schemas / "scenario.yaml"
scenario_generator_schema = schemas / "scenario_generator.yaml"

simulator_config = config / "simulator.yaml"
scenario_generator_config = config / "scenario_generator.yaml"
seacharts_config = config / "seacharts.yaml"

enc_data = root / "data" / "map"
ais_data = root / "data" / "ais"
saved_scenarios = scenarios / "saved"

animation_output = output / "animations"
figure_output = output / "figures"
