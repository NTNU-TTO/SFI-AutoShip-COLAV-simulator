"""
    paths.py

    Summary:
        Contains paths to default configuration files and schemas.

    Author: Trym Tengesdal
"""

import pathlib
import sys

local_root = pathlib.Path(sys.argv[0]).absolute().parents[1]
lib_root = pathlib.Path(__file__).absolute().parents[2]
config = local_root / "config"
package = lib_root / "colav_simulator"
scenarios = local_root / "scenarios"
output = local_root / "output"

schemas = package / "schemas"
simulator_schema = schemas / "simulator.yaml"
scenario_schema = schemas / "scenario.yaml"
scenario_generator_schema = schemas / "scenario_generator.yaml"

simulator_config = config / "simulator.yaml"
scenario_generator_config = config / "scenario_generator.yaml"
seacharts_config = config / "seacharts.yaml"

enc_data = local_root / "data" / "map"
ais_data = local_root / "data" / "ais"
saved_scenarios = scenarios / "saved"

animation_output = output / "animations"
figure_output = output / "figures"
