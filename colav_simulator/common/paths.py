"""Contains "hard-coded" paths to default configuration files"""
import pathlib

root = pathlib.Path.cwd().parents[2]
package = root / 'colav_simulator'
config = root / 'config'
simulator = config / 'simulator.yaml'
seacharts = config / 'seacharts.yaml'
ships = config / 'ships.yaml'
new_scenario = config / 'new_scenario.yaml'
map_data = root / 'map_data'
