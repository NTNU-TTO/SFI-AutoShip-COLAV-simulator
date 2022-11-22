# colav-simulator
This repository implements a framework for simulating and evaluating autonomous ship collision avoidance (COLAV) control strategies.

The main functionality is contained in the `Simulator` class of `simulator.py`, which loads and runs scenarios. One can visualize the results underway, save visualizations/animations from the results, and use the `colav_evaluation_tool` afterwards to evaluate the performance of the own-ship (potentially) running a COLAV algorithm. The `seacharts` package is used to provide usage of Electronic Navigational Charts for visualization and anti-grounding purposes.

[![platform](https://img.shields.io/badge/platform-linux-lightgrey)]()
[![python version](https://img.shields.io/badge/python-3.10-blue)]()


## Dependencies
Are all outlined in setup.cfg, and listed below:

- numpy
- matplotlib
- cartopy
- scipy
- pandas
- shapely
- pyyaml
- cerberus
- dacite
- seacharts: https://github.com/trymte/thecolavrepo
- colav_evaluation_tool: https://github.com/trymte/colav_evaluation_tool
- yaspin

## Main modules

### Simulator

The simulator runs through a set of scenarios specified from the config, visualizes these and saves the results. The scenarios can be generated randomly through a `new_scenario.yaml` config file in the ScenarioGenerator, or loaded from file using an existing scenario definition.

The simulator is configured using the `simulator.yaml` config file.

### Scenario Generator

The scenario generator is used by the simulator to create new random scenarios involving 1+ ships, with random poses, waypoints and speed plans for the vessels. An Electronic Navigational Chart module (from Seacharts) is used to define the environment. The main method is the `generate()` function, which generates a random vessel scenario from an optional config file.

The new scenarios are configured through a config file as e.g. the `new_scenario.yaml` file. Here, one can configure:

- The `ScenarioType`, which ranges from single ship to Head-on, Crossing Give-way, Multi-ship
- The number of ships in the scenario
- Minimum and maximum distance between ships
- A list containing configured ships in the scenario, with specific poses, waypoints, speed plans, models, controllers, guidance systems etc. If `n_ships` is greater than the number of elements in the `ship_list` variable, the remaining ships are configured from the default `Ship` constructor.

Look at the `schemas` folder for further clues on how to write a new scenario config file.

Seacharts is used to provide access to Electronic Navigational Charts, and an `ENC` object is used inside the `ScenarioGenerator` class for this. One must here make sure that the seacharts package is properly setup with `.gdb` data in the `data/external` folder of the package, with correctly matching `UTM` zone for the chart data. An example `seacharts.yaml`config file for the module is found under `config/`.

### Visualizer

Class responsible for visualizing scenarios run through by the Simulator, and visualizing/saving the results from these.

#### Ship
The Ship class simulates the behaviour of an individual ship and adheres to the `IShip` interface, which necessitates that the ship class provides a `forward(dt) -> np.ndarray` function that allows simple simulation of the vessel.

It can be configured to use different combinations of collision avoidance algorithms, guidance systems, controllers, estimators, sensors, and models. The key element here is that each subsystem provides a standard inferface, which any external module using the subsystem must adhere to.  See the source code for more information on how this is done.

The `Ship` object can be initialized with the following parameters

- `mmsi` (mandatory): Integer containing the Maritime Mobile Service Identity number of the ship.
- `waypoints` (optional): 2 x n_wps array containing the ship route plan.
- `speed_plan` (optional): 1 x n_wps array containing the corresponding reference speeds for the ship.
- `pose` (optional): Initial pose of the ship `xs = [x, y, U, chi]` where `x` and `y` are planar coordinates (north-east), `chi` is course (rad), `U` the ship forward speed (m/s). Note that the internal `state` of the vessel can be either 4-dimensional or 6-dimensional, depending on the ship model used.
- `config` (optional): Contains configuration settings for the ship subsystems.

Except from the ship config, the other optional arguments must be specified afterwards through the `set_initial_pose` and `set_nominal_plan` for the ship object to be functionable.

Main Functions: </p>

- `forward(dt)`: Simulates the ship `dt` seconds forward in time.
- `track_obstacles()`: (TODO): Implement function for tracking nearby vessels
- `get_ais_data(t)`: Returns an AIS message for the ship information at the given UTC timestamp.
- `get_ship_nav_data(t)`: Returns relevant simulation data for the ship at the given UTC timestamp. This includes its pose, waypoints, speed plan etc.



## Git Workflow

Everyone are encouraged to contribute to the software in this repository, to provide new features or fix existing issues.

All contributors are obligated to follow methods as outlined in <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow> for ensuring a pain-free workflow with the repository.

If you're unfamiliar with git, check out <https://try.github.io/> to get familiar, and use <https://learngitbranching.js.org/> for trying out topics on your own.
