# autoship-simulator
This repository implements a framework for simulating and evaluating autonomous ship collision avoidance (COLAV) control strategies.

The main functionality is contained in the `Simulator` class of `simulator.py`, which loads and runs scenarios. One can visualize the results underway, save animations from the results, and use the `colav_evaluation_tool` afterwards to evaluate the performance of the own-ship (potentially) running a COLAV algorithm.



## Dependencies
Are all outlined in setup.cfg, and listed below:

- dacite
- numpy
- matplotlib
- scipy
- cerberus
- pandas
- shapely
- pyyaml
- seacharts: https://github.com/trymte/thecolavrepo
- colav_evaluation_tool: https://github.com/trymte/colav_evaluation_tool
- yaspin

## Main modules

### Simulator

### Ship
The Ship class simulates the behaviour of an individual ship and adheres to the `IShip` interface, which necessitates that the ship class provides a `forward(dt) -> np.ndarray` function that allows simple simulation of the vessel.

It can be configured to use different combinations of collision avoidance algorithms, guidance systems, controllers, estimators, sensors, and models. The key element here is that each subsystem provides a standard inferface, which any external module using the subsystem must adhere to.  See the source code for more information on how this is done.

The `Ship` object must be initialized with the following parameters

- `mmsi`: String containing the Maritime Mobile Service Identity of the ship.
- `waypoints`: 2 x n_wps array containing the ship route plan.
- `speed_plan`: 1 x n_wps array containing the corresponding reference speeds for the ship.
- `state`: Initial pose of the ship, either `xs = [x, y, chi, U]` or `xs = [x, y, psi, u, v, r]^T` where for the first case, `x` and `y` are planar coordinates (north-east), `chi` is course (rad), `U` the ship forward speed (m/s). For the latter case, see the typical 3DOF surface vessel model in `Fossen2011`.
- `config`: Contains configuration settings for the ship subsystems.

Note that `n_wps` can be zero here. The parameters are loaded from the scenario library or generated using the scenario generator.

Main Functions: </p>

- `forward(dt)`: Simulates the ship `dt` seconds forward in time.
- `get_ais_data(t)`: Returns an AIS message for the ship information at the given UTC timestamp.

### Scenario Simulator

The scenario simulator module sets up and executes a single scenario involving one or more ships

Main functions:

- `init_scenario(new_scenario, scenario_file)`: Returns initialized and configured ships. If new scenario is set to true a new scenario will be generated, otherwise it will be loaded from the scenario file.
-  `run_scenario_simulation(ships, time, timestep)`: simulates one scenario and returns `visualization data` (for scenario animation) and  `ais_data` (for evaluation)

### Scenario Manager (main)
The Scenario manager sets up and runs scenario(s), evaluates and visualizes results.


## Git Workflow

All cooperators are obligated to follow the methods outlined in <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow> for ensuring a pain-free workflow with the repository.

If you're unfamiliar with git, check out <https://try.github.io/> to get familiar, and use <https://learngitbranching.js.org/> for trying out topics on your own.

.
