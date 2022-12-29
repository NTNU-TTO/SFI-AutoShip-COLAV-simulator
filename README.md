# colav-simulator
This repository implements a framework for simulating and evaluating autonomous ship collision avoidance (COLAV) control strategies.

The main functionality is contained in the `Simulator` class of `simulator.py`, which loads and runs scenarios. One can visualize the results underway, save the results, and use the `colav_evaluation_tool` afterwards to evaluate the performance of the own-ship (potentially) running a COLAV algorithm. The `seacharts` package is used to provide usage of Electronic Navigational Charts for visualization and anti-grounding purposes.

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
- seacharts: https://github.com/trymte/seacharts
- colav_evaluation_tool: https://github.com/trymte/colav_evaluation_tool
- yaspin

## Git Workflow

All contributors are obligated to follow methods as outlined in <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow> for ensuring a pain-free workflow with the repository.

If you're unfamiliar with git, check out <https://try.github.io/> to get familiar, and use <https://learngitbranching.js.org/> for trying out topics on your own.

### Main branch
The `main` branch shall always be working. This means that:

- All of its features/modules shall be properly documented through quality code + descriptive text where appropriate. Use `type annotation` for increased readability.
- The `Simulator` is successfully able to run through any number of scenarios, either generated or loaded from file. Furthermore, it should be problem free to save and load scenarios to/from (valid) scenario files.
- The `Visualizer`is successfully able to visualize each of these scenarios live, if toggled on.
- Subsystems added to the `Ship` class properly adheres to their interfaces (prefixed with `I` in front of the class name, e.g. interface `IModel` for the ship model interface).

### Project and master thesis work
For students who use this repository for their project/master work, we want you to be free to experiment with the code. In order to enable this, such work shall be performed in branches prepended with `project/<year>/`. For example, if Per in 2023 is working on his master thesis creating a deep reinforcement learning-based trajectory tracking controller, he should do so by branching out from `main` into his own branch called e.g. `project/2023/drl-controller`. This makes browsing the branches later easier.

### Features and dedicated improvements

A lot of the work being done on code repositories is adding needed features, or improving the platform in general.

The branches used to develop these features shall be prepended with `feature/`, e.g.
`feature/improved_ship_configuration`.

It is also a good idea to keep the features small, so that they are easy to test, and can quickly be merged into `main`.

### Workflow

When you're developing a feature (or beginning your thesis work) based on the simulator, the workflow begins by checking out the main branch, and creating a new branch from there.

#### Retrieving main branch

```bash
cd colav_simulator
git checkout main
git pull
# If you need to rebase because you have uncommited (non-important) changes in your working directory, you can run:
git reset --hard origin/main
```

#### Creating new branch

A new branch for new features is created like this:

```bash
git checkout -b feature/name_of_feature
```

#### Performing work and updating branch

Now that you have a separate branch, you can code, commit, run experiments and cooperate in that branch.

```bash
# Do some work
git add <files>
git commit
git push -u origin feature/name_of_feature # -u flag with origin only needed first time to set upstream.
```

#### Performing tests before merging into main

After doing some development and thorough testing, it might be time to merge the feature into the `main` branch.
Before doing that, make sure that all the steps listed under the heading [_Main branch_](#Main-branch).

To make sure new issues don't arise when merging into `main` later (because of other changes to `main`) we merge `main` into `feature/name_of_feature` first:

```bash
git checkout main   # make sure we have the latest
git pull            # version of main locally
git checkout feature/name_of_feature
git pull # in case you or someone else made changes remotely
git merge main # This attempts to merge `main` into `feature/name_of_feature`
# At this point conflicts might arise which you have to solve.
```

Now you can perform the tests.

After the tests have been completed, you can initiate a pull request on github, where eventually the changes will be merged to `main`.

#### Pull request

Go to the `colav_simulator` repo on <github.com> and navigate to your branch (`feature/name_of_feature`).
In the status bar above the commit message there is a link for _Pull request_.
Click that and describe your feature.
Assign people to it to review the changes, and create the pull request.

The pull request should now be reviewed by someone, and if it's ok, it will be merged into `main`.
Congratulations! It is now safe to delete the feature branch, and is strongly encouraged in order to keep the repository tidy.


## Main modules in the repository

Each main module have their own test files, to enable easier debug/fixing and also for making yourself familiar with the code. When developing new modules, you are encouraged to simultaneously develop test files for these, such that yourself and others more conveniently can fix/debug and test the modules separately.

### Simulator

The simulator runs through a set of scenarios specified from the config, visualizes these and saves the results. The scenarios can be generated randomly through a `new_scenario.yaml` config file in the ScenarioGenerator, or loaded from file using an existing scenario definition.

The simulator is configured using the `simulator.yaml` config file.

### Scenario Generator

The scenario generator (found inside `scenario_management.py`) is used by the simulator to create new scenarios for COLAV testing with 1+ ships. The main method is the `generate()` function, which generates a scenario from a scenario config file, which is converted into a `ScenarioConfig` object. An Electronic Navigational Chart object (from Seacharts) is used to define the environment. The scenarios can be specified/created as follows:

1) Fully random, with random poses, waypoints and speed plans for the vessels
2) Fully specified from AIS data loaded from `.csv` files. The own-ship is here specified as one of the AIS vessels (default being the first AIS vessel), or with randomly generated pose, waypoints and speed plan. The target ships from AIS follow their historical trajectories, interpolated to the match the sample interval considered in the simulator.
3) As a mix of randomly generated vessels and AIS vessels.

Scenarios are configured through a scenario `.yaml` file as the example `head_on.yaml` file under `scenarios` in the root folder.

Look at the `ScenarioConfig` dataclass, example scenario files for information on how to specify the settings for a scenario.

Look at the `schemas` folder under the package source code for further clues on how to write a new scenario config file.

Seacharts is used to provide access to Electronic Navigational Charts, and an `ENC` object is used inside the `ScenarioGenerator` class for this. One must here make sure that the seacharts package is properly setup with `.gdb` data in the `data/external` folder of the package, with correctly matching `UTM` zone for the chart data. An example `seacharts.yaml`config file for the module is found under `config/`. One can specify map data, map origin, map size etc. for the ENC object from the scenario `.yaml`config file.

### Visualizer

Class responsible for visualizing scenarios run through by the Simulator, and visualizing/saving the results from these. A basic live plotting feature when simulating scenarios is available.

The class can, as most other main modules, be configured from the example config files under `config/`.

#### Ship
The Ship class simulates the behaviour of an individual ship and adheres to the `IShip` interface, which necessitates that the ship class provides a `forward(dt) -> np.ndarray` function that allows simple simulation of the vessel.

It can be configured to use different combinations of collision avoidance algorithms, guidance systems, controllers, estimators, sensors, and models. The key element here is that each subsystem provides a standard inferface, which any external module using the subsystem must adhere to.  See the source code for more information on how this is done.

TODO: Implement interfaces for using arbitrary collision avoidance algorithms and planning hierarchys.

The `Ship` object can be initialized with the following parameters

- `mmsi` (mandatory): Integer containing the Maritime Mobile Service Identity number of the ship.
- `waypoints` (optional): 2 x n_wps array containing the ship route plan.
- `speed_plan` (optional): 1 x n_wps array containing the corresponding reference speeds for the ship.
- `pose` (optional): Initial pose of the ship `xs = [x, y, U, chi]` where `x` and `y` are planar coordinates (north-east), `chi` is course (rad), `U` the ship forward speed (m/s). Note that the internal `state` of the vessel can be either 4-dimensional or 6-dimensional, depending on the ship model used.
- `config` (optional): Contains configuration settings for the ship subsystems.

Except from the ship config, the other optional arguments must be specified afterwards through the `set_initial_pose` and `set_nominal_plan` for the ship object to be functionable.

See the source code for more in depth info on the `Ship` functionality.
