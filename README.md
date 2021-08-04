# autoship-simulator
<p> This repository implements a framework for simulating and evaluating autonomous ship collision avoidance control strategies. </p>

## Main modules

### Ship 

<p> Ship class simulates an individual ship for a short discretization step <br>

Init input: </p>

- Pose (x,y,speed,heading), waypoints, speed plan. These are parameters are loaded from the scenarios library or created using the scenario generator
- Ship model name. This loads the ship model parameters from the ship model library. Current models: telemetron and random
- mmsi, used as name and ais mmsi
- Sensors. List of initialized sensors (Radar, AIS)
- LOS parameters. Lookahead distance and radius of acceptance used for LOS guidance

Main Functions: </p>

- update_states(dt): updates the pose and reference for one discretization step
- update_target_pose_est(pose_list, t, dt): updates pose estimates of the other ships

### Scenario Simulator

<p> Scenario simulator module sets up and execute a single scenario involving two or more ships  <br>

Main functions: </p>

- init_scenario(new_scenario, scenario_file): Returns initialized and configured ships. If new scenario is set to true a new scenario will be generated, otherwise it will be loaded from the scenario file. Ship config is loaded from config_ships.ini
-  run_scenario_simulation(ships, time, timestep): simulates one scenario and returns visulation data (for scenario animation) and ais_data (for scenario COLREG evaluation)

### Scenario Manager (main)
<p> Scenario manager sets up and runs scenario(s), and evaluates and visulaizes results  <br>

## Git Workflow

All cooperators are obligated to follow the methods outlined in <https://www.atlassian.com/git/tutorials/comparing-workflows/feature-branch-workflow> for ensuring a pain-free workflow with the repository. 

If you're unfamiliar with git, check out <https://try.github.io/> to get familiar, and use <https://learngitbranching.js.org/> for trying out topics on your own.
