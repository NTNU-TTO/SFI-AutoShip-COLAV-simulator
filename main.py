from scenario_simulator import *
from animation import visualize
from map import *
from yaspin import yaspin
from os import walk
import sys
sys.path.insert(0, './UTC/colav/autoVerification/')
import pathlib

from EvalTool import EvalTool

# Create output directories
ais_path = 'output/ais'
ani_path = 'output/animation'
pathlib.Path(ais_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(ani_path).mkdir(parents=True, exist_ok=True)


@yaspin(text="Running...")
def main():
    ###############################################
    # INITIALISATION
    ###############################################

    # time
    t = np.arange(time_start, time_end + time_step, time_step)

    animation_data_list, ais_data_list = [], []
    verifier_list = []
    scenario_names = []

    if run_all_scenarios:
        ### Run all scenarios defined in the scenarios folder ###
        scenario_file_names = next(walk('scenarios'), (None, None, []))[2]
    else:
        ### Single scenario run ###
        scenario_file_names = [scenario_file]

    ###############################################
    # SIMULATION AND EVALUATION
    ###############################################
    for scenario_file_name in scenario_file_names:
        scenario_name = scenario_file_name[:-5] # scenario name without file format
        scenario_names.append(scenario_name)
        # Initiates the scenario and configures the ships
        ship_list, waypoint_list, speed_plan_list = init_scenario(new_scenario=new_scenario,
                                                                scenario_file=scenario_file_name
                                                                )

        animation_data, ais_data, colav_input = run_scenario_simulation(ships=ship_list, time=t, timestep=time_step)
        animation_data_list.append(animation_data)
        # exporting ais_data to .csv file
        filename = f'{scenario_name}.csv'
        save_path = f'{ais_path}/{filename}'
        ais_data.to_csv(save_path, sep=';')

        if evaluate_results:
            verifier = EvalTool(save_path)
            verifier.evaluate_vessel_behavior()
            verifier_list.append(verifier)
    
    ###############################################
    # ANIMATION
    ###############################################

    if show_animation or save_animation:
        for i, data in enumerate(animation_data_list):
            save_file=f'{scenario_names[i]}.gif'
            save_path = f'{ani_path}/{save_file}'
            visualize(data, t, show_waypoints, show_animation, save_animation, save_path=save_path)
    
    ###############################################
    # EVALUATION
    ###############################################
    if evaluate_results:
        for verifier in verifier_list:
            for vessel in verifier.vessels:
                verifier.print_scores(vessel)
            verifier.plot_trajectories(ownship=verifier.vessels[0], show_cpa=True, vessels=None, show_maneuvers=False, legend=True,
                                savefig=True, filename="figure.png", ax=None)

if __name__ == '__main__':
    main()



