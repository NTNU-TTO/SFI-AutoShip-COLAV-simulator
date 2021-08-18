import os

import matplotlib.pyplot as plt

from scenario_simulator import *
from animation import visualize
from map import *
from yaspin import yaspin
from os import walk
import pathlib
import shutil

from UTC.colav.autoVerification.EvalTool import EvalTool

# Create output directories
if os.path.exists('output'):
    shutil.rmtree('output')
ais_path = 'output/ais'
ani_path = 'output/animation'
eval_path = 'output/eval'
pathlib.Path(ais_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(ani_path).mkdir(parents=True, exist_ok=True)
pathlib.Path(eval_path).mkdir(parents=True, exist_ok=True)


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
            verifier = EvalTool(save_path, r_pref=radius_preferred_cpa, r_min=radius_minimum_acceptable_cpa,
                                r_nm=radius_near_miss_encounter, r_col=radius_collision,
                                r_colregs_2_max=radius_colregs_2_max, r_colregs_3_max=radius_colregs_3_max,
                                r_colregs_4_max=radius_colregs_4_max)
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
            for i, vessel in enumerate(verifier.vessels):
                pathlib.Path('output/eval/ship'+str(i)).mkdir(parents=True, exist_ok=True)
                verifier.print_scores(vessel, save_results=True)
                vessel.plot_speed()
                plt.savefig('output/eval/ship'+str(i)+'/speed'+str(i)+'.png')
                vessel.plot_heading()
                plt.savefig('output/eval/ship'+str(i)+'/heading'+str(i)+'.png')
                verifier.plot_situation(vessel) # outputs which colreg situation the ship has as status
                plt.savefig('output/eval/ship'+str(i)+'/colreg_situation'+str(i)+'.png')

            if len(verifier.vessels) <= 3:
                verifier.plot_scores()
                plt.savefig('output/eval/plot_scores.png')
            verifier.plot_trajectories(ownship=verifier.vessels[0], show_cpa=True, vessels=None, show_maneuvers=False,
                                       legend=True, savefig=True, filename="output/eval/trajectories.png", ax=None)
            #verifier.plot_course_maneuvers() #outputs course, first derivative, second derivative, third derivative for own ship
            #plt.show()

        for file in os.listdir():
            if file.endswith('.xlsx'):
                shutil.move(file, 'output/eval/' + file)


if __name__ == '__main__':
    main()



