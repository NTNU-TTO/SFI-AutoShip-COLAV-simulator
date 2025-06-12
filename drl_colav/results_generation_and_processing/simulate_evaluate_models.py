import time
import pathlib
import os
import re

import numpy as np
from colav_evaluation_tool.evaluator import Evaluator
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator
import colav_simulator.core.colav.colav_interface as ci

import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import colav_simulator.gym.reward as rewards
import colav_simulator.gym.action as actions

def extract_scenario(file_name : str):
    if "imazu" in file_name:
        match = re.search(r"imazu\d{2}", file_name)
        return match.group(0) if match else None
    else:
        return file_name.split(".yaml")[0]

def simulate_and_evaluate_sbmpc(eval_scenario_foler) -> None:
    scenario_generator = ScenarioGenerator()
    simulator = Simulator()
    evaluator = Evaluator()
    sbmpc_obj = ci.SBMPCWrapper()

    # Evaluation of standard SBMPC algortihm with no RL tuning
    
    sl = [f for f in eval_scenario_foler.iterdir() if f.is_file()]
    scenario_list = scenario_generator.generate_scenarios_from_files(sl)
    simulator.toggle_liveplot_visibility(True)
    scenario_result_data_list = simulator.run(scenario_list, colav_systems=[(0, sbmpc_obj)])
    
    count = 1
    for scenario_data in scenario_result_data_list:
        episode_simdata_list = scenario_data["episode_simdata_list"]
        evaluator.set_enc(scenario_data["enc"])

        for episode_data in episode_simdata_list:
            vessels = episode_data["vessel_data"]
            print("\nEvaluating scenario " + str(count) + " with " + str(len(vessels)) + " vessels...\n")

            evaluator.set_vessel_data(vessels)
            results = evaluator.evaluate()
            print("\n*** Standard SB-MPC ***\n")
            scenario = extract_scenario(sl[count-1].name)
            evaluator.print_vessel_scores(vessel_id=0, file_prefix=f"standard_{scenario}", save_results=False)
            #evaluator.plot_trajectories(0, save_fig=True, filename=f"{scenario}_sbmpc_traj.svg")

        count += 1
                           
def simulate_and_evaluate_rl_sbmpc(eval_scenario_folder):
    tuneable_params = ["Q_", "P_", "K_CHI_"]
    
    action_kwargs = {
        "sbmpc_param_list": tuneable_params
    }

    env_id = "COLAVEnvironment-v0"
    seed = 10

    env_config = {
        "scenario_file_folder": eval_scenario_folder,
        "render_mode": "rgb_array",
        "seed": seed,
        "verbose": True,
        "rewarder_class": rewards.SBMPCRewarder,
        "action_type_class": actions.SBMPCParameterSettingAndSolverAction,
        "action_kwargs": action_kwargs
    }

    root = pathlib.Path(__file__).parents[1]

    #tuneable_params = ["Q_", "P_", "K_CHI_", "KAPPA_", "K_DCHI_SB_", "K_DCHI_P_"]
    tuneable_params = ["K_COLL_", "Q_"]
    #tuneable_params = ["K_COLL_", "KAPPA_"]
    action_kwargs = {
        "sbmpc_param_list": tuneable_params
    }
    env_config.update({"action_kwargs": action_kwargs})
    env = Monitor(gym.make(id=env_id, **env_config))
   
    #for mp in model_paths_full:
    #    evaluate_rl_model(env, mp)

    #for mp in os.listdir(root / "models"/"checkpoints"/"cp_gen5_scratch_ACR_noTrajCost_2025-04-01_18-20-51"):
    #    mp = mp.split(".zip")[0]
    #    mp = root / "models"/"checkpoints"/"cp_gen5_scratch_ACR_noTrajCost_2025-04-01_18-20-51"/mp
    #    evaluate_rl_model(env, mp)
   
    #mp = root / "models/best_models/gen2_2025-03-17_15-32-14/best_model" # gen2_best
    #mp = root / "models/checkpoints/cp_cp_2025-03-17_15-32-14/cp_cp_ts_150k_lr_0.0003_2025-03-17_15-32-14_1050000_steps" # gen2_last
    #mp = root / "models/best_models/gen3_2025-03-18_12-18-16/best_model" # gen3_best
    #mp = root / "models/checkpoints/cp_gen3_2025-03-18_12-18-16/cp_gen3_2025-03-18_12-18-16_1200000_steps" # gen3_last
    #mp = root / "models/checkpoints/cp_gen4_2025-03-19_12-25-56/cp_gen4_2025-03-19_12-25-56_1350000_steps" # gen4_last
    #mp = root / "models/checkpoints/cp_gen4_continued_2025-03-20_16-06-57/cp_gen4_continued_2025-03-20_16-06-57_1450000_steps" # gen4_continued
    #mp = root / "models/best_models/gen3_training_set_from_scratch_2025-03-21_13-36-40/best_model" # gen3_training_set_from_scratch_best
    #mp = root / "models/checkpoints/cp_gen3_training_set_from_scratch_2025-03-21_13-36-40/cp_gen3_training_set_from_scratch_2025-03-21_13-36-40_1500000_steps" # gen3_training_set_from_scratch_last
    #mp = root / "models/best_models/gen4_training_set_from_scratch_2025-03-21_13-38-12/best_model" # gen4_training_set_from_scratch_best
    #mp = root / "models/checkpoints/cp_gen4_training_set_from_scratch_2025-03-21_13-38-12/cp_gen4_training_set_from_scratch_2025-03-21_13-38-12_1500000_steps" # gen4_training_set_from_scratch_last
    #mp = root / "models/best_models/gen4_scratch_actionChatterCost_2025-03-27_15-36-14/best_model" # gen4_ActionChatterRewarder_best
    #mp = root / "models/best_models/gen5_2025-03-26_20-34-59/best_model" # gen5_best
    #mp = root / "models/checkpoints/cp_gen5_2025-03-26_20-34-59/cp_gen5_2025-03-26_20-34-59_300000_steps" # gen5_last
    #mp = root / "models/checkpoints/cp_gen6_2025-03-28_16-17-16/cp_gen6_2025-03-28_16-17-16_450000_steps" # gen6
    #mp = root / "models/best_models/gen4_scratch_actionChatterCost_2025-03-27_15-36-14/best_model" # gen4_ActionChatterRewarder
    #mp = root / "models/checkpoints/cp_gen5_ACR_2025-04-01_11-55-28/cp_gen5_ACR_2025-04-01_11-55-28_1400000_steps" # gen5_ACR

    #mp = root / "models/best_models/gen5_ACR_TrajCost_1.0_2025-05-13_15-12-01/best_model" # gen5_scratch_ACR_TrajCost_1.0_300k
    #mp = root / "models/checkpoints/cp_gen5_ACR_TrajCost_1.0_2025-05-13_15-12-01/cp_gen5_ACR_TrajCost_1.0_2025-05-13_15-12-01_300000_steps" # gen5_scratch_ACR_TrajCost_1.0_300k_last
    
    # BEST MODEL (noTraj)
    #mp = root / "models/best_models/gen5_scratch_ACR_noTrajCost_300k_2025-04-01_18-20-51/best_model" # gen5_scratch_ACR_noTrajCost_300k
    #mp = root / "models/best_models/noTTR_noTimeObs_300k_2025-05-15_23-39-08/best_model" # noTTR_noTimeObs_300k

    #mp = root / "models/best_models/gen5_noTrajCost_stageCostBugFix_300k_2025-05-03_23-15-15/best_model" # gen5_noTrajCost_stageCostBugFix_300k

    #mp = root / "models/best_models/rho_course_dev_0.1_300k_2025-05-08_22-18-03/best_model" # rho_course_dev_0.1_300k
    #mp = root / "models/best_models/rho_d2goal_1.0_300k_2025-05-08_22-16-12/best_model.zip" # rho_d2goal_1.0_300k

    #mp = root / "models/best_models/gen5_noTrajCost_300k_duplicate_2025-04-28_16-01-58/best_model" # noTrajCost_300k_duplicate
    #mp = root / "models/best_models/gen5_noTrajCost_300k_doubleDuplicate_2025-04-28_16-02-46/best_model" # noTrajCost_300k_doubleDuplicate

    #mp = root / "models/best_models/gen5_noTrajCost_new_ranges_300k_2025-04-25_14-31-13/best_model" # noTrajCost_new_ranges_retrained_300k
    #mp = root / "models/best_models/gen5_noTrajCost_noQ_noP_300k_2025-04-25_14-34-46/best_model" # noTrajCost_noQ_noP_300k

    #mp = root / "models/best_models/gen5_noTrajCost_300k_lower_increments_2025-04-29_14-18-10/best_model" # noTrajCost_300k_lower_increments
    #mp = root / "models/best_models/gen5_noTrajCost_300k_lower_increments_setup2_2025-04-29_16-08-07/best_model" # noTrajCost_300k_lower_increments_setup2
    
    #mp = root / "models/best_models/TrajCost_0.1_300k_2025-05-06_22-27-33/best_model" # TrajCost_0.1_300k
    #mp = root / "models/best_models/TrajCost_0.105_2025-05-13_15-05-51/best_model" # TrajCost_0.105_300k
    #mp = root / "models/best_models/TrajCost_0.108_2025-05-19_20-08-56/best_model" # TrajCost_0.108_300k

    # BEST MODEL (TrajCost_0.11)
    #mp = root / "models/best_models/TrajCost_0.11_2025-05-13_15-09-15/best_model" # TrajCost_0.11_300k
    #mp = root / "models/best_models/TrajCost_0.11_ACR_0.5_S_safety_r_weight_5.0_2025-05-16_14-22-36/best_model" # TrajCost_0.11_S_safety_r_weight_5.0_300k
    #mp = root / "models/best_models/TrajCost_0.12_2025-05-12_17-06-43/best_model" # TrajCost_0.12_300k
    #mp = root / "models/best_models/TrajCost_0.15_2025-05-12_17-11-53/best_model" # TrajCost_0.15_300k

    #mp = root / "models/best_models/TrajCost_0.12_ACR_1.0_2025-05-13_15-07-39/best_model" # TrajCost_0.12_ACR_1.0_300k
    #mp = root / "models/best_models/TrajCost_0.13_ACR_1.0_2025-05-16_14-20-28/best_model" # TrajCost_0.13_ACR_1.0_300k
    #mp = root / "models/best_models/TrajCost_0.15_ACR_1.0_2025-05-16_14-20-49/best_model" # TrajCost_0.15_ACR_1.0_300k
    #mp = root / "models/best_models/TrajCost_0.17_ACR_1.0_2025-05-16_14-21-18/best_model" # TrajCost_0.17_ACR_1.0_300k
    #mp = root / "models/best_models/TrajCost_0.2_ACR_1.0_300k_2025-05-15_11-49-13/best_model" # TrajCost_0.2_ACR_1.0_300k
    #mp = root / "models/best_models/TrajCost_0.5_ACR_1.0_300k_2025-05-15_11-47-44/best_model" # TrajCost_0.5_ACR_1.0_300k
    #mp = root / "models/best_models/TrajCost_1.0_ACR_1.0_300k_2025-05-15_11-50-22/best_model" # TrajCost_1.0_ACR_1.0_300k

    #mp = root / "models/best_models/TrajCost_10e-3_no_r_ram_300k_2025-05-05_19-22-49/best_model" # TrajCost_10e-3_no_r_ram_300k
    #mp = root / "models/best_models/noTrajCost_noRAMCost_300k_2025-05-06_12-09-58/best_model" # noTrajCost_noRAMCost_300k
    #mp = root / "models/best_models/halved_RAMCost_300k_2025-05-06_22-25-55/best_model" # halved_RAMCost_300k

    #mp = root / "models/best_models/test_land_avoidance_2025-05-06_20-50-28/best_model" # test_land_avoidance (K_COLL and Q tuning, GroundingRewarder, 10k steps)
    #mp = root / "models/best_models/test_land_avoidance_2_2025-05-06_21-22-18/best_model" # test_land_avoidance_2 (K_COLL and Q tuning, GroundingRewarder, EvalRewarder, 10k steps)
    #mp = root / "models/best_models/test_grounding_GroundingRewarder_200k_2025-05-06_22-19-09/best_model" # test_grounding_GroundingRewarder_200k
    #mp = root / "models/best_models/test_grounding_avoidance_3_scenarios_200k_2025-05-10_15-50-17/best_model" # test_grounding_avoidance_3_scenarios_200k
    #mp = root / "models/best_models/test_land_avoidance_4_scenarios_300k_2025-05-15_11-52-40/best_model" # test_land_avoidance_4_scenarios_300k
    #mp = root / "models/best_models/test_land_avoidance_4_scenarios_obs_range_500_300k_2025-05-19_19-51-44/best_model" # test_land_avoidance_4_scenarios_obs_range_500_300k
    #mp = root / "models/best_models/test_land_avoidance_4_scenarios_obs_range_300_300k_2025-05-19_19-51-16/best_model" # test_land_avoidance_4_scenarios_obs_range_300_300k
    #mp = root / "models/best_models/GHA_4_scenarios_2025-05-21_13-35-10/best_model" # GHA_4_scenarios_300k
    
    # BEST MODEL FOR GROUNDING AVOIDANCE (only K_COLL and Q tuning)
    mp = root / "models/best_models/test_land_avoidance_4_scenarios_obs_range_200_300k_2025-05-19_19-50-30/best_model" # test_land_avoidance_4_scenarios_obs_range_200_300k
    #mp = root / "models/best_models/land_avoidance_6_scenarios_KAPPA_2025-05-22_10-51-19/best_model" # land_avoidance_6_scenarios_KAPPA_300k

    evaluate_rl_model(env, mp)
    
    env.close()

def evaluate_rl_model(env, model_path) -> None:
    model = SAC.load(model_path, env)
    
    parts = os.path.normpath(model_path).split(os.sep)
    model_name = parts[-1]
    if model_name == "best_model":
        model_name = parts[-2]
    #for p in parts:
    #    if p.startswith('SAC') or p.startswith('gen') or p.startswith('cp'):
    #        model_name = p
    #        break
        
    print(f"\nEvaluating model: {model_name}")    
    print("\n*** RL tuned SB-MPC ***\n")
    
    eval_files = [f for f in env.unwrapped.scenario_file_folder.iterdir() if f.is_file()]
    none_imazu_files = [f for f in eval_files if "imazu" not in f.name]
    none_imazu_files = ["eval" + f.name.split("eval", 1)[-1] for f in none_imazu_files]
    none_imazu_files = [f.split(".yaml")[0] for f in none_imazu_files]

    imazu_files = [f for f in eval_files if "imazu" in f.name]
    imazu_files = ["eval_imazu" + f.name.split("imazu", 1)[-1] for f in imazu_files]
    imazu_files = [f.split(".yaml")[0] for f in imazu_files]

    sl = none_imazu_files + imazu_files

    # To save results: set save_results=True in EvaluatorRewarder in reward.py
    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=len(sl), deterministic=True
    )
    print(f"\nModel: {model_name}\nMean Reward: {mean_reward}, Std Reward: {std_reward}\n")

    # Add model to filename(s)
    for scenario_name in sl:
        old_filename = f"{scenario_name}.xlsx"
        new_filename = f"{model_name}_{old_filename}"
        if os.path.exists(old_filename):
            os.rename(old_filename, new_filename)
            #print("Renamed file: " + old_filename + " to " + new_filename)
        else:
            print(f"File {old_filename} does not exist")
        
        old_filename = f"{scenario_name}_sbmpc_param_log.csv"
        scenario_name = extract_scenario(scenario_name)
        renamed_old_filename = f"{scenario_name}_sbmpc_param_log.csv"
        new_filename = f"{model_name}_{renamed_old_filename}"
        
        if os.path.exists(old_filename):
            os.rename(old_filename, new_filename)
            #print("Renamed file: " + old_filename + " to " + new_filename)
        else:
            print(f"File {old_filename} does not exist")

if __name__ == "__main__":
    
    root = pathlib.Path(__file__).parents[1]
    scenario_folder = root/"scenario"
    eval_scenario_foler = scenario_folder/"eval"

    simulate_and_evaluate_sbmpc(eval_scenario_foler)
    #simulate_and_evaluate_rl_sbmpc(eval_scenario_foler)

    # Checklist for evaluating a new model and saving the results:
    # 1. Make sure to set save_results=True in EvaluatorRewarder in reward.py
    # 2. Set eval_scenario_folder = scenario_folder/"eval"
    # 3. Set the model path in the simulate_and_evaluate_rl_sbmpc function
    # 4. Set render_mode = "none" in env_config