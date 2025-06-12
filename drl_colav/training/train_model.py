import numpy as np
import datetime as dt
import time
import copy
import os

import torch
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback
from colav_simulator.gym.environment import COLAVEnvironment
from colav_simulator.scenario_generator import ScenarioGenerator
import gymnasium as gym
from stable_baselines3 import SAC, PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy 
import colav_simulator.gym.reward as rewards
import colav_simulator.gym.action as actions
import pathlib

def make_env(env_id, env_config, seed, rank=0):
    def _init():
        env = gym.make(env_id, **env_config)
        env.seed(seed + rank)
        env = Monitor(env)
        return env
    return _init

# Learning Rate Schedulers
def exponential_schedule(initial_value, decay_rate=0.99):
    def schedule(progress_remaining):
        return initial_value * (decay_rate ** (1 - progress_remaining))
    return schedule

class ProgressCallback(BaseCallback):
    def __init__(self, total_timesteps, verbose=1):
        super(ProgressCallback, self).__init__(verbose)
        self.num_timesteps = 0
        self.total_timesteps = total_timesteps
        self.start_time = None
        
    def _on_training_start(self) -> None:
        print(f"\n*** Starting training ***\n")
        self.start_time = time.time()
        
    def _on_step(self) -> bool:
        
        if self.start_time is None:
            self.start_time = time.time()
        progress = self.num_timesteps / self.total_timesteps
        elapsed_time = int(time.time() - self.start_time)
        minutes, seconds = divmod(elapsed_time, 60)
        hours, minutes = divmod(minutes, 60)
        eta = (elapsed_time / progress) - elapsed_time
        eta_minutes, _ = divmod(eta, 60)
        eta_hours, eta_minutes = divmod(eta_minutes, 60)

        hours_print = ""
        if hours > 0:
            hours_print = f"{hours} h "
        eta_hours_print = ""
        if eta_hours > 0:
            eta_hours_print = f"{int(eta_hours)} h "
        
        print(f"Step: {self.num_timesteps} of {self.total_timesteps} "
              f"({progress:.2%}) - Elapsed time: {hours_print}{minutes} m {seconds} s - "
              f"{eta_hours_print}{int(eta_minutes)}m left", end="\r"
        )
        return True
    
    def _on_training_end(self) -> None:
        elapsed_time = int(time.time() - self.start_time)
        minutes, seconds = divmod(elapsed_time, 60)
        hours, minutes = divmod(minutes, 60)
        print(f"\n*** Training completed in {hours} h {minutes} m {seconds} s ***\n")

        
def main():

    model_name = "TrajCost_0.11_WITH_LAND_AVOIDANCE"
          
    # Training parameters
    
    algorithm = "SAC"
    device = "cpu" # "cuda" for GPU, "cpu" for CPU
      
    total_timesteps = 300000 #1500000
    learning_rate = 3e-04
    batch_size = 256
    
    eval_freq = 50000
    save_freq = 50000 #50000
    
    # Folders and paths
    
    root = pathlib.Path(__file__).parent.parent.resolve()
    scenario_folder = root/"scenario"
    training_scenario_folder = scenario_folder/"training_land_Obs"
    training_land_scenario_folder = scenario_folder/"training_land"
    eval_scenario_folder = scenario_folder/"eval"
    test_folder = scenario_folder/"test"
    logs_folder = root/"logs"
    eval_logs_folder = logs_folder/"eval_logs"
    tb_log_path = logs_folder/"tb_logs"
    date_time_now = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    models_folder = root/"models"
    cp_models_folder = models_folder/"checkpoints"
    checkpoints_path = cp_models_folder/f"cp_{model_name}_{date_time_now}"
    best_models_folder = models_folder/"best_models"
    model_path = best_models_folder/f"{model_name}_{date_time_now}"
    
    # Environment setup
    
    render_on = False

    parameters_to_tune = ['K_COLL_', 'Q_', 'P_', 'K_CHI_', 'KAPPA_', 'K_DCHI_SB_', 'K_DCHI_P_'] #['K_COLL_', 'Q_']
    action_kwargs = {"sbmpc_param_list": parameters_to_tune}
    
    render_mode = "rgb_array" if render_on else "none"
    render_update_rate = 1 if render_on else 0
    reload_map = render_on
    
    env_id = "COLAVEnvironment-v0"
    seed = 42
    
    env_config = {
        "scenario_file_folder": eval_scenario_folder, #training_scenario_folder,
        "render_mode": render_mode,
        "render_update_rate": render_update_rate,
        "reload_map": reload_map,
        "seed": seed,
        "verbose": False,
        "rewarder_class": rewards.SBMPCRewarder,
        "action_type_class": actions.SBMPCParameterSettingAndSolverAction,
        "action_kwargs": action_kwargs
    }
            
    #env = make_vec_env(lambda: gym.make(env_id, **env_config), n_envs=12)
    #env = SubprocVecEnv([make_env(env_id, env_config, seed, i) for i in range(4)])
    env = Monitor(gym.make(env_id, **env_config))
    
    if algorithm == "SAC":
        model = SAC(
            "MultiInputPolicy",
            env,
            device = device,
            learning_rate = learning_rate, #exponential_schedule(learning_rate),
            ent_coef = 0.1,
            seed=seed+10,
            batch_size = batch_size,
            verbose = 1,
            tensorboard_log = tb_log_path
        )
    elif algorithm == "PPO":
        model = PPO(
            "MultiInputPolicy",
            env,
            learning_rate = learning_rate,
            batch_size = batch_size,
            verbose = 1,
            tensorboard_log = tb_log_path
        )
    # Load from checkpoint/continue training
    else:
        latest_checkpoint = models_folder/"checkpoints/cp_gen5_scratch_ACR_noTrajCost_2025-04-01_18-20-51/cp_gen5_scratch_ACR_noTrajCost_2025-04-01_18-20-51_300000_steps"
        print(f"Loading model from {latest_checkpoint}")
        model = SAC.load(latest_checkpoint, env=env, device=device)
    
    eval_env_config = copy.deepcopy(env_config)
    eval_env_config.update({
        "scenario_file_folder": eval_scenario_folder, #eval_scenario_folder,
        "seed": seed+20,    # Different seed for evaluation environment
    })
    
    eval_env = Monitor(gym.make(env_id, **eval_env_config))
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path = model_path,
        log_path = eval_logs_folder,
        eval_freq = eval_freq,
        deterministic = True,
        verbose = 0
    )
    
    if not os.path.exists(checkpoints_path):
        os.makedirs(checkpoints_path)
    
    checkpoint_callback = CheckpointCallback(
        save_freq = save_freq,
        save_path = checkpoints_path,
        name_prefix = f"cp_{model_name}_{date_time_now}"
    )
    
    # Callback for printing progress during training
    progress_callback = ProgressCallback(total_timesteps)
    
    # Train the model
    # Adjust total_timesteps if resuming from a checkpoint
    if model.num_timesteps > 0:
        progress_callback.total_timesteps += model.num_timesteps

    """
    obs = env.reset()
    for i in range(100):
        action = env.action_space.sample()  # Sample random action
        obs, reward, done, trunc, info = env.step(action)
        distance_bearing_to_hazards = obs.get('DistanceBearingToHazardObservation', None)
        #print(distance_bearing_to_hazards)
        if done or trunc:
            env.close()
            break
    """

    model.learn(
        total_timesteps = total_timesteps,
        tb_log_name = f"log_{algorithm}_ts_{total_timesteps//1000}k_lr_{learning_rate}_{date_time_now}",
        reset_num_timesteps = False,
        callback = [eval_callback, checkpoint_callback, progress_callback]
    )
    
    print("\nEvaluating final model...\n")    
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=1,
        deterministic=True
    )
    print(f"Mean reward: {mean_reward}, Std: {std_reward}")
    
    env.close()

if __name__ == "__main__":
    main()