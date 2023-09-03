from gymnasium.envs.registration import register

register(
    id="colav_env-v0",
    entry_point="colav_simulator.gym.envs:COLAVEnvironment",
)
