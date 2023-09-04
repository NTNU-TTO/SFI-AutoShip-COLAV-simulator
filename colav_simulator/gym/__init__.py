from gymnasium.envs.registration import register

register(
    id="COLAVEnvironment-v0",
    entry_point="colav_simulator.gym.envs:COLAVEnvironment",
)
