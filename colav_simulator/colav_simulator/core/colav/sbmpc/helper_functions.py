import numpy as np
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import colav_simulator.common.math_functions as mf

def param_incr_to_unnorm_dict(
    action: np.ndarray, 
    current_params: np.ndarray,
    param_list: List[str],
    param_ranges: Dict[str, Any], 
    param_incr_ranges: Dict[str, Any]
) -> Dict[str, Union[float, np.ndarray]]:
    """Unnormalizes the DNN output parameter increment tensor, adds it to the current parameter values, and returns a dict with the updated parameter values."""

    params = {}

    for indx, param in enumerate(param_list):
        param_range = param_ranges[param]
        param_incr_range = param_incr_ranges[param]

        param_incr_norm = action[indx]
        param_incr_unnorm = mf.linear_map(param_incr_norm, (-1.0, 1.0), tuple(param_incr_range))

        current_param_value = current_params[indx]
        new_param_value = np.array(current_param_value + param_incr_unnorm)
        new_param_value = np.clip(new_param_value, param_range[0], param_range[1])

        params[param] = new_param_value.astype(np.float32)

    return params

def normalize_sbmpc_param(
    param_obs: np.ndarray,
    param_list: List[str],
    param_ranges: Dict[str, Any],
) -> np.ndarray:
    """Normalize the input parameter"""

    param_obs_norm = np.zeros_like(param_obs, dtype=np.float32)

    for indx, param_name in enumerate(param_list):
        param_range = param_ranges[param_name]

        param_obs_unnorm = param_obs[indx]
        param_obs_norm[indx] = mf.linear_map(param_obs_unnorm, tuple(param_range), (-1.0, 1.0))

    return param_obs_norm

def unnormalize_sbmpc_param(
    param_obs: np.ndarray,
    param_list: List[str],
    parameter_ranges: Dict[str, Any],
) -> np.ndarray:
    """Unnormalize the input parameter."""
    
    param_obs_unnorm = np.zeros_like(param_obs, dtype=np.float32)

    for indx, param_name in enumerate(param_list):
        param_range = parameter_ranges[param_name]
        
        param_obs_norm = param_obs[indx]
        param_obs_unnorm[indx] = mf.linear_map(param_obs_norm, (-1.0, 1.0), tuple(param_range))

    return param_obs_unnorm

def huber_loss(x_squared, delta = 1.0):
    return (np.sqrt(1.0 + x_squared / (delta**2)) - 1.0) * delta**2

def rate_cost(
    r, a, alpha_app, K_app, r_max: float, a_max: float
):
    q_chi = alpha_app[0] * r**2 + (1.0 - np.exp(-(r**2) / alpha_app[1]))
    q_chi_max = 1.0  # alpha_app[0] * r_max**2 + (1.0 - csd.exp(-(r_max**2) / alpha_app[1]))
    q_U = alpha_app[2] * a**2 + (1.0 - np.exp(-(a**2) / alpha_app[3]))
    q_U_max = 1.0  # alpha_app[2] * a_max**2 + (1.0 - csd.exp(-(a_max**2) / alpha_app[3]))
    course_cost = K_app[0] * q_chi / q_chi_max
    speed_cost = K_app[1] * q_U / q_U_max
    return course_cost + speed_cost, course_cost, speed_cost