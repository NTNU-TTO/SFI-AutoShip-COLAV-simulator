"""Contains functionality for reading configuration files used for customizing the simulation"""

import configparser

from . import paths as dcp


def verify_directory_exists(dir_name, dir_path=dcp.map_data):
    if not (dir_path / dir_name).is_dir():
        raise FileNotFoundError(f"Folder {dir_name} not found at:\r\n{dir_path}.")

def read_settings(file_name=dcp.simulator, category='DEFAULT'):
    settings = {}
    sconfig = configparser.ConfigParser()
    sconfig.read(file_name, encoding='utf8')
    if not sconfig.has_section(category):
        raise ValueError("Category not found in configuration file.")

    for key, value in sconfig[category].items():
        settings[key] = [v.strip(' ') for v in value.split(',')]
    return settings


def parse_key(key, settings):
    value = settings.get(key, None)
    if value is None:
        raise ValueError(
            f"Missing input parameter: '{key}': was not provided, "
            f"and could not located in the current configuration file.")
    return value


def validate(key, value, v_type, sub_type=None, length=None):
    """Validates type of value at given key to be equal to v_type, or sub_type if value is a list/tuple of len = length.

    Args:
        key (str): Name of parameter to check value for.
        value (_type_): Value of parameter in config at the given key.
        v_type (_type_): Type of value.
        sub_type (_type_, optional): Subtype if value is a list/tuple. Defaults to None.
        length (_type_, optional): Length if value is list/tuple. Defaults to None.

    Raises:
        ValueError: On wrong type of value at the given key.
    """
    if isinstance(value, list) or isinstance(value, tuple):
        if not all([isinstance(v, sub_type) for v in value]):
            raise ValueError(f"Invalid input format: " f"'{key}' should be a {v_type.__name__} of {sub_type}.")
        if length is not None and len(value) != length:
            raise ValueError(f"Invalid input format: " f"'{key}' should be a {v_type.__name__} of length {length}.")
    else:
        if not isinstance(value, v_type):
            raise ValueError(f"Invalid input format: " f"'{key}' should have type {v_type}.")

def validate_user_config(user, defaults):
    for key, value in defaults.items():
        if user[key] is None:
            user[key] = value


def read_simulator_config(file_name=dcp.simulator):
    """Reads configuration from file, returns dictionary of settings. If section_name is specified, only the subset of the configuration with name section_name is read.

    Args:
        file_name (str, optional): Name of the configuration file. Defaults to dcp.simulator.
        section_name (str, optional): Name of section in configuration settings to consider.

    Returns:
        dict: Dictionary of configuration settings
    """
    user = read_settings(file_name, category='USER')
    defaults = read_settings(file_name, category='DEFAULT')

    validate_user_config(user, defaults)

    settings = tuple(user.values())
    return settings


def read_ship_config(file_name=dcp.ships, section_name='SHIP1'):
    """Reads configuration settings from file_name for ship category with section name

    Args:
        file_name (str, optional): Absolute path to the ship configuration file. Defaults to dcp.ships.
        section_name (str, optional): Name of ship

    Returns:
        dict: Configuration settings as dictionary.
    """
    # section_name: f'SHIP{i}' or 'DEFAULT'
    defaults = read_settings(file_name, category='DEFAULT')
    try:
        user = read_settings(file_name, category=section_name)
    except NotFoundErr:
        return tuple(defaults.values())

    validate_user_config(user, defaults)
    return tuple(user.values())

def read_scenario_gen_config(file_name=dcp.new_scenario):
    """Reads scenario generation settings from configuration file.

    Args:
        file_name (str): Absolute path to the scenario configuration file. Defaults to dcp.new_scenario.

    Returns:
        tuple: Configuration settings as tuple of all parameter values.
    """
    user = read_settings(file_name, category='USER')
    defaults = read_settings(file_name, category='DEFAULT')

    validate_user_config(user, defaults)

    return tuple(user.values())
