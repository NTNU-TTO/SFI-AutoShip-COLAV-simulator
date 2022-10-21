"""Contains functionality for reading configuration files for customizing the simulation"""

import configparser
import pathlib

root = pathlib.Path.cwd().parents[2]
package = root / 'colav_simulator'
config = root / 'config'
simulator_config = config / 'simulator.ini'
ships_config = config / 'ships.ini'
new_scenario_config = config / 'new_scenario.ini'

# def verify_directory_exists(dir_path):
#     if not (path_shp / dir_path).is_dir():
#         raise FileNotFoundError(f"Folder {dir_path} not found at:\r\n{path_shp}.")

def read_settings(file_name=simulator_config, category='DEFAULT'):
    """Reads settings from specified category in input file

    Args:
        file_name (str, optional):  The name of the file to read. Defaults to simulator_config.
        category (str, optional): Name of the category to read. Defaults to 'DEFAULT'.

    Returns:
        dict: Dictionary containing the settings
    """
    settings = {}
    sconfig = configparser.ConfigParser()
    sconfig.read(file_name, encoding='utf8')
    for key, value in sconfig[category].items():
        settings[key] = [v.strip(' ') for v in value.split(',')]
    return settings


def parse_key(key, defaults):
    """Reads key from default settings dictionary

    Args:
        key (str): The name of the key to read
        defaults (dict): Dictionary of settings to read from

    Raises:
        ValueError: On missing input key

    Returns:
        Value: Value of settings key
    """
    default = defaults.get(key, None)
    if default is None:
        raise ValueError(
            f"Missing input parameter: '{key}': was not provided, "
            f"and could not located in the current configuration file.")
    return default


def validate(key, value, v_type, sub_type=None, length=None):
    """_summary_

    _extended_summary_

    Args:
        key (_type_): _description_
        value (_type_): _description_
        v_type (_type_): _description_
        sub_type (_type_, optional): _description_. Defaults to None.
        length (_type_, optional): _description_. Defaults to None.

    Raises:
        ValueError: On invalid input formats.
    """
    if isinstance(value, list) or isinstance(value, tuple):
        if not all([isinstance(v, sub_type) for v in value]):
            raise ValueError(f"Invalid input format: " f"'{key}' should be a {v_type.__name__} of {sub_type}.")
        if length is not None and len(value) != length:
            raise ValueError(f"Invalid input format: " f"'{key}' should be a {v_type.__name__} of length {length}.")
    else:
        if not isinstance(value, v_type):
            raise ValueError(f"Invalid input format: " f"'{key}' should have type {v_type}.")

def confirm_input(user, key, check, defaults, i_0, i_1):
    """Check if the user input is correct. If it is wrong it is changed with the default parameter.
    i_0 is always = 0, i_1 = 1 only when parameter has length two else i_1 = 0
    ret: values from dict as tuple

    Args:
        user (dict): Dictionary with user settings
        key (str): _description_
        check (_type_): _description_
        defaults (dict): Dictionary with default settings
        i_0 (_type_): _description_
        i_1 (_type_): _description_

    Raises:
        ValueError: On wrong format of keys.
    """
    try:
        if i_1:
            user[key] = check(user[key][i_0]), check(user[key][i_1])
            if check == int and user[key][i_0] < 0 or user[key][i_1] < 0:
                print(key,"must be strictly positive.")
                raise ValueError("Must be strictly positive.")
        else:
            user[key] = check(user[key][i_0])
            if check == int and user[key] < 0:
                print(key, "must be strictly positive.")
                raise ValueError("Must be strictly positive.")
            if key == 'scenario_num' and (not 0 <= user[key] <= 5):
                print(key, "must be set to 0, 1, 2, 3, 4 or 5.")
                raise ValueError("Must be set to 0, 1, 2, 3, 4 or 5.")
            if key == 'show_waypoints' and (not 0 <= user[key] <= 1):
                print(key, "must be set to 0 or 1.")
                raise ValueError("Must be set to 0 or 1.")
            if key == 'file':
                user[key] = [user[key]]
                validate(key, user[key], list, str)
                #verify_directory_exists(user[key][0])
    except: # pylint: disable=bare-except
        default = parse_key(key, defaults)
        if i_1:
            user[key] = check(default[i_0]), check(default[i_1])
            if key == 'size' or key == 'center':
                default_size = parse_key('size', defaults)
                user['size'] = check(default_size[i_0]), check(default_size[i_1])
                print('Using default setting for size with value (', check(default_size[i_0]), ',',
                      check(default_size[i_1]), ')')
                default_center = parse_key('center', defaults)
                user['center'] = check(default_center[i_0]), check(default_center[i_1])
                print('Using default setting for center with value (', check(default_center[i_0]), ',',
                      check(default_center[i_1]), ')')
                user['new_data'] = [eval('True')]
            else:
                print('Using default setting for', key, 'with value ', user[key])
        else:
            user[key] = check(default[i_0])
            if key == 'file':
                user[key] = check(default[i_0])
                user[key] = [user[key]]
                user['new_data'] = [eval('True')]
                print(f"Folder {user[key][0]} not found at:\r\n{path_shp}.")
            print('Using default setting for', key, 'with value ', user[key])


def read_config(file_name=simulator_config):
    """Reads entire configuration from file, returns dictionary of simulator/scenario/viz
    :param file_name: name of the configuration file
    :return: dictionary of configuration settings
    :rtype: dict
    """
    user = read_settings(file_name, category='USER')
    defaults = read_settings(file_name, category='DEFAULT')

    key = 'size'
    confirm_input(user, key, int, defaults, i_0=0, i_1=1)
    validate(key, user[key], tuple, int, 2)

    key = 'center'
    confirm_input(user, key, int, defaults, i_0=0, i_1=1)
    validate(key, user[key], tuple, int, 2)

    key = 'file'
    confirm_input(user, key, str, defaults, i_0=0, i_1=0)
    validate(key, user[key], list, str)
    #verify_directory_exists(user[key][0])

    key = 'new_data'
    confirm_input(user, key, eval, defaults, i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'time_start'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'time_step'
    confirm_input(user, key, float, defaults,  i_0=0, i_1=0)
    validate(key, user[key], float)

    key = 'time_end'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'run_all_scenarios'
    confirm_input(user, key, eval, defaults, i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'new_scenario'
    confirm_input(user, key, eval, defaults, i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'scenario_file'
    confirm_input(user, key, str, defaults, i_0=0, i_1=0)
    validate(key, user[key], str)

    key = 'colav_all_ships'
    confirm_input(user, key, eval, defaults, i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'save_animation'
    confirm_input(user, key, eval, defaults, i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'show_animation'
    confirm_input(user, key, eval, defaults, i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'show_waypoints'
    confirm_input(user, key, eval, defaults, i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'evaluate_results'
    confirm_input(user, key, eval, defaults, i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'radius_preferred_cpa'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'radius_minimum_acceptable_cpa'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'radius_near_miss_encounter'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'radius_collision'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'radius_colregs_2_max'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'radius_colregs_3_max'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'radius_colregs_4_max'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    settings = tuple(user.values())
    print('')
    return settings


def read_ship_config(section_name):
    """_summary_

    _extended_summary_

    Args:
        section_name (_type_): _description_

    Returns:
        _type_: _description_
    """
    # section_name: f'SHIP{i}' or 'DEFAULT'
    config_parser = configparser.ConfigParser()
    config_parser.read(ships_config, encoding='utf8')
    defaults = read_settings(file_name=ships_config, category='DEFAULT')
    if config_parser.has_section(section_name):
        user = read_settings(file_name=ships_config, category=section_name)
    else:
        user = defaults.copy()

    key = 'ship_model'
    confirm_input(user, key, str, defaults, i_0=0, i_1=0)
    validate(key, user[key], str)

    ### Radar parameters ###
    key = 'radar_active'
    confirm_input(user, key, bool, defaults,  i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'radar_meas_rate'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'radar_sigma_z'
    confirm_input(user, key, float, defaults, i_0=0, i_1=0)
    validate(key, user[key], float)

    ### AIS parameters ###
    key = 'ais_active'
    confirm_input(user, key, bool, defaults,  i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'ais_meas_rate'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'ais_sigma_z'
    confirm_input(user, key, float, defaults, i_0=0, i_1=0)
    validate(key, user[key], float)

    key = 'ais_loss_prob'
    confirm_input(user, key, float, defaults,  i_0=0, i_1=0)
    validate(key, user[key], float)

    ### LOS guidance parameters ###
    key = 'lookahead_distance'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'radius_of_acceptance'
    confirm_input(user, key, float, defaults, i_0=0, i_1=0)
    validate(key, user[key], float)


    settings = tuple(user.values())
    print('')
    return settings

def read_scenario_gen_config(file_name):
    """_summary_

    _extended_summary_

    Args:
        file_name (str): _description_. Defaults to new_scenario_config.

    Returns:
        _type_: _description_
    """
    user = read_settings(file_name, category='USER')
    defaults = read_settings(file_name, category='DEFAULT')

    key = 'num_waypoints'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'scenario_num'
    confirm_input(user, key, int, defaults,  i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'os_max_speed'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'ts_max_speed'
    confirm_input(user, key, int, defaults,  i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'num_ships'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    settings = tuple(user.values())
    print('')
    return settings
