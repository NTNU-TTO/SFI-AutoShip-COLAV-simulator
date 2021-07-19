import configparser
import pathlib

cwd = pathlib.Path.cwd()
path = cwd / 'config_simulator.ini'
path_shp = cwd / 'data' / 'external'
path_ctrl = cwd / 'config_control.ini'


def read_settings(category='DEFAULT', path=path):
    settings = {}
    config = configparser.ConfigParser()
    config.read(path, encoding='utf8')
    for key, value in config[category].items():
        settings[key] = [v.strip(' ') for v in value.split(',')]
    return settings


def parse(key, defaults):
    default = defaults.get(key, None)
    if default is None:
        raise ValueError(
            f"Missing input parameter: '{key}': was not provided, "
            f"and could not located in the current configuration file.")
    return default


def validate(key, value, v_type, sub_type=None, length=None):
    if isinstance(value, list) or isinstance(value, tuple):
        if not all([isinstance(v, sub_type) for v in value]):
            raise ValueError(f"Invalid input format: " f"'{key}' should be a {v_type.__name__} of {sub_type}.")
        if length is not None and len(value) != length:
            raise ValueError(f"Invalid input format: " f"'{key}' should be a {v_type.__name__} of length {length}.")
    else:
        if not isinstance(value, v_type):
            raise ValueError(f"Invalid input format: " f"'{key}' should have type {v_type}.")


def verify_directory_exists(dir_path):
    if not (path_shp / dir_path).is_dir():
        raise FileNotFoundError(f"Folder {dir_path} not found at:\r\n{path_shp}.")


def confirm_input(user, key, check, defaults, i_0, i_1):
    '''
    arg:dict, value, data type, dict, index 0, index 1
    Check if the user input is correct. If it is wrong it is changed with the default parameter.
    i_0 always = 0, i_1 = 1 only when parameter has length two else i_1 = 0
    ret: values from dict as tuple
    '''
    try:
        if i_1:
            user[key] = check(user[key][i_0]), check(user[key][i_1])
            if check == int and user[key][i_0] < 0 or user[key][i_1] < 0:
                print(key,"must be strictly positive.")
                raise ValueError(f"must be strictly positive.")
        else:
            user[key] = check(user[key][i_0])
            if check == int and user[key] < 0:
                print(key, "must be strictly positive.")
                raise ValueError(f"must be strictly positive.")
            if key == 'scenario_num' and (not 0 <= user[key] <= 5):
                print(key, "must be set to 0, 1, 2, 3, 4 or 5.")
                raise ValueError(f"must be set to 0, 1, 2, 3, 4 or 5.")
            if key == 'show_waypoints' and (not 0 <= user[key] <= 1):
                print(key, "must be set to 0 or 1.")
                raise ValueError(f"must be set to 0 or 1.")
            if key == 'file':
                user[key] = [user[key]]
                validate(key, user[key], list, str)
                verify_directory_exists(user[key][0])
    except:
        default = parse(key, defaults)
        if i_1:
            user[key] = check(default[i_0]), check(default[i_1])
            if key == 'size' or key == 'center':
                default_size = parse('size', defaults)
                user['size'] = check(default_size[i_0]), check(default_size[i_1])
                print('Using default setting for size with value (', check(default_size[i_0]), ',',
                      check(default_size[i_1]), ')')
                default_center = parse('center', defaults)
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


def read_config():
    user = read_settings(category='USER')
    defaults = read_settings(category='DEFAULT')

    key = 'size'
    confirm_input(user, key, int, defaults, i_0=0, i_1=1)
    validate(key, user[key], tuple, int, 2)

    key = 'center'
    confirm_input(user, key, int, defaults, i_0=0, i_1=1)
    validate(key, user[key], tuple, int, 2)

    key = 'file'
    confirm_input(user, key, str, defaults, i_0=0, i_1=0)
    validate(key, user[key], list, str)
    verify_directory_exists(user[key][0])

    key = 'new_data'
    confirm_input(user, key, eval, defaults, i_0=0, i_1=0)
    validate(key, user[key], bool)

    key = 'time_start'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'time_step'
    confirm_input(user, key, int, defaults,  i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'time_end'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'waypoint_num'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'scenario_num'
    confirm_input(user, key, int, defaults,  i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'ship_model_name'
    confirm_input(user, key, str, defaults, i_0=0, i_1=0)
    validate(key, user[key], str)

    key = 'os_max_speed'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'ts_max_speed'
    confirm_input(user, key, int, defaults,  i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'ship_num'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'show_waypoints'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'own_ship_max_acc'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'own_ship_max_turn_rate'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)


    settings = tuple(user.values())
    print('')
    return settings

def read_control_config():
    user = read_settings(category='USER', path=path_ctrl)
    defaults = read_settings(category='DEFAULT', path=path_ctrl)

    key = 'lookahead_distance'
    confirm_input(user, key, int, defaults, i_0=0, i_1=0)
    validate(key, user[key], int)

    key = 'radius_of_acceptance'
    confirm_input(user, key, int, defaults,  i_0=0, i_1=0)
    validate(key, user[key], int)


    settings = tuple(user.values())
    print('')
    return settings