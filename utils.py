from ship_models import telemetron, random_ship_model
import math

def create_ship_model(ship_model_name='random'):
    if ship_model_name == 'telemetron':
        return telemetron.Telemetron()
    else:
        return random_ship_model.Random_ship_model()

def wrap_to_pi(angle):
    #wraps the angle to [0,2*pi)
    return math.fmod(angle + 2 * math.pi, 2 * math.pi)
