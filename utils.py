from numpy import pi
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
    #return normalize_angle(angle)

def normalize_angle(angle):
	#wraps the angle to [-pi, pi)
	while angle <= -math.pi:
		angle += 2*math.pi
	while angle > math.pi:
		angle -= 2*math.pi

	return angle


def normalize_angle_diff(angle, angle_ref):

	diff = angle_ref - angle

	if (diff > 0):
		new_angle = angle +(diff - math.fmod(diff, 2*math.pi))
	else:
		new_angle = angle + (diff + math.fmod(-diff, 2*math.pi))
	
	diff = angle_ref - new_angle
	if (diff > math.pi):
		new_angle += 2*math.pi
	elif (diff < -math.pi):
		new_angle -= 2*math.pi
	return new_angle

