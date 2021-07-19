from ship_models import telemetron, random_ship_model

def create_ship_model(ship_model_name='random'):
    if ship_model_name == 'telemetron':
        return telemetron.Telemetron()
    else:
        return random_ship_model.Random_ship_model()
