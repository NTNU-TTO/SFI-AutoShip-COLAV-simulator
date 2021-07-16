from ship_models import telematron, random_ship_model

def create_ship_model(ship_model_name):
    if ship_model_name == 'telematron':
        return telematron.Telematron()
    if ship_model_name == 'random':
        return random_ship_model.Random_ship_model()

    # Default returns random ship
    return random_ship_model.Random_ship_model()