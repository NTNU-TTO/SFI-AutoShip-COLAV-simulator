from colav_simulator.scenario_management import ScenarioGenerator

if __name__ == "__main__":

    scenario_generator = ScenarioGenerator()

    ship_list = scenario_generator.generate()

    print("done")
