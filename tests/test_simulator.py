import colav_simulator.core.colav.colav_interface as ci
from colav_simulator.scenario_generator import ScenarioGenerator
from colav_simulator.simulator import Simulator

if __name__ == "__main__":
    sbmpc_obj = ci.SBMPCWrapper()
    scenario_generator = ScenarioGenerator()
    scenario_data_list = scenario_generator.generate_configured_scenarios()
    simulator = Simulator()
    simulator.toggle_liveplot_visibility(True)
    output = simulator.run(scenario_data_list, colav_systems=[(0, sbmpc_obj)])
    print("done")
