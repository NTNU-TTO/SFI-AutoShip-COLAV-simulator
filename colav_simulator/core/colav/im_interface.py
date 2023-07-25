"""
    colav_interface.py

    Summary:
        Contains the interface used by all COLAV planning algorithms that
        wants to be run with the COLAV simulator.

        To add a new COLAV planning algorithm:

        1: Import the algorithm in this file.
        2: Add the algorithm name as a type to the COLAVType enum.
        3: Add the algorithm as an optional entry to the LayerConfig class.
        4: Create a new wrapper class for your COLAV algorithm,
        which implements (inherits as this is python) this interface. It should take in a Config object as input.
        5: Add an entry in the COLAVBuilder class, which builds it from config if the type matches.
        See an example for the Kuwata VO below.

    Author: Trym Tengesdal
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import colav_simulator.common.config_parsing as cp
import colav_simulator.core.colav.cpp_to_py_interfaces.intention_model_interface.build.parameters as im_param
import colav_simulator.core.colav.cpp_to_py_interfaces.intention_model_interface.build.geometry as im_geom
import colav_simulator.core.colav.cpp_to_py_interfaces.intention_model_interface.build.intention_model as im
import colav_simulator.core.guidances as guidance
import matplotlib.pyplot as plt
import numpy as np
from seacharts.enc import ENC

class IMWrapper:
    """Intention model wrapper"""

    def __init__(self, num_ships, own_ship_obj) -> None:
        self.own_ship_obj = own_ship_obj
        self.ship_intentions = {}
        self.parameters = im_param.default_parameters(num_ships)
        self.intention_prediction_file = "test.csv"
        self.intention_model_path = "colav_simulator/core/colav/cpp_to_py_interfaces/external/ship_intention_inference/files/intention_models/intention_model_from_code.xdsl"

    def calculate_intentions(self, ship_list, timestep) -> None:

        ship_states = im.IntVectorMap()
        mmsi_list = im.IntVector()
        for ship_obj in ship_list:
            mmsi_list.append(ship_obj.mmsi)
            ship_states[ship_obj.mmsi] = ship_obj.csog_state

        if self.own_ship_obj.mmsi in self.ship_intentions:
            self.ship_intentions[self.own_ship_obj.mmsi].run_inference(ship_states, mmsi_list)
            x, y = ship_states[self.own_ship_obj.mmsi][0:2]
            self.ship_intentions[self.own_ship_obj.mmsi].save_intention_predictions_to_file(self.intention_prediction_file,\
                                                                                       x, y, timestep)
        own_ship_sog = ship_states[self.own_ship_obj.mmsi][2]

        for ship_obj in ship_list:
            if ship_obj != self.own_ship_obj:
                if ship_obj.mmsi in self.ship_intentions:
                    self.ship_intentions[ship_obj.mmsi].run_inference(ship_states, mmsi_list)
                    x, y = ship_states[ship_obj.mmsi][0:2]
                    self.ship_intentions[ship_obj.mmsi].save_intention_predictions_to_file(self.intention_prediction_file,\
                                                                                       x, y, timestep)
                else:
                    dist = im_geom.evaluateDistance(ship_states[ship_obj.mmsi][im_geom.PX] - ship_states[self.own_ship_obj.mmsi][im_geom.PX],\
                                                    ship_states[ship_obj.mmsi][im_geom.PY] - ship_states[self.own_ship_obj.mmsi][im_geom.PY])

                    if  ((dist < self.parameters.starting_distance) \
                         and (own_ship_sog > 0.1)):

                        if self.own_ship_obj.mmsi not in self.ship_intentions:
                            self.ship_intentions[self.own_ship_obj.mmsi] = im.IntentionModel(self.intention_model_path \
                                                                                             , self.parameters \
                                                                                             , self.own_ship_obj.mmsi \
                                                                                             , ship_states)

                            if ship_obj.mmsi not in self.ship_intentions:
                                self.ship_intentions[ship_obj.mmsi] = im.IntentionModel(self.intention_model_path \
                                                                                        , self.parameters \
                                                                                        , ship_obj.mmsi \
                                                                                        , ship_states)
