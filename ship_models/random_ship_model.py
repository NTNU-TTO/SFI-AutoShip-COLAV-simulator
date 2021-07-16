import math
import random
import numpy as np

class Random_ship_model:
    """
        Creates a ship with random parameters
    """
    def __init__(self):
        self.length = random.randint(10, 200)
        #self.l_r = random.randint(0, 10)
        self.A = self.length/2
        self.B = self.length/2
        self.C = self.length/4
        self.D = self.length/4
        self.width = self.C + self.D
        self.os_x = self.A - self.B
        self.os_y = self.D - self.C

        # Motion limits
        self.r_max = np.deg2rad(random.uniform(1,5))
        self.U_max = random.randint(15, 25)
        self.a_max = random.uniform(0.2,2)

        self.m = self.length*self.width*20
        #self.I_z = 19703.0

        
        




