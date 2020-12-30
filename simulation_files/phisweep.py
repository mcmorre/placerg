#main.py
# set up track structure
# this one loops over alpha and epsilon
import numpy as np
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
from placerg.runfunc import *




for i in np.array([0.8, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5]):
    N0 = 3048 # number of cells
            
    nstim = 10 # number of nonplace stimuli
        
    percell= 1.0 # probability that each field is accepted

    placeprob = 'None'

    bothprob = 0.5

    time = np.float(0.1)

    phi=i # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = 6.0 # multiply by this constant to increase overall activity of 
    # network

    epsilon= -16./6.
    
    runsim(N0, nstim, percell, placeprob, bothprob, time, phi, eta, epsilon)
