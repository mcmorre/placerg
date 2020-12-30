#main.py
# set up track structure
# this one loops over alpha and epsilon
import numpy as np
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
from placerg.runfunc import *


for i in np.array([2.8, 3., 3.2, 3.4, 3.6,3.8, 4.,4.2,4.4, 4.6, 4.8, 5.,5.2, 5.4, 5.6, 5.8, 6., 6.2, 6.4, 6.6, 6.8, 7.0]):
    N0 = 3048 # number of cells
            
    nstim = 10 # number of nonplace stimuli
        
    percell= 1.0 # probability that each field is accepted

    placeprob = 'None'

    bothprob = 0.5

    time = np.float(0.1)

    phi=1.0 # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = i # multiply by this constant to increase overall activity of 
    # network

    epsilon= -16./6.
    
    runsim(N0, nstim, percell, placeprob, bothprob, time, phi, eta, epsilon)
