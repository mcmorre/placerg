#main.py
# set up track structure
# this one loops over alpha and epsilon
import numpy as np
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
from placerg.runfunc import *

for i in np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]):
    N0 = 2048 # number of cells
            
    nstim = i # number of nonplace stimuli
        
    percell= 1.0 # probability that each field is accepted

    placeprob = 'None'

    bothprob = 0.5

    time= np.float(0.1)

    phi=1.0 # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = 6.0 # multiply by this constant to increase overall activity of 
    # network

    epsilon= -16.0/6.
    
    runsim(N0, nstim, percell, placeprob, bothprob, time, phi, eta, epsilon)
