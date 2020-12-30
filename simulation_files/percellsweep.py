#main.py
# set up track structure
# this one loops over alpha and epsilon
import numpy as np
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
from placerg.runfunc import *


for i in np.array([1.0, 0.98, 0.95, 0.93, .9, 0.88, 0.85, 0.83, .8, 0.75, .7, 0.65, .6, 0.55, .5, 0.45, .4, 0.35, .3, 0.25 ]):
    N0 = 3048 # number of cells
            
    nstim = 10 # number of nonplace stimuli
        
    percell= i # probability that each field is accepted

    placeprob = 'None'

    bothprob = 0.5

    time=np.float(0.1)

    phi=1.0 # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = 6.0 # multiply by this constant to increase overall activity of 
    # network

    epsilon= -16.0/6.
    
    runsim(N0, nstim, percell, placeprob, bothprob, time, phi, eta, epsilon)
