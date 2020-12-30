#main.py
# set up track structure
# this one loops over alpha and epsilon
import numpy as np
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
from placerg.runfunc import *



for i in np.array([-8./6., -8.5/6., -9./6., -9.5/6., -10./6., -10.5/6., -11./6.,-11.5/6., -12./6.,-12.5/6., -13./6., -13.5/6., -14./6., -14.5/6., -15./6., -15.5/6., -16./6., -16.5/6., -17./6., -17.5/6.]):
    N0 = 2048 # number of cells

    nstim = 10 # number of nonplace stimuli
        
    percell= 1.0 # probability that each field is accepted

    placeprob = "None"

    bothprob = 0.5

    time = np.float(0.1)
    phi=1.0 # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = 6.0 # multiply by this constant to increase overall activity of 
    # network

    epsilon= i
    
    runsim(N0, nstim, percell, placeprob, bothprob, time, phi, eta, epsilon)
