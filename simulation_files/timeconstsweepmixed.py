#main.py
# set up track structure
# this one loops over alpha and epsilon
import numpy as np
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
from placerg.runfunc import *

  
nstim = 23 #10 # number of nonplace stimuli



inl= ['choose_23'] #['choose']


#tc = [np.array([0.1, 0.33, 0.66, 1.0,3.3, 6.6, 10, 33, 66, 100])]
tc = [np.array([0.066, 0.088, 0.1, 0.15, 0.19, 0.33, 0.66, 0.88, 1.0, 1.5, 1.9, 3.3, 6.6, 8.8, 10, 15, 19, 33, 66, 88, 100, 150, 190])]
for i in range(len(inl)):
    N0 = 3048 # number of cells
            

        
    percell= 1.0 # probability that each field is accepted

    placeprob = 'None'

    bothprob = 0.5

    time=tc[i]

    phi=1.0 # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = 6.0 # multiply by this constant to increase overall activity of 
    # network

    epsilon= -16.0/6.
    
    runsim(N0, nstim, percell, placeprob, bothprob, time, phi, eta, epsilon, inputlabel=inl[i])
