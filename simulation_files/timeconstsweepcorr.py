#main.py
# set up track structure
# this one loops over alpha and epsilon
import numpy as np
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
from placerg.runfunc import *

  
nstim = 10 # number of nonplace stimuli

inl=['corrbinary5.1', 'corr5.', 'corr3.', 'corr2.']
tc= [np.array([5.,5.,5.,5.,5.,.1,.1,.1,.1,.1]), np.full((nstim,),5.), np.full((nstim,),3.), np.full((nstim,),2.)]

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
