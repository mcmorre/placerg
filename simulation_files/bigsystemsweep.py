#main.py
# set up track structure
# this one loops over alpha and epsilon
import numpy as np
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
from placerg.runfunc import *

  
nstim = 10 # number of nonplace stimuli


# i have done these simulations: i=[.05, .1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0, 1.1, 1.2]
#['binary.4.1', 'binary.1.5', 'binary.1.8', 'binary.2.1', 'binary.05.1']
#['gamma.5.2', 'gamma.3.1', 'gamma.2.05']
#['unif.09.11', 'unif.07.13', 'unif.05.15', 'unif.02.18']
#inl=['unif.09.11', 'unif.07.13', 'unif.05.15', 'unif.02.18','unif.01.19']
inl=['bigsystem']
#[np.random.choice([.4,.1], size=(nstim,), p=[.5,.5]), np.random.choice([.1,.5], size=(nstim,), p=[.5,.5]), np.random.choice([.8,.1], size=(nstim,), p=[.5,.5]), \
    #np.random.choice([.2,.1], size=(nstim,), p=[.5,.5]), np.random.choice([.05,.1], size=(nstim,), p=[.5,.5]) ]
#[gamma(.5, .2, nstim).flatten(), gamma(.3, .1, nstim).flatten(), gamma(.2, .05, nstim).flatten()]
#[np.random.uniform(.09, 0.11, size=(nstim,)), np.random.uniform(.07, 0.13, size=(nstim,)), np.random.uniform(.05, 0.15, size=(nstim,)), np.random.uniform(.02,0.18, size=(nstim,))]
#tc= [np.random.uniform(.09, 0.11, size=(nstim,)), np.random.uniform(.07, 0.13, size=(nstim,)), np.random.uniform(.05, 0.15, size=(nstim,)), np.random.uniform(.02,0.18, size=(nstim,)), np.random.uniform(.01, 0.19, size=(nstim,))]

for i in range(len(inl)):
    N0 = 2**13+2000 # number of cells
    N = 2**13
    k=11

        
    percell= 1.0 # probability that each field is accepted

    placeprob = 'None'

    bothprob = 0.5

    time=0.1

    phi=1.0 # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = 6.0 # multiply by this constant to increase overall activity of 
    # network

    epsilon= -16.0/6.
    
    runsim_big(N0, nstim, percell, placeprob, bothprob, time, phi, eta, epsilon, inputlabel=inl[i], k=k, N=N)
