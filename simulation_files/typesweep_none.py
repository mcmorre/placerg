#main.py
import numpy as np
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
from placerg.runfunc import *


N0 = 2048 # number of cells

nstim = 10 # number of nonplace stimuli

percell= 1.0 # probability that each field is accepted

placeprob = "None"

bothprob = 0.0

time=np.float(0.1)

phi=1.0 # multiply by this constant to adjust overall activity of 
# nonplace cells

eta = 6.0 # multiply by this constant to increase overall activity of 
# network

epsilon= -16./6.

runsim(N0, nstim, percell, placeprob, bothprob, time, phi, eta, epsilon)


