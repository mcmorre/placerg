# import stuff
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.funcsall import *
from placerg.objects import *
from placerg.funcsboot import *
from placerg.runfunc import *

name_all='variables/loop_stim10e-16.0et6.0phvaryp1.0t0.1plNonebp0.5.pkl'
name_sum='variables/sum_stim10e-16.0et6.0phvaryp1.0t0.1plNonebp0.5.pkl'
labelname= 'phi'
arra='/home/mia/OneDrive/simsrg/a_stim10e-16.0et6.0ph*p1.0t0.1plNonebp0.5.pkl'
arrenv='/home/mia/OneDrive/simsrg/env_stim10e-16.0et6.0ph*p1.0t0.1plNonebp0.5.pkl'

globfunc(arra, arrenv, name_all, name_sum, labelname)

