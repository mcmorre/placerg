# import stuff
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.funcsall import *
from placerg.funcsboot import *
from placerg.runfunc  import *
#name_all='variables/loop_env_stim10e-16.0et6.0ph1.0p1.0t0.1plNonebp0.5_big' #Run these first!
#name_sum='variables/sum_env_stim10e-16.0et6.0ph1.0p1.0t0.1plNonebp0.5_big'  # Run these first!
name_all='variables/loop_env_stim10e-16.0et6.0ph1.0p1.0t0.1plNonebp0.5_big_partial' #after running this file with first 2 lines run with this and following to collect error bars. 
name_sum='variables/sum_env_stim10e-16.0et6.0ph1.0p1.0t0.1plNonebp0.5_big_partial'  # if you want to run this code with larger system size I suggest rewriting to analysis to apply to general system size
labelname= 'time'


arra='/home/mia/OneDrive/simsrg/a_stim10e-16.0et6.0ph1.0p1.0t0.1plNonebp0.5_big*.pkl'
arrenv='/home/mia/OneDrive/simsrg/env_stim10e-16.0et6.0ph1.0p1.0t0.1plNonebp0.5_big*.pkl'

globfunc_partial(arra, arrenv, name_all, name_sum, labelname)







