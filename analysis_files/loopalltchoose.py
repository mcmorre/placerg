# import stuff
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.funcsall import *
from placerg.funcsboot import *
from placerg.runfunc  import *
#env_stim10e-16.0et6.0ph1.0p1.0tchooseplNonebp0.5
name_all='variables/loop_env_stim10e-16.0et6.0ph1.0p1.0tchoose_23plNonebp0.5'
name_sum='variables/sum_env_stim10e-16.0et6.0ph1.0p1.0tchoose_23plNonebp0.5'
labelname= 'time'


arra='/home/mia/OneDrive/simsrg/a_stim23e-16.0et6.0ph1.0p1.0tchoose_23plNonebp0.5.pkl'
arrenv='/home/mia/OneDrive/simsrg/env_stim23e-16.0et6.0ph1.0p1.0tchoose_23plNonebp0.5.pkl'

globfunc(arra, arrenv, name_all, name_sum, labelname)


