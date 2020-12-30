# import stuff
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.funcsall import *
from placerg.funcsboot import *

name_all='variables/loop_stim10e-16.0et6.0ph1.0p1.0t0.1plvarybpvary.pkl'
name_sum='variables/sum_stim10e-16.0et6.0ph1.0p1.0t0.1plvarybpvary.pkl'

labelname= 'type'

arra=['/home/mia/OneDrive/simsrg/a_stim10e-16.0et6.0ph1.0p1.0t0.1plNonebp0.5.pkl', '/home/mia/OneDrive/simsrg/a_stim10e-16.0et6.0ph1.0p1.0t0.1pl0.5bp0.0.pkl', '/home/mia/OneDrive/simsrg/a_stim10e-16.0et6.0ph1.0p1.0t0.1plNonebp0.0.pkl', '/home/mia/OneDrive/simsrg/nolatent_a_stim3e-8.0et6.0ph0.0p0.5t0.1pl1.0bp0.0.pkl']
arrenv=['/home/mia/OneDrive/simsrg/env_stim10e-16.0et6.0ph1.0p1.0t0.1plNonebp0.5.pkl', '/home/mia/OneDrive/simsrg/env_stim10e-16.0et6.0ph1.0p1.0t0.1pl0.5bp0.0.pkl', '/home/mia/OneDrive/simsrg/env_stim10e-16.0et6.0ph1.0p1.0t0.1plNonebp0.0.pkl', '/home/mia/OneDrive/simsrg/nolatent_env_stim3e-8.0et6.0ph0.0p0.5t0.1pl1.0bp0.0.pkl']

pltall, expall= loopall(arra, arrenv, labelname)
save_object(pltall, name_all)
save_object(expall, name_sum)
