#runfunc.py
import numpy as np
from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
from placerg.funcsboot import *
import glob


def runsim(N0, nstim, percell, placeprob, bothprob, time, phi, eta, epsilon, inputlabel='normal', k=8, N=1024):
    N0 = N0 # number of cells

    N = N # number of cells after removing silent ones
 
    loop = 200 # number of track runs

    xmax = 50 # track length

    dt=1. # increment for measurement locations

    # set up network structure

    nstim = nstim # number of nonplace stimuli

    percell= percell # probability that each field is accepted

    if placeprob == "None":
        placeprob = ["None", "None"]
    else:
        placeprob = [1-placeprob,placeprob]  #np.array([1.,0.])  probability that cell is coupled to place field
    # [p(not coupled), p(coupled)] 
    if nstim==0:
        npprob=np.array([1,0])
    else:
        npprob=np.array([1-percell, percell]) 
    # probability that cell is coupled to percell fields
    bothprob = np.array([1-bothprob,bothprob]) # probability that cell is coupled 
    # to both place and nonplace field
    # [p(not coupled), p(coupled)] 

    # now initialize distribution parameters of fields and couplings
    # here the place field couplings will be gamma distributed
    # the nonplace field couplings will be normally distributed

    vj = 0. # mean of couplings for nonplace fields
  
    vjplace = 1.0 # mean of couplings for place fields, if 'None'
    # then runs simulation of nonplace only

    sj = 1. # standard deviation of couplings for nonplace fields

    sjplace = 1.  # standard deviation of couplings for place fields

    time=time

    if type(time)!=float:
        print('running with non-uniform time constants')
        timeconst=time
        timelabel=inputlabel
        
    else: 
        timeconst = np.full((nstim,), time) # mean length of stochastic process in track lengths
        timelabel=time

    phi=phi # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = eta # multiply by this constant to increase overall activity of 
    # network

    epsilon= epsilon

    env=environ(N0=N0, N=N, loop=loop, xmax=xmax, dt=dt, nstim=nstim,\
    percell=percell, bothprob=bothprob,placeprob=placeprob,\
    npprob=npprob, vj=vj, vjplace=vjplace, sj=sj,\
    sjplace=sjplace,timeconst=timeconst,\
    phi=phi, eta=eta, epsilon=epsilon)


    k=k
    a=infoset(N, env.pmat, k)
    #afake=infoset(N, env.pmatfake, k)
    #arand=infoset(N, env.randommat, k)


    name_env='/home/mia/OneDrive/simsrg/env_stim{}e{}et{}ph{}p{}t{}pl{}bp{}.pkl'.format(nstim, np.round(epsilon*6., 1), np.round(eta,1), phi, percell, timelabel,placeprob[1], bothprob[1])
    name_a = '/home/mia/OneDrive/simsrg/a_stim{}e{}et{}ph{}p{}t{}pl{}bp{}.pkl'.format(nstim,np.round(epsilon*6., 1), np.round(eta,1), phi, percell, timelabel,placeprob[1], bothprob[1])
    #name_afake = '/home/mia/OneDrive/simsrg/afake1_stim{}e{}a{}p{}.pkl'.format(nstim,epsilon, alpha, i)
    #name_arand = 'variables/arand1_e{}a{}.pkl'.format(i,j)
    save_object(env, name_env)
    save_object(a, name_a)
    #save_object(afake, name_afake)
    #save_object(arand, name_arand)
    print('sweep complete')
    #del env
    #envnew=load_object('pkl_env.pkl')
    
def runsim_big(N0, nstim, percell, placeprob, bothprob, time, phi, eta, epsilon, inputlabel='normal', k=8, N=1024):
    N0 = N0 # number of cells

    N = N # number of cells after removing silent ones
 
    loop = 200 # number of track runs

    xmax = 50 # track length

    dt=1. # increment for measurement locations

    # set up network structure

    nstim = nstim # number of nonplace stimuli

    percell= percell # probability that each field is accepted

    if placeprob == "None":
        placeprob = ["None", "None"]
    else:
        placeprob = [1-placeprob,placeprob]  #np.array([1.,0.])  probability that cell is coupled to place field
    # [p(not coupled), p(coupled)] 
    if nstim==0:
        npprob=np.array([1,0])
    else:
        npprob=np.array([1-percell, percell]) 
    # probability that cell is coupled to percell fields
    bothprob = np.array([1-bothprob,bothprob]) # probability that cell is coupled 
    # to both place and nonplace field
    # [p(not coupled), p(coupled)] 

    # now initialize distribution parameters of fields and couplings
    # here the place field couplings will be gamma distributed
    # the nonplace field couplings will be normally distributed

    vj = 0. # mean of couplings for nonplace fields
  
    vjplace = 1.0 # mean of couplings for place fields, if 'None'
    # then runs simulation of nonplace only

    sj = 1. # standard deviation of couplings for nonplace fields

    sjplace = 1.  # standard deviation of couplings for place fields

    time=time

    if type(time)!=float:
        print('running with non-uniform time constants')
        timeconst=time
        timelabel=inputlabel
        
    else: 
        timeconst = np.full((nstim,), time) # mean length of stochastic process in track lengths
        timelabel=time

    phi=phi # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = eta # multiply by this constant to increase overall activity of 
    # network

    epsilon= epsilon

    env=environ(N0=N0, N=N, loop=loop, xmax=xmax, dt=dt, nstim=nstim,\
    percell=percell, bothprob=bothprob,placeprob=placeprob,\
    npprob=npprob, vj=vj, vjplace=vjplace, sj=sj,\
    sjplace=sjplace,timeconst=timeconst,\
    phi=phi, eta=eta, epsilon=epsilon)

    name_env='/home/mia/OneDrive/simsrg/env_stim{}e{}et{}ph{}p{}t{}pl{}bp{}_big.pkl'.format(nstim, np.round(epsilon*6., 1), np.round(eta,1), phi, percell, timelabel,placeprob[1], bothprob[1])
    pmat_save = env.pmat 
    
    save_object(env, name_env)
    del env
    
    k=k
    a=infoset(N, pmat_save, k)
    #afake=infoset(N, env.pmatfake, k)
    #arand=infoset(N, env.randommat, k)


    name_a = '/home/mia/OneDrive/simsrg/a_stim{}e{}et{}ph{}p{}t{}pl{}bp{}_big.pkl'.format(nstim,np.round(epsilon*6., 1), np.round(eta,1), phi, percell, timelabel,placeprob[1], bothprob[1])
    #name_afake = '/home/mia/OneDrive/simsrg/afake1_stim{}e{}a{}p{}.pkl'.format(nstim,epsilon, alpha, i)
    #name_arand = 'variables/arand1_e{}a{}.pkl'.format(i,j)

    save_object(a, name_a)
    del a
    #save_object(afake, name_afake)
    #save_object(arand, name_arand)
    print('sweep complete')
    #del env
    #envnew=load_object('pkl_env.pkl')

def globfunc(arra, arrenv, name_all, name_sum, labelname):
    arra=sorted(glob.glob(arra))
    print(str(len(arra)) +' analysis objects found')
    arrenv=sorted(glob.glob(arrenv))
    print(str(len(arrenv)) +' simulation objects found')
    pltall, expall= loopall(arra, arrenv, labelname)
    save_object(pltall, name_all)
    save_object(expall, name_sum)

def runsimverif(N0=3048, nstim=3, percell=0.5, placeprob=0.5, bothprob=0.0, time=0.1, phi=1.0, eta=6.0, epsilon=-16./6., inputlabel='normal'):
    N0 = N0 # number of cells

    N = 1024 # number of cells after removing silent ones
 
    loop = 200 # number of track runs

    xmax = 50 # track length

    dt=1. # increment for measurement locations

    # set up network structure

    nstim = nstim # number of nonplace stimuli

    percell= percell # probability that each field is accepted

    if placeprob == "None":
        placeprob = ["None", "None"]
    else:
        placeprob = [1-placeprob,placeprob]  #np.array([1.,0.])  probability that cell is coupled to place field
    # [p(not coupled), p(coupled)] 
    if nstim==0:
        npprob=np.array([1,0])
    else:
        npprob=np.array([1-percell, percell]) 
    # probability that cell is coupled to percell fields
    bothprob = np.array([1-bothprob,bothprob]) # probability that cell is coupled 
    # to both place and nonplace field
    # [p(not coupled), p(coupled)] 

    # now initialize distribution parameters of fields and couplings
    # here the place field couplings will be gamma distributed
    # the nonplace field couplings will be normally distributed

    vj = 0. # mean of couplings for nonplace fields
  
    vjplace = 1.0 # mean of couplings for place fields, if 'None'
    # then runs simulation of nonplace only

    sj = 1. # standard deviation of couplings for nonplace fields

    sjplace = 1.  # standard deviation of couplings for place fields

    time=time

    if type(time)!=float:
        print('running with non-uniform time constants')
        timeconst=time
        timelabel=inputlabel
        
    else: 
        timeconst = np.full((nstim,), time) # mean length of stochastic process in track lengths
        timelabel=time

    phi=phi # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = eta # multiply by this constant to increase overall activity of 
    # network

    epsilon= epsilon

    env=environ(N0=N0, N=N, loop=loop, xmax=xmax, dt=dt, nstim=nstim,\
    percell=percell, bothprob=bothprob,placeprob=placeprob,\
    npprob=npprob, vj=vj, vjplace=vjplace, sj=sj,\
    sjplace=sjplace,timeconst=timeconst,\
    phi=phi, eta=eta, epsilon=epsilon)


    k=8
    a=infoset(N, env.pmat, k)
    #afake=infoset(N, env.pmatfake, k)
    #arand=infoset(N, env.randommat, k)


    name_env='/home/mia/OneDrive/simsrg/verif_env_stim{}e{}et{}ph{}p{}t{}pl{}bp{}.pkl'.format(nstim, np.round(epsilon*6., 1), np.round(eta,1), phi, percell, timelabel,placeprob[1], bothprob[1])
    name_a = '/home/mia/OneDrive/simsrg/verif_a_stim{}e{}et{}ph{}p{}t{}pl{}bp{}.pkl'.format(nstim,np.round(epsilon*6., 1), np.round(eta,1), phi, percell, timelabel,placeprob[1], bothprob[1])
    #name_afake = '/home/mia/OneDrive/simsrg/afake1_stim{}e{}a{}p{}.pkl'.format(nstim,epsilon, alpha, i)
    #name_arand = 'variables/arand1_e{}a{}.pkl'.format(i,j)
    save_object(env, name_env)
    save_object(a, name_a)
    #save_object(afake, name_afake)
    #save_object(arand, name_arand)
    print('sweep complete')
    #del env
    #envnew=load_object('pkl_env.pkl')

def runsimnolatent(N0=3048, nstim=3, percell=0.5, placeprob=1.0, bothprob=0.0, time=0.1, phi=0.0, eta=6.0, epsilon=-16./12., inputlabel='normal'):
    N0 = N0 # number of cells

    N = 1024 # number of cells after removing silent ones
 
    loop = 200 # number of track runs

    xmax = 50 # track length

    dt=1. # increment for measurement locations

    # set up network structure

    nstim = nstim # number of nonplace stimuli

    percell= percell # probability that each field is accepted

    if placeprob == "None":
        placeprob = ["None", "None"]
    else:
        placeprob = [1-placeprob,placeprob]  #np.array([1.,0.])  probability that cell is coupled to place field
    # [p(not coupled), p(coupled)] 
    if nstim==0:
        npprob=np.array([1,0])
    else:
        npprob=np.array([1-percell, percell]) 
    # probability that cell is coupled to percell fields
    bothprob = np.array([1-bothprob,bothprob]) # probability that cell is coupled 
    # to both place and nonplace field
    # [p(not coupled), p(coupled)] 

    # now initialize distribution parameters of fields and couplings
    # here the place field couplings will be gamma distributed
    # the nonplace field couplings will be normally distributed

    vj = 0. # mean of couplings for nonplace fields
  
    vjplace = 1.0 # mean of couplings for place fields, if 'None'
    # then runs simulation of nonplace only

    sj = 1. # standard deviation of couplings for nonplace fields

    sjplace = 1.  # standard deviation of couplings for place fields

    time=time

    if type(time)!=float:
        print('running with non-uniform time constants')
        timeconst=time
        timelabel=inputlabel
        
    else: 
        timeconst = np.full((nstim,), time) # mean length of stochastic process in track lengths
        timelabel=time

    phi=phi # multiply by this constant to adjust overall activity of 
    # nonplace cells

    eta = eta # multiply by this constant to increase overall activity of 
    # network

    epsilon= epsilon

    env=environ(N0=N0, N=N, loop=loop, xmax=xmax, dt=dt, nstim=nstim,\
    percell=percell, bothprob=bothprob,placeprob=placeprob,\
    npprob=npprob, vj=vj, vjplace=vjplace, sj=sj,\
    sjplace=sjplace,timeconst=timeconst,\
    phi=phi, eta=eta, epsilon=epsilon)


    k=8
    a=infoset(N, env.pmat, k)
    #afake=infoset(N, env.pmatfake, k)
    #arand=infoset(N, env.randommat, k)


    name_env='/home/mia/OneDrive/simsrg/nolatent_env_stim{}e{}et{}ph{}p{}t{}pl{}bp{}.pkl'.format(nstim, np.round(epsilon*6., 1), np.round(eta,1), phi, percell, timelabel,placeprob[1], bothprob[1])
    name_a = '/home/mia/OneDrive/simsrg/nolatent_a_stim{}e{}et{}ph{}p{}t{}pl{}bp{}.pkl'.format(nstim,np.round(epsilon*6., 1), np.round(eta,1), phi, percell, timelabel,placeprob[1], bothprob[1])
    #name_afake = '/home/mia/OneDrive/simsrg/afake1_stim{}e{}a{}p{}.pkl'.format(nstim,epsilon, alpha, i)
    #name_arand = 'variables/arand1_e{}a{}.pkl'.format(i,j)
    save_object(env, name_env)
    save_object(a, name_a)
    #save_object(afake, name_afake)
    #save_object(arand, name_arand)
    print('sweep complete')
    #del env
    #envnew=load_object('pkl_env.pkl')
    
    
def globfunc_partial(arra, arrenv, name_all, name_sum, labelname):
    arra=sorted(glob.glob(arra))
    print(str(len(arra)) +' analysis objects found')
    arrenv=sorted(glob.glob(arrenv))
    print(str(len(arrenv)) +' simulation objects found')
    pltall, expall= loopall_partial(arra, arrenv, labelname)
    save_object(pltall, name_all)
    save_object(expall, name_sum)

