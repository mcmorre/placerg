#funcs.py

from blis.py import gemm # for 
import numpy as np
import pickle
import os
import nbformat
import nbparameterise
from nbconvert.preprocessors import ExecutePreprocessor
from scipy.special import gamma as gammafunc
import matplotlib.pyplot as plt
"""
Generating jupyter notebooks from a template notebook
----------------------------------------------------------------------
"""
def execute_notebook(notebook_filename, notebook_filename_out, params_dict, 
    run_path="", timeout=6000000):
    """
    this function will generate a jupyter notebook from a provided template notebook with
    the specified variabe params_dict changed to the specified value. it uses helper functions 
    read_in_notebook and set_parameters.
    -------------------------------------------
    Inputs:
    notebook_filename: name of the template notebook
    notebook_filename_out: name of the notebook you want to generate
    params_dict: specify the parameter you want to include in the new notebook as follows: {'num': '3'} 
    where num is the variable and 3 is the value we want it to have in the generated notebook
    """    
    notebook_fp = os.path.join(run_path, notebook_filename)
    nb = read_in_notebook(notebook_fp)
    new_nb = set_parameters(nb, params_dict)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3')

    try:
        ep.preprocess(new_nb, {'metadata': {'path': run_path}})
    except:
        msg = 'Error while executing: "{0}".\n\n'.format(notebook_filename)
        msg = '{0}See notebook "{1}" for traceback.'.format(
                msg, notebook_filename_out)
        print(msg)
        raise
    finally:
        with open(notebook_filename_out, mode='wt') as f:
            nbformat.write(new_nb, f)   


def read_in_notebook(notebook_fp):
    with open(notebook_fp) as f:
        nb = nbformat.read(f, as_version=4)
    return nb


def set_parameters(nb, params_dict):
    orig_parameters = nbparameterise.extract_parameters(nb)
    params = nbparameterise.parameter_values(orig_parameters, **params_dict)
    new_nb = nbparameterise.replace_definitions(nb, params, execute=False)
    return new_nb

"""
Saving and loading objects
----------------------------------------------------------------------------
"""

def save_object(obj, filename):
    """
    Save python object
    -------------------------------------------
    Inputs:
    obj: object I want to save
    filename: name of pickle file I want to dump into. Example: 'dump_file.pkl'
    """

    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """
    Load python object
    -------------------------------------------
    Inputs:
    filename: name of pickle file I want to load. Example: 'dump_file.pkl'
    -------------------------------------------
    Output:
    loaded object
    """
    with open(filename, "rb") as input_file:
        e = pickle.load(input_file)
    return e

"""
Helper functions
----------------------------------------------------
"""

def rollx(arr):
    """
    rolls indices of each neuron's times series independently
    this is the method meshulam et al used to shuffle data
    -------------------------------------------------- 
    Inputs:
    x: matrix, we want indices of rows to be rolled independently
    -------------------------------------------------------
    output: matrix with rolling procedure applied on it
    """
    x=np.zeros(arr.shape)
    num=np.random.choice(x.shape[1], size=(x.shape[0],))
    for i in range(x.shape[0]):
        x[i,:]=np.roll(arr[i,:],num[i])
    return x	

def blis_gemm(X, W):
    """
    Fast matrix multiplication using blis.py
    -------------------------------------------
    Inputs:
    X: matrix shape (a,b,c)
    W: matrix shape (c,d)
    ------------------------------------------
    Output:
    X /dot W: matrix shape  (a,b,d)
    """
    contain=[]
    for i in range(X.shape[0]):
        y=gemm(X[i,:,:], W, trans1=False, trans2=False)
        contain.append(y)
    contain=np.array(contain)
    return(contain)

def drawpdf(dist, binz):
    
    """
    Draw probability density function from a given data set.
    ------------------------------------------------------------------------
    Inputs:
    dist: given data set. shape: (dist.size,)
    dt: bin width for pdf calculation. shape: scalar
    -----------------------------------------------------------------------
    Output: 
    x: bin locations. shape: (res.size,)
    res: probability of data being in that bin. shape:(int((max(dist)-
         min(dist))/dt),)
    """
    
    x,y = np.histogram(dist, bins=binz, density=True)
    #x=x/np.sum(x)
    for i in range(0,len(y)-1):
        y[i]=(y[i]+y[i+1])/2
    y=y[0:len(y)-1]
    return y,x

def fillerrorexp(mup, muerrp, x0, xf, axs, color):
    x=np.arange(x0-.2, xf+.2, .01)
    axs.hlines(mup, x0-.2, xf+.2, linestyle='--' , color=color)
    axs.fill_between(x, mup+muerrp,mup-muerrp, color=color, alpha=.5)  

def plotexps(allo, label, inds, fontsize, ticksize, t0, b0, t1, b1, t2, b2, t3, b3, xx0, y0, xx1, y1, xx2, y2, xx3, y3):
    if label=='eta':
        xlabel=r'$\eta$'
        xd=allo.eta
    if label=='epsilon':
        xlabel=r'$\epsilon$'
        xd=allo.epsilon
    if label=='percell':
        xlabel=r'$q$'
        xd=allo.percell
    if label=='timeconst':
        xlabel=r'$\tau$'
        xd=allo.timeconst
    if label=='stim':
        xlabel=r'$N_f$'
        xd=allo.stim
    if label=='phi':
        xlabel=r'$\phi$'
        xd=allo.phi
    fig, ax = plt.subplots(2,2, figsize=(10,10), sharex=True)
    for i in inds:
        if label=='timeconst':
            xdi=xd[i][0]
            x0=np.min(np.vstack(xd)[:,0].flatten()[inds])
            xf=np.max(np.vstack(xd)[:,0].flatten()[inds])
        else:
            xdi=xd[i]
            x0=np.min(np.array(xd).flatten()[inds])
            xf=np.max(np.array(xd).flatten()[inds])
        ax[0,0].errorbar(xdi, allo.alpha[i][1], allo.alphaerr[i][0], marker='o', color='black', markersize=5,\
                       linewidth=2)
        #ax[0,0].set_xlabel(xlabel, fontsize=fontsize)
        ax[0,0].set_ylabel(r'${\alpha}$', fontsize=fontsize)
        ax[0,0].tick_params(labelsize=ticksize)
        ax[0,1].errorbar(xdi, allo.beta[i][1], allo.betaerr[i][0], marker='o', color='black', markersize=5,\
                       linewidth=2)
        #ax[0,1].set_xlabel(xlabel, fontsize=fontsize)
        ax[0,1].set_ylabel(r'$\tilde{\beta}$', fontsize=fontsize)
        ax[0,1].tick_params(labelsize=ticksize)
        ax[1,0].errorbar(xdi, allo.z[i][1], allo.zerr[i][0], marker='o', color='black', markersize=5,\
                       linewidth=2)
        ax[1,0].set_xlabel(xlabel, fontsize=fontsize)
        ax[1,0].set_ylabel(r'$\tilde{z}$', fontsize=fontsize)
        ax[1,0].tick_params(labelsize=ticksize)
        ax[1,1].errorbar(xdi ,allo.mu[i][1], allo.muerr[i][0], marker='o', color='black', markersize=5,\
                       linewidth=2)
        ax[1,1].set_xlabel(xlabel, fontsize=fontsize)
        ax[1,1].set_ylabel(r'$\mu$', fontsize=fontsize) 
        ax[1,1].tick_params(labelsize=ticksize)
    mup=1.4
    muerrp=0.06
    fillerrorexp(mup, muerrp, x0, xf, ax[0,0], 'pink')
    mup=1.56 
    muerrp=0.03
    fillerrorexp(mup, muerrp, x0, xf, ax[0,0], 'skyblue')
    mup=1.73
    muerrp=0.11
    fillerrorexp(mup, muerrp, x0, xf, ax[0,0], 'gray')

    mup=0.88
    muerrp=0.01
    fillerrorexp(mup, muerrp, x0, xf, ax[0,1], 'pink') 
    mup=0.89
    muerrp=0.01
    fillerrorexp(mup, muerrp, x0, xf, ax[0,1], 'skyblue') 
    mup=0.86
    muerrp=0.02
    fillerrorexp(mup, muerrp, x0, xf, ax[0,1], 'gray') 
    mup=0.87
    muerrp=0.03
    #fillerrorexp(mup, muerrp, x0, xf, ax[1], 'palegreen') 

    mup=0.16
    muerrp=0.02
    fillerrorexp(mup, muerrp, x0, xf, ax[1,0], 'pink')
    mup=0.17
    muerrp=0.03
    fillerrorexp(mup, muerrp, x0, xf, ax[1,0], 'skyblue')
    mup=0.34
    muerrp=0.12
    fillerrorexp(mup, muerrp, x0, xf, ax[1,0], 'gray')
    mup=0.11
    muerrp=0.01
    #fillerrorexp(mup, muerrp, x0, xf, ax[2], 'palegreen')

    mup=-0.71
    muerrp=0.06
    fillerrorexp(mup, muerrp, x0, xf, ax[1,1], 'pink') 
    mup=-0.73
    muerrp=0.01
    fillerrorexp(mup, muerrp, x0, xf, ax[1,1], 'skyblue')   
    mup=-0.83
    muerrp=0.07
    fillerrorexp(mup, muerrp, x0, xf, ax[1,1], 'gray')  
    mup=-0.71
    muerrp=0.15
    #fillerrorexp(mup, muerrp, x0, xf, ax[3], 'palegreen')  

    ax[0,0].tick_params(labelsize=ticksize)
    ax[0,1].tick_params(labelsize=ticksize)
    ax[1,0].tick_params(labelsize=ticksize)
    ax[1,1].tick_params(labelsize=ticksize)
    ax[0,0].set_ylim(top=t0, bottom=b0)
    ax[0,1].set_ylim(top=t1, bottom=b1)
    ax[1,0].set_ylim(top=t2, bottom=b2)
    ax[1,1].set_ylim(top=t3, bottom=b3)
    ax[0,0].text(xx0, y0, r'(A)', fontsize=ticksize, weight='bold')
    ax[0,1].text(xx1, y1, r'(B)', fontsize=ticksize, weight='bold')
    ax[1,0].text(xx2, y2, r'(C)',fontsize=ticksize, weight='bold')
    ax[1,1].text(xx3, y3, r'(D)', fontsize=ticksize, weight='bold')
    plt.tight_layout()


def gamma(mean, stdev, N):
    
    """
    Generate a sample of size (N,1) from a chi-squared distribution with a 
    specified mean and variance.
    ---------------------------------------------------------------------
    Inputs: 
    mean: desired mean of resulting distribution. shape: scalar
    var: desired variance of resulting distribution. shape: scalar
    N: generates a sample of size (N,1) from specified distribution. 
       shape: scalar
    --------------------------------------------------------------------
    Output: sample from specified distribution. shape: (N,1)
    """
    
    a = (mean**2)/(stdev**2)
    
    b = (stdev**2)/mean
    
    dist = np.random.gamma(a, b, (N,1))
    
    return dist

def chisquare(mean, stdev, N):
    
    """
    Generate a sample of size (N,1) from a chi-squared distribution with a 
    specified mean and variance.
    ---------------------------------------------------------------------
    Inputs: 
    mean: desired mean of resulting distribution. shape: scalar
    var: desired variance of resulting distribution. shape: scalar
    N: generates a sample of size (N,1) from specified distribution. 
       shape: scalar
    -----------------------------------------------------------------
    Output: sample from specified distribution. shape: (N,1)
    """

    A = (stdev**2)/(2*mean)
    
    dist = A*np.random.chisquare(mean/A, (N,1))
    
    return dist

def autocorr(series, norm='real'):

    """
    generate normalized autocorrelation function
    ---------------------------------------------------
    Inputs:
    series: 1D array holding sequence I wish to calculate normalized 
            autocorrelation function of
    norm: if 'real': returns autocorrelation normalized by variance (default)
          if 'mom': returns autocorrelation normalized by mean fluctuations 
          squared, appropriate for momentum space
    --------------------------------------------------
    Output:
    normalized correlation function, of shape series.size+2
    """
    
    plotcorr = np.correlate(series,series,'full')
    nx = int(plotcorr.size/2)
    lags = np.arange(-nx, nx+1) # so last value is nx
    plotcorr /= (len(series)-lags)
    plotcorr -= np.mean(series)**2
    if norm == 'real':
        plotcorr /= np.var(series)
    if norm == 'mom':
        plotcorr /= np.mean((series-np.mean(series))**2)
    return lags, plotcorr

def crosscorr(a,b):
    """
    generate normalized crosscorrelation function
    ---------------------------------------------------
    Inputs:
    a,b: 1D arrays holding sequences I wish to calculate normalized 
         crosscorrelation function of
    --------------------------------------------------
    Output:
    normalized crosscorrelation function, of shape a.size+2
    """
    a = (a - np.mean(a)) / (np.std(a))
    b = (b - np.mean(b)) / (np.std(b))
    c = np.correlate(a, b, 'full')
    nx = int(c.size/2)
    lags = np.arange(-nx , nx+1) # so last value is nx - 1
    c /= (len(c)-lags)
    return lags, c

def stim(taus, sigmas, dt, leng):
    
    """
    Refer to http://th.if.uj.edu.pl/~gudowska/dydaktyka/Lindner_stochastic.pdf
    Ornstein-Uhlenbeck process, using Euler-Maruyama method.
    Here the mean of the process generated is 0.
    -------------------------------------------------------------
    Inputs:
    recall that nstim is the number of nonplace stimuli
    sigmas: standard deviation of stochastic process. shape: (nstim,)
    taus: time constant of stochastic process. shape: (nstim,)
    vs: mean of the stochastic process. shape: (nstim,)
    dt: time step. shape: scalar
    leng: desired length of process. in this case the desired length will be 
          loop*dt*xmax
    ----------------------------------------------------------
    Output: states of given nonplace fields, over time period leng at 
            intervals of dt.
            shape: (nstim, leng) --> (number of nonplace stimuli, loop*dt*xmax)
    """
    
    numstim=taus.size
    
    gamm=1./taus
    
    D=(sigmas**2)/taus
    
    arr=np.zeros((numstim,leng))
    
    for i in range(1,leng):
        
        rands=np.random.randn(numstim)
        
        arr[:,i]=arr[:,i-1]*(1-gamm*dt)+np.sqrt(2*D*dt)*rands
        
    return arr

def gauss(x,A,b,c):
    # make function of mean and standard dev, normalize it
    return A*np.exp((-(x-b)**2)/(2*c**2))

def gaussian(x,b,c):
    # make function of mean and standard dev, normalize it
    return (1/(c*np.sqrt(2*np.pi)))*np.exp((-(x-b)**2)/(2*c**2))

def linfunc(b,a,c):
    return a*b**c

def probfunc(K, a, b):
    return (a*K**b)

def eigfunc(r, m, C):
    return C*(r)**m

def expfunc(r, m, C):
    return C*np.exp(-m*r)

def gammafit(x,a,b,c,d):
    return c*(1/(gammafunc(a)*(b**a)))*(x**(a-1))*np.exp(-(x/b))+d

def expfunc2(x,b):
    return (x)**(-b)

def linear(x,a):
    return -x*a
"""
Array filling
---------------------------------------------------------
"""


def spikesbetter(P):
    """
    same as the custom cython function _dice6, a python implementation for easy use on other computers
    does spin selection procedure based on given array of probabilities
    --------------------------------------------------------------------
    Inputs:
    P: probability of silence array. shape (loop, xmax, N)
    -------------------------------------------------------------------
    Output: 
    array of spin values in {0,1} with shape (loop, xmax, N) 
    """
    spikes=np.zeros(P.shape)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            for k in range(P.shape[2]):
                if np.random.rand() > P[i,j,k]:
                    spikes[i,j,k] += 1
    return spikes

def fillfields(N, x, v, vdev, fields, process, loop):
    """
    fill empty fields array with entries. Note that fields array should be an 
    array of zeros
    ---------------------------------------------------
    Inputs:
    N: number of cells
    x: stack of N copies of measurement locations: as in 
       np.tile(np.arange(0,xmax, dt), (N,1))
    v: means of place cell waveforms. has shape (1,N)
    vdev: standard deviations of place cell waveforms. has shape (1,N)
    fields: empty fields array. has shape (loop, xmax, N+nstim), must be array 
            of zeros
    process: array holding all nonplace fields at every time step. has shape 
             (loop*xmax*dt, nstim)
    loop: number of track runs. integer
    --------------------------------------------------
    Output:
    fields array filled with fields
    """
    # fill fields array with nonplace stimuli
    #(taus, sigmas, dt, leng)

    fields[:,:,N:] = np.array(np.vsplit(process, loop)) # save filled nonplace 
    # fields in fields array
    fields[:,:,:N] = fillplace(fields[:,:,:N], x, v, vdev) # fill fields with 
    # place fields
    return fields

def fillJ(J, N, vjplace, sjplace, vj, sj, nstim, placeprob, npprob, bothprob, choice, phi):    
    """
    fill empty J array with entries
    ---------------------------------------------------
    Inputs:
    J: empty J array
    N: number of cells
    vjplace: mean of the place cell couplings. scalar 
    sjplace: standard deviation of the place cell couplings. scalar 
    vj: mean of the nonplace cell couplings. scalar 
    sj: standard deviation of the nonplace cell couplings. scalar 
    placeprob: probability that cell is coupled to place field. [p(not 
               coupled), p(coupled)]
    stimprob: probability that cell is coupled to nonplace field. [p(not 
              coupled), p(coupled)] 
    placeonlyprob: probability that place cell is coupled only to a place 
                   field. [p(only coupled to a place field), p(may be coupled 
                   to some nonplace fields)]
    choice: possible spin values. [0,1]
    const: normalize nonplace part of hamiltonian by this constant. 
           that is: I multiply every nonplace coupling my const such that when 
           I compile the hamiltonian, I get H(cell) = J_{cell}^{(place)}
           *h_{cell}^{(place)}
                                                                                                                         
           +(1/sqrt(percell)) *\sum_i{(J_{cell, i}^{(nonplace)}*h_{cell, i}
           ^{(nonplace)})

    --------------------------------------------------
    Output:
    J array filled with couplings
    """
    # fill couplings array with entries
    # recall that choice is [0,1]

    if placeprob[1] != 'None':
        J[:N, :] = gamma(vjplace, sjplace, N)*np.diagflat(\
           np.random.choice(choice,(N,), p=placeprob))
    if nstim !=0 : 
        wnp = np.array(np.where(J[(np.diag_indices(J[:N,:].\
           shape[1]))] ==0)).flatten()
        J[N:,wnp] = np.random.normal(vj, sj, \
            J[N:, wnp].shape)* np.random.choice(choice, J[N:,wnp].shape, \
            p=npprob)
        countcells=np.array((np.count_nonzero(J[N:,wnp],axis=0)))
        percell=(phi/np.sqrt(np.mean(countcells)))
        J[N:,wnp] *= percell
        if bothprob[1] != 'None':
            counts=np.array((np.count_nonzero(J[:N,:],\
                axis=0),np.count_nonzero(J[N:,:], axis=0)))
            #print(counts[0].size, counts[1].size) 
            nonplacecell=np.array(\
                np.where(np.logical_and(counts[0,:] == 0.,\
                counts[1,:] != 0.))).flatten()
            J[nonplacecell,nonplacecell] += np.reshape(gamma(vjplace, sjplace,\
                J[nonplacecell,nonplacecell].size), J[nonplacecell,nonplacecell].shape)*\
                np.random.choice(choice, J[nonplacecell,nonplacecell].shape, \
                p=bothprob)
    return J

def placestim(x,v,vdev):
    
    """
    Fill place fields with e^((-(x-v))^2/vdev)
    recall that N is the number of cells, xmax is the length of the track
    --------------------------------------------------
    Inputs:
    x: array of measurement locations. shape: (N, xmax*dt)
    v: means of place fields. shape: (1,N)
    vdev: variances of place fields. shape: (1,N)
    --------------------------------------------------
    Output: 
    all place fields over time peried length xmax, interval dt. shape: (N, 
    xmax*dt)
    """
    
    return np.exp(-(x-v)**2/(2*vdev))

def fillplace(fields, x, v, vdev):
    
    """
    Fill fields array with place fields
    ---------------------------------------------------
    Inputs:
    x: array holding locations of measurement. shape: (N, xmax*dt)
    v: means of place fields. shape: (1,N)
    vdev: variances of place fields. shape: (1,N)
    --------------------------------------------------
    Output:
    fields array filled with place fields. shape: (loop, xmax, N+nstim)
    """
    
    calcfields = placestim(x,v.T,vdev.T) # keeps track of place fields
                                       # note that calcfields has shape (N, 
    # xmax) and we need shape (xmax, N)
                                       # so I take its transpose
            
    fields += (calcfields.T) # add place fields
                                 # to given fields array       
    return fields

def computeh(fields, J, eta, epsilon):
    """
    Fast computation of hamiltonian. Uses blis.py matrix multiplication.
    Note that here the maximum field value is subtracted off the hamiltonian
    ---------------------------------------------------
    Inputs:
    fields: fields array. shape (loop, xmax, N+nstim)
    J: coupling array. shape (N+nstim, N)
    --------------------------------------------------
    Output:
    fields array filled with place fields. shape: (loop, xmax, N+nstim)
    """
    h = blis_gemm(fields,J) # perform dot product to make hamiltonian
    # note that this above function using the external blis.py library.
    # it is faster that np.dot
    #h = np.dot(fields,J) # perform dot product to make hamiltonian
    h += epsilon
    h *= eta
    return h

def computeP(h):
    """
    Compute probablilities of silence given hamiltonian array
    ---------------------------------------------------
    Inputs:
    h: hamiltonian array. shape (loop, xmax, N)
    --------------------------------------------------
    Output:
    P(silence) array. shape: (loop, xmax, N)
    """   
    return 1./(1+np.exp(h)) # compute the probability of silence

def spikesfake(num,pmat):
    """
    Create fake activity array. 
    Use a tree structure: take activity for a single cell, make two copies, 
    flip num spins in each copy. save. 
    iterate until desired dimensions are acheived.
    ---------------------------------------------------
    Inputs:
    num: number of cells to flip each loop in each activity copy. integer.
    pmat: full activity array to take a single cell's activity out of to 
    iterate over. shape: (loop, xmax, N)
    --------------------------------------------------
    Output:
    fields array filled with place fields. shape: (loop, xmax, N+nstim)
    """
    num=2
    begin=np.reshape(pmat[0,:], (1, pmat.shape[1]))
    i=0
    while i < pmat.shape[0]:
        contain=[]
        for i in range(begin.shape[0]):
            pmatfake=np.vstack((begin[i,:], begin[i,:]))
            inds=np.random.choice(pmatfake.shape[1], (2,num))
            pmatfake[0,inds[0,:]]=-pmatfake[0,inds[0,:]]+1
            pmatfake[1,inds[1,:]]=-pmatfake[1,inds[1,:]]+1
            contain.append(pmatfake)
        begin=np.vstack(np.array(contain))
        i = begin.shape[0]
    return begin

def nonzerocell(pmat, size):
    """
    Remove silent cells from ensemble.
    -----------------------------------------------------
    Inputes:
    pmat: activity array shape shape:(Number of cells, number of time steps)
    ------------------------------------------------------------------
    Output:
    pmatnew: activity array with silent cells removed shape:(Number of cells, number of time steps)
    """
    means=(np.mean(pmat, axis=1))
    wh=np.where(means>0.)[0]
    print(str(pmat.shape[0]-wh.size) +  ' cells were silent and therefore removed')
    select=np.random.choice(wh.size, size=size, replace=False)
    return wh[select]

def bootcell(arr, keeps):    
    num=int(arr.shape[1]/4)
    inds=np.random.randint(low=num, high=arr.shape[1], size=4)
    s=[]
    for i in range(4):
        binds=(nonzerocell(arr[:, inds[i]-num:inds[i]], keeps))
        s.append(arr[binds, inds[i]-num:inds[i]])
    return s


