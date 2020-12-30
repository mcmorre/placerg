#funcsall.py

from placerg.funcs import *
from placerg.funcsrg import *
from placerg.objects import *
import copy

def fieldshist(env):
    pl=env.fields[:,:,env.nonplacecell].flatten()
    plott=drawpdf(pl, 100)
    return plott

def hamhist(env):
    pl=env.h[:,:,env.nonplacecell].flatten()
    plott=drawpdf(pl, 100)
    return plott


def probhist(env):
    pl=1-env.P.flatten()
    plott=drawpdf(pl, 100)
    return plott

def cellraterank(env):
    plott=(np.arange(1,env.N+1)/env.N,np.sort(np.mean\
            (env.pmat, axis=1))[::-1])
    return plott

def corrcoefhist(env):
    samp=np.corrcoef(env.pmat)
    shuff=copy.deepcopy(env.pmat)
    shuff=rollx(shuff)
    sampshuff=np.corrcoef(shuff)
    np.fill_diagonal(samp,0.)
    np.fill_diagonal(sampshuff,0.)
    samp=samp.flatten()
    sampshuff=sampshuff.flatten()
    samp[np.where(np.isnan(samp)==True)]=0.
    sampshuff[np.where(np.isnan(sampshuff)==True)]=0.
    pltcorr=drawpdf(samp,100)
    pltcorrshuff=drawpdf(sampshuff,100)
    result=np.array((pltcorr, pltcorrshuff))
    return result

def eigplt(i, a):
    """
    Plot eigenvalues from each successive RG step, averaged over all clusters
    --------------------------------------------------------------------------------------
    Inputs:
    i: number of RG step performed. shape: scalar
    --------------------------------------------------------------------------------------
    Output: cluster 1's eigenvalues from RG step i. shape: (N/(2**i),)
    
    """
    hold=[]
    for j in range(int(a.N/2**i)):
        hold.append(a.eigsnew(i, j)[0])
    hold=np.mean(np.vstack(hold), axis=0)
    return hold

def eigplotall(a):
    eigs=[]
    xplot=[]
    for i in np.arange(4,a.k+1):
        plot=eigplt(i,a)
        plot[np.where(plot < 1.*10**(-7))]=0.
        xplot.append(np.arange(1,plot.size+1)/(plot.size))
        eigs.append(plot)
    result=np.array((xplot,eigs))
    print('eigenvalue spectra analysis complete')
    return result

"""
calculate variance of activity at each RG step (over all clusters)
"""
def varpltover(i, a):
    """
    Calculate variance over all coarse grained variables (clusters) at RG step i.
    -------------------------------------------------------------------------------
    Inputs: 
    i: RG step
    -------------------------------------------------------------------------------
    Output: variance over all course grained variables at RG step i
    
    """
    varplot=np.var(a.pmatarr(i))
    #print(i)
    return varplot
def varplotall(a):
    varover=[]
    for i in range(a.k+1):
        varover.append(varpltover(i, a))
    result=(2**(np.arange(a.k+1)), varover)
    print('variance scaling analysis complete')
    return result

"""
Plot log probability of complete cluster silence vs cluster size
"""
def probplotall(a):
    probdata=[]
    probx=[]
    for i in range(a.k+1):
        whpr=np.where(a.pmatarr(i)==0)
        prob=np.array(whpr[1]).size
        contain=np.log(prob/a.pmatarr(i).size)
        probdata.append(contain)
        probx.append(2**i)
    probdata=(np.array(probdata))
    probx=np.array(probx)
    result=(probx, probdata)
    print('free energy scaling analysis complete')
    return result

def activemom(a):
    collectx=[]
    collect=[]
    for i in 2**(np.arange(4,a.k)):
        ppmatnew=normmom(RGmom(i,a))
        result=drawpdf(ppmatnew.flatten(), 100)
        collectx.append(result[0])
        collect.append(result[1])
    print('momentum space activity distribution analysis complete')
    return collectx, collect

"""
For each successive RG step, calculate average autocorrelation over all 
coarse grained variables 
"""

def calccorrreal(act, interval, mode='real'): 
    """
    Calculate average autocorrelation over all cells in given activity array
    """
    nx=act.shape[1]
    lags = []
    ys=[]
    for l in range(act.shape[0]):
        autocorrs = autocorr(act[l, :], norm=mode)
        ys.append(autocorrs[1])
        lags.append(autocorrs[0])
    y=np.vstack(np.array(ys))
    y=np.mean(y, axis=0)
    x=autocorrs[0]
    result=x[int(nx)-interval:int(nx)+interval], y[int(nx)-interval:int(nx)+interval]
    return result

def correalt(a, interval, i):
    """
    Calculate average autocorrelation over all cells in given activity array a.pmatarr(i)
    """
    act=(a.pmatarr(i)).astype('float')
    x,y=calccorrreal(act, interval, mode='real')
    return x,y

def calccorrmulti(a):
    inter=600
    result=[]
    for i in range(1, a.k):
        result.append(correalt(a,inter, i)[1])
    result=np.vstack(result)
    x=correalt(a,inter, 1)[0]
    print('dynamic scaling analysis complete')
    return x, result




