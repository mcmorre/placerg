#funcsrg.py

import numpy as np
import copy
from placerg.funcs import *


def RGrealstep(pmat, cluster, corr):
    """
    Perform real space RG step
    Here we first calculate the correlation matrix, c_(ij)= (C_(ij))/
   (sqrt(C_(ii)*C_(jj)))
    Where C is the covariance matrix
    A complication here is that if a cell i never fires or always fires, 
    C_(ii)=0. 
    Thus c_(ij) will be undefined.
    To deal with this I set Nans in the covariance matrix to 0. 
    
    I then set the diagonal to Nan so we do not count cells twice. Then pick 
    out maximally correlated
    cells and combine their activities, set the cell's corresponding rows and 
    columns to Nan.
    Then iterate until entire array is Nan.
    Update clusters at every iteration
    ---------------------------------------------------------
    Inputs:
    pmat: array holding all cells' spike trains. shape:(N, xmax*dt*loop)
    cluster: array holding the cells which make up each cluster. shape: (N,1)
    corr: correlation matrix, note that this may have Nans in it
    ----------------------------------------------------
    Output: RG transformed activity array. shape: (N/(2**i), xmax*dt*loop), 
            updated cluster array. shape: (N/(2**i), 2**i)
    
    """
    j=0
    corr1=copy.deepcopy(corr) # make a copy of correlations for processing
    corr1[np.where(np.isnan(corr1)==True)]=0. # set Nans to 0
    np.fill_diagonal(corr1, None) # set diagonal to Nan so we dont double 
    # count cells
    pmat1=copy.deepcopy(pmat) #make a copy of spike trains for processing
    pmatnew=np.zeros((int(pmat1.shape[0]/2), pmat1.shape[1])) #holds post RG 
    # step activity
    clusternew=np.zeros((int(pmat1.shape[0]/2), 2*cluster.shape[1])) #holds 
    # post RG clusters
    while j != pmatnew.shape[0]: #while new activity array is not filled up
        maxp=np.nanmax(corr1.flatten()) #pick out maximum non-Nan correlation
        wh=np.array(np.where(corr1==maxp)) #pick out indices where max corr is 
        # present
        i=np.random.choice(np.arange(wh.shape[1])) #choose the random index of 
        # these indices
        #i=np.min(np.where(np.abs(wh[1]-j)==np.min(np.abs(wh[1]-j))))
        #now we have 2 maximally correlated cells, wh[0,i] and wh[1,i]
        #now set rows and columns corresponding to these cells to Nan
        corr1[wh[0, i], :]=None
        corr1[:, wh[0,i]]=None
        corr1[wh[1, i], :]=None
        corr1[:, wh[1, i]]=None
        #now add activities of our chosen cells wh[0,i] and wh[1,i] and update 
        # clusters
        calc=pmat1[wh[0, i], :]+pmat1[wh[1, i], :]
        pmatnew[j, :]=calc
        clusternew[j, :]= np.concatenate(np.array([cluster[wh[0,i], :],\
            cluster[wh[1,i], :]]), axis=None)
        j += 1 #we have completed a row of the new activity array, count it!
        if j== pmatnew.shape[0]: #break if we have completed counting
            break 
    return pmatnew.astype(int), clusternew.astype(int)

def calcrg(pmatnew, clusternew, k):
    """
    Perform real space RG step using RGrealstep(pmat, cluster, corr)
    --------------------------------------------------------
    Inputs:
    pmatnew: array holding all cells' spike trains. shape: (N, xmax*dt*loop)
    clusternew: array holding the cells which make up each cluster. 
    shape: (N,1)
    k: number of RG steps to be performed. shape: scalar
    -------------------------------------------------------
    Output: spike train for cell i. shape: (N/(2**i), xmax*dt*loop)
            updated cluster array. shape: (N/(2**i), 2**i)
    
    """
    corr=np.corrcoef(pmatnew) #calculate correlation matrix  
    for i in range(0, k): # for every RG step
        pmatnew, clusternew = RGrealstep(pmatnew, clusternew, corr) # perform 
        # RG step
        corr=np.corrcoef(pmatnew) # calculate new correlation matrix
    return pmatnew, clusternew

def selectstep(clusters, k):
    """
    Returns the indices of cells in each cluster at each RG step
    -----------------------------------------------------------
    Inputs:
    clusters: the resulting cluster array of the last (kth) RG step.
              shape: (N/(2**k), 2**k)
    k: total number of RG steps performed. shape: scalar
    -----------------------------------------------------------
    Output: array holding the cell indices in each cluster at each RG step.   
            Note in reverse order:
            first subarray is the last RG step, last subarray is the 0th RG 
            step
            shape: holds k arrays, each (N/(2**i), 2**i)
    """
    clusterlist=[]
    for i in 2**np.arange(k):
        clusterlist.append(np.vstack((np.hsplit(clusters, i)))) # split up 
        # into the clusters added at each step
    clusterlist.append(clusters.flatten().T) # append the original 
    # "clusters" (1 cell)
    return clusterlist

def eiggen(corr):
    """
    Calculate eigenvalues and sort largest to smallest
    ---------------------------------------------------------
    Inputs:
    corr: input correlation matrix. shape: (number of cells, number of cells)
    ----------------------------------------------------------
    Output: sorted eigenvalues for correlation matrix corr. 
            shape: (number of cells,)
    """
    
    eigs=np.linalg.eig(corr)
    arg=np.argsort(eigs[0])[::-1]
    eigvals=eigs[0][arg]
    eigvecs=eigs[1][:,arg]
    return eigvals, eigvecs

# functions for the momentum space RG

def eigmom(pmat):
    """
    Calculate the eigenvectors in preperation for momentum space RG
    -----------------------------------------------------
    Inputs:
    pmat: the activity array of all cells' spike trains. 
          shape: (N, xmax*dt*loop)
    -------------------------------------------------------------
    Output: array of eigenvectors. Each eigenvector is a column in this array. 
            shape: (N,N)
    """
    corr=np.cov(pmat)
    #np.fill_diagonal(corr, 0.)
    corr[np.where(np.isnan(corr)==True)]=0.
    eigs=np.linalg.eig(corr) #calculate eigenvectors and values of original 
    #print(eigs[0].shape)
    # activity
    arg=np.argsort(eigs[0])[::-1] #get indices for sorted eigenvalues
    eigvec=eigs[1][:,arg] #sort eigenvectors
    return eigvec


def fluc(pmat):
    """
    Calculate fluctuations in preparation for projection onto chosen 
    eigenvectors for momentum space RG
    -----------------------------------------------------------
    Inputs:
    pmat: activity matrix holding all cells' spike trains
    ------------------------------------------------------------
    Output: array holding fluctuations away from mean for each cell
    """
    return pmat - np.reshape(np.mean(pmat, axis=1), (pmat.shape[0],1)) 

def RGmom(l, a):
    """
    Perform momentum space RG step
    --------------------------------------------------------------------------------------
    Inputs:
    l: total number of eigenvectors/l = number of eigenvectors I will 
        project fluctuations onto. shape:scalar
    a: object
    --------------------------------------------------------------------------------------
    Output: RG transformed activity array. shape: (N/l, xmax*dt*loop)
    """
    eigvec=a.eigvector[:,:int(a.eigvector.shape[1]/l)] #sort eigenvectors, cut out some
    ppmat=np.dot(eigvec,np.dot(eigvec.T,a.flucs))
    #print(ppmat.shape)
    #project fluctuations onto chosen eigenvectors
    return ppmat
def normmom(ppmat):
    """
    Makes the sum of squares of momentum space RG'd activity equal to 1
    ----------------------------------------------------------------------------
    Inputs: 
    ppmat: momentum space RG'd activity array.  shape: (N/l, xmax*dt*loop)
    --------------------------------------------------------------------------
    Output: normalized momentum space RG'd activity array.  shape: (N/l, xmax*dt*loop)
    """
    ppmatnew=np.empty(ppmat.shape)
    for i in range(ppmat.shape[0]): #enforce that sum of squares must be 1
        test=(np.sqrt(ppmat.shape[1])*ppmat[i,:])/(np.sqrt(np.sum(ppmat[i,:]**2)))
        vartest=np.mean(test**2)
        ppmatnew[i,:]=test
    return ppmatnew

