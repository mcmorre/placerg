#objects.py

import numpy as np
import copy
from placerg.funcs import *
from placerg.funcsrg import *
import placerg._dice6 as _dice6

class expsum:
    def __init__(self, tau, tauerr, mu, muerr, alpha, alphaerr, beta, betaerr, z, zerr):
        self.tau=tau
        self.tauerr=tauerr
        self.mu=mu
        self.muerr=muerr
        self.alpha=alpha
        self.alphaerr=alphaerr
        self.beta=beta
        self.betaerr=betaerr
        self.z=z
        self.zerr=zerr

class bootstrap:
    def __init__(self, rate, coeff, eigspec, var, psil, actmom, autocorr, tau, mu, alpha, beta, z):
        self.rate=rate
        self.coeff=coeff
        self.eigspec=eigspec
        self.var=var
        self.psil=psil
        self.actmom=actmom
        self.autocorr=autocorr
        self.tau=tau
        self.mu=mu
        self.alpha=alpha
        self.beta=beta
        self.z=z

class recordall:
    def __init__(self, hamx, ham, probx, \
        prob, ratex, rate, rateerr, coeffx, coeff, coefferr, shuffcoeffx, shuffcoeff, eigspecx,\
        eigspec, eigspecerr, varx, var, varerr, psilx, psil, psilerr, actmomx, actmom, actmomerr, autocorrx,\
                autocorr, autocorrerr, tau, tauerr, mu, muerr, alpha, alphaerr, beta, betaerr, z, zerr, phi, eta, epsilon, percell, stim, timeconst, labeltype, label):
        self.hamx=hamx
        self.ham=ham
        self.probx=probx
        self.prob=prob
        self.ratex=ratex
        self.rate=rate
        self.rateerr=rateerr
        self.coeffx=coeffx
        self.coeff=coeff
        self.coefferr=coefferr
        self.shuffcoeffx=shuffcoeffx
        self.shuffcoeff=shuffcoeff
        self.eigspecx=eigspecx
        self.eigspec=eigspec
        self.eigspecerr=eigspecerr
        self.varx=varx
        self.var=var
        self.varerr=varerr
        self.psilx=psilx
        self.psil=psil
        self.psilerr=psilerr
        self.actmomx=actmomx
        self.actmom=actmom
        self.actmomerr=actmomerr
        self.autocorrx=autocorrx
        self.autocorr=autocorr
        self.autocorrerr=autocorrerr
        self.tau=tau
        self.tauerr=tauerr
        self.mu=mu
        self.muerr=muerr
        self.alpha=alpha
        self.alphaerr=alphaerr
        self.beta=beta
        self.betaerr=betaerr
        self.z=z
        self.zerr=zerr
        self.phi=phi
        self.eta=eta
        self.epsilon=epsilon
        self.percell=percell
        self.stim=stim
        self.timeconst=timeconst
        self.labeltype=labeltype
        self.label=label


class environ:
    def __init__(self, N0, N, loop, xmax, dt, nstim, percell, bothprob,\
                 placeprob, npprob, vj, vjplace, sj, \
                 sjplace, timeconst, phi, eta, epsilon):
        self.choice=np.array([0,1])
        self.N0=N0 # number of cells
        self.N=N # number of cells after removing silent ones
        self.loop=loop # number of loops run
        self.xmax=xmax # maximum track length
        self.dt=dt # time step
        self.nstim=nstim # number of nonplace stimuli
        self.percell=percell # average number of nonplace 
                                # stimuli assigned per cell
        self.bothprob=bothprob # probability that cell 
                                # is coupled to nonplace field. 
                                # [p(not coupled), p(coupled)] 
        self.placeprob=placeprob # probability that place cell is coupled 
                                # only to a place field. 
                                # [p(only coupled to a place field), 
                                # p(may be coupled to some 
                                # nonplace fields)]
        self.npprob=npprob # probability that place cell 
                                # is coupled 
                                # only to a place field. 
                                # [p(only coupled to a place field), 
                                # p(may be coupled to some 
                                # nonplace fields)]
        self.vj=vj # mean of the nonplace cell couplings. scalar 
        self.vjplace=vjplace # mean of the place cell couplings. scalar 
        self.sj=sj #standard deviation of the 
                                # nonplace cell couplings. scalar 
        self.sjplace=sjplace #standard deviation of the place 
                                # cell couplings. scalar 
        self.timeconst=timeconst # mean length of stochastic process 
                                # in track lengths

        self.phi=phi

        self.eta=eta # multiply by this constant 
                            # to increase overall activity of network
        self.epsilon=epsilon

        self.x = np.tile(np.arange(0,self.xmax, self.dt), (self.N0,1)) 
                            # stack N copies of measurement locations
        self.v=np.random.uniform(0, self.xmax, (1,self.N0)) 
                            # means of place fields. shape: (1,N)
        self.vdev=gamma(self.xmax/10., (self.xmax/20.), self.N0).T 
                            # variances of place fields. shape: (1,N)
        self.sigmas= np.full((self.nstim,), 1.) 
                            # standard deviation of stochastic process. 
                            # shape: (nstim,)
        self.taus= self.timeconst*self.xmax
                            #gamma(self.timeconst*self.xmax, \
                            #self.timeconststd*self.xmax, self.nstim).flatten() 
                            # time constant of stochastic process. 
                            # shape: (nstim,)
        self.vs=np.full((self.nstim,), 0.) 
                            # mean of the stochastic process. 
                            # shape: (nstim,)
        self.process= stim(self.taus, self.sigmas, self.dt, \
                    int(self.loop*self.xmax*self.dt)).T 
                            # this makes an array of shape 
                            # (loop*dt*xmax, nstim)
        self.J0=fillJ(np.zeros((self.N0+self.nstim, self.N0)), self.N0,\
                self.vjplace, self.sjplace, self.vj,\
                self.sj, self.nstim, self.placeprob, self.npprob,\
                self.bothprob, self.choice, self.phi) 
                            # coupling array. shape (N+nstim, N)
        self.fields=fillfields(self.N0, self.x, self.v, self.vdev, \
                np.zeros((self.loop, self.xmax, self.N0+self.nstim)),\
                self.process, self.loop) 
                            # fields array. shape (loop, xmax, N+nstim)
        self.h= computeh(self.fields, self.J0, self.eta, self.epsilon) # hamiltonian array. 
                            # shape (loop, xmax, N)
        self.P= computeP(self.h) #  P(silence) array. 
                            # shape: (loop, xmax, N)
        self.pmatprocess0=_dice6.dice6(self.P) # unreshaped activity 
                            # array, has shape of P: 
                            # shape: (loop, xmax, N)
                            # note that this command is the same as the one below,
                            # but using the faster custom function dice6
        #self.pmatprocess0=spikesbetter(self.P) # unreshaped activity 
                            # array, has shape of P: 
                            # shape: (loop, xmax, N)
        self.pmat0=np.vstack(self.pmatprocess0).T
        self.inds=nonzerocell(self.pmat0, self.N)
        self.boots=bootcell(self.pmat0, self.N)
        self.pmat=self.pmat0[self.inds,:] # reshape activity array, 
                            # shape (N, loop*xmax*dt)
        self.J=np.vstack((self.J0[self.inds,:][:,self.inds], self.J0[self.N0:, self.inds]))
        self.h=self.h[:,:,self.inds]
        self.P=self.P[:,:,self.inds]
        self.p=np.vstack(self.P).T # make (N,loop*dt*xmax) array of 
                            # probabilties
        self.fields=np.dstack((self.fields[:,:,self.inds], self.fields[:,:,self.N0:]))

        self.counts=np.array((np.count_nonzero(self.J[:self.N,:],\
                axis=0),np.count_nonzero(self.J[self.N:,:], axis=0))) 
                            # count number of nonzero entries in 
                            # J[:N,:] (for place cells), 
                            # J[N:,:] for nonplace cells. 
                            # Returns array 
                            # [(number of cells with place fields),
                            #(number of cells with nonplace fields)] 

        self.placecell=np.array(\
                np.where(np.logical_and(self.counts[0,:] != 0.,\
                self.counts[1,:] == 0.))).flatten()
                                                                                     
                            # holds J indices for cells with 
                            # with only place fields
        self.bothcell= np.array(np.where(np.logical_and(self.counts[0,:] \
                != 0., self.counts[1,:] != 0.))).flatten() 
                                                                             
                            # holds J indices for             
                            # cells with both place and nonplace 
                            # fields
         
        self.nonplacecell=np.array(\
                np.where(np.logical_and(self.counts[0,:] == 0.,\
                self.counts[1,:] != 0.))).flatten()
                                                                                        
                            # holds J indices for cells
                            # with only nonplace fields

        self.pmatfake=spikesfake(10,self.pmat) 
                            # returns fake spike train array of shape 
                            # of pmat with 10 spin flips per iteration
        #self.randommat=(np.random.randint(2, size=self.pmat.shape)).astype(float)
        
    def grabstim(self,grabstim, num='all'): 
        """
        returns array holding all indices of cells coupled to a 
        given nonplace stimulus
        Inputs:
        grabstim: desired nonplace field, indexed [0:nstim). 
                  We want to find all cells coupled to this field.
        num: can specify how many total nonplace fields the returned
             cells are coupled to. 
             if "all", all cells coupled to the given field are returned. 
             If num=1, all cells coupled to only the given field 
             grabstim are returned
        -------------------------------------------------------------
        indices of cells coupled to grabstim with specified number of
        total nonplace stimuli. shape (number of cells coupled, )
        """
        if num=='all':
            searchJ=self.J[self.N+grabstim, :] 
                            # grab couplings for a nonplace field 
            result=np.array(np.where(np.abs(searchJ) != 0)).flatten() 
                            # return indices of all cells coupled to this 
                            # field
        else:
            result=np.array(np.where(\
                np.logical_and(self.counts[1]==num ,\
                np.abs(self.J[self.N+grabstim, :]) != 0.))).flatten() 
                # return indices of cells coupled to num 
                # nonplace fields, one of which is 
                                                                                                                                
                # the grabbed field
        return result

    def countstim(self, inds='all'): 
        """
        returns histogram of counts of cells specified by inds vs number
        of fields each cell is coupled to
        Inputs:
        inds: indices of cells which I want to include in the resulting
              histogram
          
        -------------------------------------------------------------
        Output: 
        histogram of counts of cells vs number of fields the each 
        cell is coupled to
        Note that the output is of the shape generated by np.histogram,
        so it consists of 2 arrays, the first with counts and the second
        with bins.
        """
        if inds=='all':
            result=self.counts[0,:]+self.counts[1,:]
            result=np.histogram(result, bins=np.arange(np.max(result)+2))
        else:
            result=self.counts[0,:]+self.counts[1,:]
            result=result[inds]
            result=np.histogram(result, bins=np.arange(np.max(result)+2))
        return result
            
    def grabatt(self, cell, att='field'):
        """
        returns desired attributes of the specified cells in an array
        Inputs:
        cell: indices of cells which I want to extract attributes for
        att: the desired attribute. 'field' returns fields of the specified 
             cells, 'pmat' returns spike trains of specified cells, 'pmatav'  
             returns mean activity across track runs for specified cells,
             'h' returns hamiltonians for specified cells, 'p' returns 
              probabilities of silence for specified cells, 'J' returns 
              couplings for specified cells, 'fieldind' returns indices 
              of fields which the specified cells are coupled to
        ------------------------------------------------------------------
        Output: 
        array holding the desired attribute for the desired cells
        """
        if att == 'field':
            wh=np.array(np.where(self.J[:,cell] != 0.)).flatten() 
            # grab field indices that cell is coupled to
            result= self.fields[:,:,wh] 
            # return fields that cell is coupled to in array
        if att == 'pmat':
            result = self.pmat[cell, :] 
            # return cell's activity row
        if att == 'pmatav':
            result = np.mean(self.pmatprocess[:,:,cell], axis=0) 
            # return cell's mean activity across track runs
        if att == 'h':
            result = self.h[:,:, cell] 
            # return cell's hamiltonian as a function of time
        if att == 'p':
            result = self.p[cell, :] 
            # return cell's probability of silence as a function of time
        if att == 'J':
            result = self.J[:, cell] # return cell's couplings
        if att == 'fieldind':
            result = np.array(np.where(J[:,cell] != 0.)).flatten() 
            # return indices of fields cell is coupled to
        return result

    def grabcoup(self, cell, field, att='positive'):
        """
        grab indices of cells with couplings sorted by sign for a specified 
        nonplace field
        Inputs:
        cell: specified cell(s) index/indices whose coupling to specified 
              filed will be returned
        field: specified nonplace field to which specified cell(s) coupling 
               will be returned indexed [0:nstim)
        att: return positive or negative or all indices of given cells with 
             couplings to given nonplace field
        --------------------------------------------------------------------
        Output:
        Either empty list [] if no couplings obeying given constraints are 
        found or array holding indices of cells with couplings obeying given 
        constraints
        """
        if att == 'positive':
            inds=self.grabatt(cell, att='J')[self.N+field] 
        # grab groups of cells' coupling for field
            result=cell[np.where(inds>0.)] 
        # pass result if positive, pass [] if negative
            
        if att == 'negative':
            inds=self.grabatt(cell, att='J')[self.N+field] 
            # grab group of cells' coupling for field
            result=cell[np.where(inds<0.)] 
            # pass result if negative, [] if positive

        if att == 'all':
            result=self.grabatt(cell, att='J')[self.N+field] 
            # grab group of cells' coupling for field
        return result
    
    def corrnonplace(self, field, bins=6, att='positive'):
        """
        Calculate cross correlation of digitized nonplace field and average 
        cell activity, in which the cells average over are only coupled to the 
        given nonplace field. Here "digitized" refers to the
        following process: bins are created to span the field values and each 
        time point
        of the field time series is assigned to a bin.
        Inputs:
        field: index of desired nonplace field, indexed [0:nstim)
        bins: number of bins which will be used to digitize the field time 
              series
        att: select cells which are positively or negatively coupled to given 
             field. Recall that these cells are solely coupled to this field 
             and no other fields.
        -------------------------------------------------------------------
        Output:
        cross correlation of digitized nonplace field and average cell 
        activity, or 'None' if no cells solely coupled to the given field with 
        the specified sign restriction are found
        """
        cells=self.grabcoup(self.grabstim(field, num=1), field, att=att) 
        # grab couplings for cells coupled to 1 nonplace field for a given 
        # field, with either positive or negative coupling (att)
        wh=np.where(np.abs(self.J[self.N+field,cells]) > 1.5) 
        # select cells with greater than absolute value 1 coupling
        if np.array(wh).size==0:
            result='None!'
        else:
            cells=cells[wh]
            avg=np.vstack(self.grabatt(cells[0], att='field')).T.flatten() 
            # reshape the field into pmat shape (cell, loop*dt*xmax)
            avghist=np.histogram(avg, bins=bins) # make bins for field
            digavghist=np.digitize(avg,avghist[1]) # change the field array 
            # into array of bin assignments (digitize it)
            avgact=np.mean(self.grabatt(cells, att='pmat'), axis=0) 
            # calculate average activity across cells coupled only to this 
            # field with att couplings greater than 1
            result = crosscorr(digavghist,avgact) 
            # calculate cross correlation of digitized field and average cell 
            # activity
        return result
    
    def corrplace(self, num, bins=6):
        """
        Calculate cross correlation of digitized place field and place cell 
        activity. Here "digitized" refers to the
        following process: bins are created to span the field values and each 
        time point
        of the field time series is assigned to a bin.
        Inputs:
        num: index of desired place field, indexed [0:N)
        bins: number of bins which will be used to digitize the field time 
              series
        ------------------------------------------------------------------
        Output:
        cross correlation of digitized place field and cell activity
        """
        cell=self.placecell[num] # select place cell
        avgact=self.grabatt(cell, att='pmat') 
        # grab activity of place cell
        avg=np.vstack(self.grabatt(cell, att='field')).T.flatten() 
        # reshape place cell field
        avghist=np.histogram(avg, bins=bins) 
        # make bins for place cell field
        digavghist=np.digitize(avg,avghist[1]) 
        # digitize place cell field
        result = crosscorr(digavghist,avgact) 
        # calculate cross correlation between 
        # digitized place cell field and  cell activity
        return result



class infoset:
    def __init__(self, N, pmat, k):
        """
        This object, named a in the simulation, is responsible for holding 
        the necessary results and attributes of the RG analysis.
        ----------------------------------------------------------------------
        Inputs: 
        N: number of cells in network
        pmat: activity array holding spike trains for all cells. 
              shape (N, loop*xmax*dt)
        k: number of RG steps taken
        """
        self.N=N # number of cells
        self.k=k # number of RG steps taken
        self.pmat=pmat 
        # activity array holding spike trains for all cells. 
        # shape (N, loop*xmax*dt)
        self.cluster=np.reshape(np.arange(self.N), (np.arange(self.N).size,1)) 
        # empty cluster array holding 0th rg iteration's indices
        self.clusterlist=selectstep(calcrg\
            (self.pmat,self.cluster, self.k)[1], self.k) 
        # array holding cell indices for each cluster at each RG step
        # stuff only neede for momentum space RG
        self.flucs=fluc(self.pmat) 
        # array holding each cells' fluctuation from mean firing rate 
        self.eigvector=eigmom(self.pmat) 
        # array holding all covariance array's eigenvectors
    
    def pmatarr(self,i): 
        """
        returns array holding activity of all clusters at given RG step i
        --------------------------------------------------------
        Inputs:
        i: desired RG step at which we want activity of all clusters
        -------------------------------------------------------
        Output: array of shape (number of clusters, xmax*dt*loops) 
                in which the number of clusters is N/2**k
        """
        return calcrg(self.pmat,self.cluster, i)[0]
    
    def clusterstep(self,i): 
        """
        returns the list of indices in each cluster at given RG step i
        ---------------------------------------------------------------------------------
        Inputs:
        i: desired RG step at which we want activity of all clusters
        -------------------------------------------------------------------------------
        Output: array of shape (number of clusters, number of cells in each cluster)
                in which the number of clusters is N/2**i and the number of cells 
                in cluster is 2**i   
        """
        return self.clusterlist[self.k-i] 
        # [k-i], because recall that clusterlist is in reverse order of RG step
    
    def clustertrain(self, i, j): 
        """
        returns spike trains for cells within a given cluster j at RG step i
        Inputs:
        i: desired RG step 
        j: desired cluster index
        --------------------------------------------------------
        Output: array of shape (number of cells in cluster, xmax*loops*dt) 
                in which the number of cells in cluster is 2**i       
        """
        #print(self.pmat.shape)
        return self.pmat[self.clusterstep(i)[j, :],:]
    
    def corrnew(self, i, j): 
        """
        returns correlation matrix for members of cluster j and RG step i
        Inputs:
        i: desired RG step 
        j: desired cluster index
        ----------------------------------------------------------
        Output: array of shape (number of cells in cluster j at RG step i, 
                number of cells in cluster j at RG step i) in which the 
                number of cells in cluster is 2**i    
        """
        data= self.clustertrain(i,j)
        data -= np.reshape(data.mean(axis=1), (data.shape[0], 1))
        return np.corrcoef(data)

    def covnew(self, i, j): 
        """
        returns covariance matrix for members of cluster j and RG step i
        Inputs:
        i: desired RG step 
        j: desired cluster index
        ----------------------------------------------------------
        Output: array of shape (number of cells in cluster j at RG step i, 
                number of cells in cluster j at RG step i) in which the 
                number of cells in cluster is 2**i    
        """
        data= self.clustertrain(i,j)
        data -= np.reshape(data.mean(axis=1), (data.shape[0], 1))
        return np.cov(data)
        
    def eigsnew(self, i, j): 
        """
        returns sorted eigenvalues for correlation matrix for 
        members of cluster j and RG step i
        Inputs:
        i: desired RG step 
        j: desired cluster index
        -----------------------------------------------------------
        Output: eigenvalues of the correlation matrix of shape 
                (number of cells in cluster j at RG step i, 
                number of cells in cluster j at RG step i) in 
                which the number of cells in cluster is 2**i   
        """
        return eiggen(self.covnew( i , j))

    def eigsnewcorr(self, i, j): 
        """
        returns sorted eigenvalues for correlation matrix for 
        members of cluster j and RG step i
        Inputs:
        i: desired RG step 
        j: desired cluster index
        -----------------------------------------------------------
        Output: eigenvalues of the correlation matrix of shape 
                (number of cells in cluster j at RG step i, 
                number of cells in cluster j at RG step i) in 
                which the number of cells in cluster is 2**i   
        """
        return eiggen(self.corrnew( i , j))

    def varnewover(self, i):    
        """
        returns variance of activity over all clusters at RG step i
        Inputs:
        i: desired RG step at which we want activity of all clusters
        -------------------------------------------------
        Output: variance of over all coarse grained variables at RG step i    
        """
        result=[]
        for j in range(int(self.N/(2**i))):
            result.append(np.var(self.clustertrain(i,j)))
        result=np.mean(np.array(result))
        return result

    def probgen(self, i,j):
        """
        For cluster j, calculate probability that every cell in cluster is silent 
        --------------------------------------------------------------------------------------
        Inputs:
        i: desired RG step 
        j: desired cluster index
        --------------------------------------------------------
        Output: probability that cluster j is silent at RG step i
        """

        pmat=self.clustertrain(i, j)  # returns spike trains for a given cluster j at RG step i
        calc=(1.-np.mean(pmat)) # calculate probability of silence within cluster
        return calc


def orderplot(allo):
    """
    Force the recordall object to have data sets in order of parameter sweep value
    Look at placerg.objects to find structure of recordall object.
    For example, suppose you do a parameter sweep of the parameter 'epsilon' and record the data
    from all your simulations into a single recordall object. The analysis notebooks take this object and 
    generate plots from each attribute of the recordall object. You want each of your subplots to appear in order 
    of increasing epsilon. So use this function as demonstrated in the analysis notebooks to do that!
    -------------------------------------------
    Inputs:
    allo: a recordall object
    """
    subs=[allo.hamx, allo.ham, allo.probx, \
        allo.prob, allo.ratex, allo.rate, allo.rateerr, allo.coeffx, allo.coeff, allo.coefferr, allo.shuffcoeffx, \
        allo.shuffcoeff, allo.eigspecx,\
        allo.eigspec, allo.eigspecerr, allo.varx, allo.var, allo.varerr, allo.psilx, allo.psil, allo.psilerr,\
        allo.actmomx, allo.actmom, allo.actmomerr, allo.autocorrx,\
        allo.autocorr, allo.autocorrerr, allo.tau, allo.tauerr, allo.mu, allo.muerr, allo.alpha, allo.alphaerr, \
        allo.beta, allo.betaerr, allo.z, allo.zerr, allo.timeconst]
    sorts=allo.label
    argsorts=np.argsort(allo.label)
    for i in range(len(subs)):
        subs[i]=[x for _,x in sorted(zip(sorts,subs[i]))]
    hamx, ham, probx, \
        prob, ratex, rate, rateerr, coeffx, coeff, coefferr, shuffcoeffx, shuffcoeff, eigspecx,\
        eigspec, eigspecerr, varx, var, varerr, psilx, psil, psilerr, actmomx, actmom, actmomerr, autocorrx,\
        autocorr, autocorrerr, tau, tauerr, mu, muerr, alpha, alphaerr, beta, betaerr, z, zerr, timeconst = subs
    allo.hamx=hamx
    allo.ham=ham
    allo.probx=probx
    allo.prob=prob
    allo.ratex=ratex
    allo.rate=rate
    allo.rateerr=rateerr
    allo.coeffx=coeffx
    allo.coeff=coeff
    allo.coefferr=coefferr
    allo.shuffcoeffx=shuffcoeffx
    allo.shuffcoeff=shuffcoeff
    allo.eigspecx=eigspecx
    allo.eigspec=eigspec
    allo.eigspecerr=eigspecerr
    allo.varx=varx
    allo.var=var
    allo.varerr=varerr
    allo.psilx=psilx
    allo.psil=psil
    allo.psilerr=psilerr
    allo.actmomx=actmomx
    allo.actmom=actmom
    allo.actmomerr=actmomerr
    allo.autocorrx=autocorrx
    allo.autocorr=autocorr
    allo.autocorrerr=autocorrerr
    allo.tau=tau
    allo.tauerr=tauerr
    allo.mu=mu
    allo.muerr=muerr
    allo.alpha=alpha
    allo.alphaerr=alphaerr
    allo.beta=beta
    allo.betaerr=betaerr
    allo.z=z
    allo.zerr=zerr
    allo.timeconst=timeconst
    allo.label=np.array(allo.label)[argsorts]
    allo.phi=np.array(allo.phi)[argsorts]
    allo.eta=np.array(allo.eta)[argsorts]
    allo.epsilon=np.array(allo.epsilon)[argsorts]
    allo.percell=np.array(allo.percell)[argsorts]
    allo.stim=np.array(allo.stim)[argsorts]
   

