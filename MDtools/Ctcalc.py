#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is a collection of functions to attempt to standarize our calculation of
correlation functions. 

For calculation of correlation functions, we generally have a sum of the form
C(t)=sum(c_i*<a_i(tau)*a_i(t+tau)>), where we must sum over i. We could just
implement a calculator for the linear correlation functions and add up the 
result elsewhere, but this ignores a few things that make could make this faster.

Our main goal is to make calculation of the linear correlation functions with
Fourier transforms as fast as possible, but we will also implement some sparse
sampling approaches.

Note we set values a and b, which if using Fourier Transform mode, are 
transformed to yield A and B, respectively, and are eventually used to 
calculate the correlation function.

1) We can take the inverse Fourier transform of the product OUTSIDE of the sum,
in some cases reducing the number of inverse transforms by ~81 fold, reducing 
computational time by about 1/3.
    -Note that for the most difficult calcultions, there are 2*81 FTs and 81 IFTs
    
2) For the C_00(t) correlation function (motion in f), a and b are the same, so
we can reduce the number of forward transforms by 1/2

3) For the C_00,C_01, C_02 correlation functions, components of ZZ always appear
and therefore can be recycled over the three correlation functions.

Then, we will create an iterable Ct object, with fields a and b. For a single
correlation function, one simply sets a and b equal to the two terms to be
correlated, and optionally sets a weighting term, c. 

ctf=Ct_fast(index=None,mode='auto') #index is for sparse sampling, 
                                    #mode is the type of ct calc

ctf.a=a
ctf.b=b
ctf.c=c     #Optional, initialized value is 1

Then, after entry, we say add, which adds the results of this term to the 
correlation function

ctf.add() 

We may add additional terms using the same procedure.

Finally, we call return, which yields the result.

ctf=ctf.return(offset=0)

We may request an offset, for example for C_00, which just adds on a term before
return the result.


Thus far, we have resolved problem 1). 

Problem 2) is simply resolved. If we set B to None-or just don't define it-
then we correlate A with itself.

Finally, to resolve Problem 3), we make Ct_fast iterable. The critical point is
this: ctf[k].A  is the same for all iterations of ctf, but ctf[k].B depends on
the iteration. Note when using FT, the complex conjugate is applied to elements
in B, not in A.


for a,b,c in zip(agen,bgen,cgen):
    ctf.a=a
    for k in range(3):
        ctf[k].b=b[k]
        ctf[k].c=c[k]
        ctf[k].add()

offset=[-1/2,0,0]
ct=[ctf[k].return(off) for k,off in zip(range(3),offset)]

Here, Agen, Bgen, and cgen are generator objects (or anything you can loop),
and we get a different set of these for all terms in the correlation function.
A returns just one element, but B and c may contain different values for each
correlation function being calculated.

Note, ctf.a, ctf.b, ctf.c may always be assigned. ctf.B, ctf.c are just getting
assigned to different elements of a list. Which element is edited and returned
depends on the state of ctf._index. When we call ctf[k], it's just editing the
state of ctf._index, and then returning itself.

Also note, we do not store a, b, but rather its FT-unless we're not in FT mode,
so if you assign a, and then check its value, it will remain as None, but A 
will be assigned.
                                                    

Finally, we'll use class variables for storage of A and B so that they may
be accessed globally to accelerate parallelization

Created on Thu Feb  3 11:24:10 2022

@author: albertsmith
"""

#%% Imports
import numpy as np
import multiprocessing as mp
import pkg_resources
from pyDR import Defaults

installed = [pkg.key for pkg in pkg_resources.working_set]
if 'pyfftw' in installed:
    import pyfftw.interfaces.scipy_fft as fft
    from pyfftw.interfaces import cache
    cache.enable()
    cache.set_keepalive_time(60)
else:
    print('Warning: pyfftw not installed, reverting to scipy')
    from scipy import fft

if 'numba' in installed:
    from numba import njit
else:
    def njit(fun,*args,**kwargs):
        return fun
    print('Warning: numba not installed. Consider installing for faster calculation of correlation functions')

dtype=Defaults['dtype']
dtypec=Defaults['dtype_complex']


"""
A few notes on Fourier transforms and correlation functions
    1) The FT of a real signal is Hermitian
    2) The complex conjugate of a Hermitian FT remains Hermitian
    3) The product of two Hermitian FTs is still Hermitian
    4) The IFT of a Hermitian signal is real
    5) The addition of an imaginary multiplier results in an FT that is
    antihermitian
    6) Antihermitian*Hermitian is Antihermitian
    7) The IFT of the Antihermitian signal is imaginary
    
    Then, we ought to be able to use the real Fourier transform or the Hermitian
    Fourier transforms instead of the full fft and ifft
    (but how? And why do they have an extra data point?)
    
    Some notes on real and hermitian ffts
    1) The real fft equals the complex conjugate of the inverse hermitian fft
    multiplied by the number of data points
        rfft(x)==ihfft(x).conj()*x.shape[-1]
        rfft(x).conj()/x.shape[-1]=ihfft(x)
        (note, these functions require real inputs)
    2) Similarly, the ihfft can be inverted either by hfft or by first taking
    the conjugate, multiplying by the shape and applying the rfft, and vice versa
        rfft(ihfft(x).conj()*x.shape[-1])==x
        hfft(rfft(x).conj()/x.shape)==x
        
    We will assume that input into a is always real, but input into b may be
    complex. Then, we'll split the transforms such that we can apply only 
    real/hermite transforms, thus accelerating the calculation (hopefully?)
    
    
    We also use symmetry of the correlation functions to accelerate calculations
    By default, we assume correlation functions are symmetric in time.
"""

#%% Class for calculating correlation functions
class Ctcalc():
    """
    Class for calculating correlation functions. Assumption is that all required
    correlation functions are a sum of linear  correlations of the form:
        C(t)=sum(c*(a(tau),b(t+tau))) #Each term averaged over tau
    
    Then, in the ideal case, we can take the sum over linear correlation functions
    that correlate a with b, having weighting c. An offset of the correlation
    function may be included at output
    
    ctc=Ctcalc()
    for a0,b0,c0 in zip(a,b,c):
        ctc.a,ctc.b,ctc.c=a0,b0,c0
        ctc.add()
    ct=ctc.Return(offset=0)
    
    If only autocorrelation is required, we may calculate
    ctc=Ctcalc()
    for a0,c0 in zip(a,c):
        ctc.a,ctc.c=a0,c0
        ctc.add()
    ct=ctc.Return(offset=1/2)
    
    If multiple correlation functions are calculated where a is the same but
    b changes (let b and c contain lists of lists)
    
    ctc=Ctcalc(length=len(b[0])) #Setting length creates an iterable ctc. 
    #We can also index ctc, in which case it will be extended to be at least as long as the called element
    
    for a0,b0,c0 in zip(a,b,c):
        ctc.a=a0
        for b1,c1,ctc1 in zip(b0,c0,ctc): #We can also index ctc instead of iterating over it
            ctc1.b,ctc1.c=b1,c1
            ctc1.add()
    ct=[ctc1.Return(offset=off) for ctc1,off in zip(ctc,offset)] #Return all correlation functions
    
    If a sparse index is used, set ctc.index before assigning ctc.a and ctc.b
    
    mode deterermines how to calculate the correlation function 
    direct (d) and fourier transform (f) implemented. Default is auto (a)
    
    sym controls assumptions about the symmetry of the correlation functions. By
    default, we assume correlation functions are symmetric in time (sym='sym').
    Alternatively, set sym to '0p' or 'p0' depending on whether the correlation
    functions are for the initial or final component being 0.
    
    index, mode, length, and sym must be set at initialization.
    """
    def __init__(self,mode='auto',index=None,length=1,dt=0.005,calc_ct=True,sym='sym'):
        self.A=None     #Storage for Fourier Transformed input
        self.B=list()   #Storage for Fourier Transformed input
        self.a=None     #Storage for direct input
        self.b=list()   #Storage for direct input
        self.c=list()   #Storage for scaling factor
        self.CtFT=list()    #Storage for product of correlation function
        self.Ct=list()      #Storage for correlation function
        self.aEq=None   #Storage for the equilibrium values
        self.bEq=list() #Storage for the equilibrium values      
        self.Eq=list()  #Storage for the equilibrium values 
        self.N=None if index is None else index[-1]+1
        self._i=0       #This is the index to determine which correlation function we're using right now
        self.calc_ct=calc_ct
        assert sym in ['sym','0p','p0'],"sym must be 'sym','0p',or 'p0'"
        self.sym=sym        #'0p','p0','sym'
        if index is not None and np.diff(index).max()>1:    #Check that we actually need index
            self._index=index
        else:
            self._index=None
        assert mode[0].lower() in ['a','d','f'],"mode must be 'a' (auto), 'd' (direct), or 'f' (Fourier transform)"
        self.__mode=mode[0].lower()
        
        self.nc=mp.cpu_count()
        self.parallel=self.nc>1
        self[length-1]
        self.dt=dt
        
        #Some setup for sparse sampling
        super().__setattr__('i1',None)
        if self._index is not None:
            super().__setattr__('count',get_count(index))
            super().__setattr__('i',self.count!=0)
            super().__setattr__('i1',(np.cumsum(self.i)-1).astype(int))
            
    
    @property
    def mode(self):
        if self.__mode=='a':
            #We could put something more advanced here for sparse sampling
            if self._index is None or np.diff(self.index).max()==1:return 'f'
            return 'd'
        return self.__mode[0].lower()
    
    @property
    def index(self):
        "Index for sparse sampling of the trajectory"
        if self._index is not None:return self._index
        assert self.N is not None,"Cannot call self.index until 'index' or 'a' is set"
        return np.arange(self.N)
    
    
    @property
    def norm(self):
        return np.arange(self.N,0,-1) if self._index is None else self.count
    
    @property
    def t(self):
        return self.dt*np.arange(self.index[-1]+1)[self.norm!=0]
    
    
    
    def __setattr__(self,name,value):
        """
        Controls for setting different variable types
        """
        
        if name in ['index','count','i','i1']:
            print('Warning: access to {0} is restricted'.format(name))
            return
        
        if (name in ['A','a'] and value is None) or (name in ['B','b','c','CtFT','Ct'] and isinstance(value,list)):
            super().__setattr__(name,value)
            return
        
        if name in ['A','B']:
            print('Warning: {0} should not be set directly (not set)'.format(name))
            return
        
        if name in ['a','b']:
            #Storage for equilibrium values
            if name=='a':
                self.aEq=value.mean(-1)
            else:
                self.bEq[self._i]=value.mean(-1)
                
            if not(self.calc_ct):return #Exit now if we're only calculating equilibrium values
            #Preparation of the data for sparse sampling
            if self._index is not None and np.diff(self.index).max()>1 and self.mode=='f':
                value0=np.zeros([*value.shape[:-1],self.N],dtype=dtypec if np.iscomplexobj(value) else dtype).T
                value0[self.index]=value.T
                value=value0.T
            elif self.mode!='f':
                value=value.T
                
            #Storage of the values or their Fourier transforms
            if self.mode=='f':
                if name=='a':
                    super().__setattr__('A',fft.rfft(value,n=value.shape[-1]<<1,axis=-1))
                    self.N=value.shape[-1]
                elif name=='b':
                    if np.iscomplexobj(value):
                        self.B[self._i]=[fft.ihfft(value.real,n=value.shape[-1]<<1,workers=self.nc)*(value.shape[-1]<<1),\
                                              fft.ihfft(value.imag,n=value.shape[-1]<<1,workers=self.nc)*(value.shape[-1]<<1)]
                    else:
                        self.B[self._i]=fft.ihfft(value.real,n=value.shape[-1]<<1,workers=self.nc)*(value.shape[-1]<<1)
                return
            else:
                if name=='a':
                    self.N=value.shape[0] if self._index is None else self.index[-1]+1
                    super().__setattr__(name,value)
                else:
                    self.b[self._i]=value
                return
        
        #Storage in lists
        if name in ['c','CtFt','Ct']:
            getattr(self,name)[self._i]=value
        else:
            super().__setattr__(name,value)
            
    def add(self):
        "Here we check that the equilibrium values are preallocated"
        if self.Eq[self._i] is None:
            if self.bEq[self._i] is not None and np.iscomplexobj(self.bEq[self._i]):
                self.Eq[self._i]=np.zeros(self.aEq.shape,dtype=dtypec)
            else:
                self.Eq[self._i]=np.zeros(self.aEq.shape,dtype=dtype)
        
        self.Eq[self._i]+=self.aEq*self.bEq[self._i]*self.c[self._i] \
            if self.bEq[self._i] is not None else self.aEq**2*self.c[self._i]
        if not(self.calc_ct):
            self.B[self._i]=None
            self.b[self._i]=None
            self.c[self._i]=1
            return #Exit now if only calculating equilibrium values
        
        "Here we check that the correlation functions are preallocated"
        if self.mode=='f':
            if self.CtFT[self._i] is None:
                if self.B[self._i] is not None and len(self.B[self._i])==2:
                    self.CtFT[self._i]=[np.zeros(self.A.shape,dtype=dtypec) for _ in range(2)]
                else:
                    self.CtFT[self._i]=np.zeros(self.A.shape,dtype=dtypec)   
        else:
            if self.Ct[self._i] is None:
                if self._index is None:
                    self.Ct[self._i]=np.zeros([self.N,*self.a.shape[1:]],dtype=dtype)
                else:
                    self.Ct[self._i]=np.zeros([np.sum(self.i),*self.a.shape[1:]],dtype=dtype)
        
        "Add the contribution from the current iteration"
        if self.mode=='f':
            if self.B[self._i] is not None:
                if self.B[self._i].__len__()==2:
                    for ctft,B in zip(self.CtFT[self._i],self.B[self._i]):
                        if self.sym=='sym': #Don't calculate imaginary components
                            ctft+=(self.A.real*B.real-self.A.imag*B.imag)*self.c[self._i]
                        elif self.sym=='0p':
                            ctft+=self.A*B*self.c[self._i]
                        else:
                            ctft+=(self.A*B).conj()*self.c[self._i]
                else:
                    if self.sym=='sym': #Don't  calculate imaginary components
                        self.CtFT[self._i]+=(self.A.real*self.B[self._i].real-\
                            self.A.imag*self.B[self._i].imag)*self.c[self._i]
                    elif self.sym=='0p':
                        self.CtFT[self._i]+=self.A*self.B[self._i]*self.c[self._i]
                    else:
                        self.CtFT[self._i]+=(self.A*self.B[self._i]).conj()*self.c[self._i]
                self.B[self._i]=None
                self.c[self._i]=1
            else:
                self.CtFT[self._i]+=(self.A.real**2+self.A.imag**2)*self.c[self._i] #A.conj()*A*c
        elif self.mode=='d':
            Ct_jit(self.Ct[self._i],self.index,self.i1,self.a,self.b[self._i],self.c[self._i])
            self.b[self._i]=None
            self.c[self._i]=1
                        
            
    def Return(self,offset=0):
        if self.Eq[self._i] is None:        #Nothing calculated
            return None,None
        if not(self.calc_ct):
            return None,self.Eq[self._i]+offset #Only eq. value calculated
        
        if self.mode=='f':
            if len(self.CtFT[self._i])==2:
                out=fft.irfft(self.CtFT[self._i][0]).T[:self.N].T+\
                    1j*fft.irfft(self.CtFT[self._i][1]).T[:self.N].T
            else:
                out=fft.irfft(self.CtFT[self._i]).T[:self.N].T
            self.CtFT[self._i]=None
            if self._index is not None:out=out.T[self.i].T
        else:
            out=self.Ct[self._i].T            
            self.Ct[self._i]=None

        out/=self.norm[self.i] if self._index is not None else self.norm
        dt_out=dtypec if np.iscomplexobj(out) else dtype        
        return out.astype(dt_out)+offset,self.Eq[self._i]+offset #Return correlation function and equilibrium
        
    def __getitem__(self,k):
        self._i=k
        while not(k<len(self)):
            self.B.append(None)
            self.b.append(None)
            self.c.append(1)
            self.CtFT.append(None)
            self.Ct.append(None)
            self.Eq.append(None)
            self.bEq.append(None)
        return self
    
    def __len__(self):
        return len(self.B)
    
    def __iter__(self):
        self._i=-1
        return self
    
    def __next__(self):
        self._i+=1
        if self._i<len(self):
            return self
        else:
            self._i=0
            raise StopIteration
    
    def reset(self):
        self.A=None
        self.B=list()
        self.a=None
        self.b=list()
        self.c=list()
        self.CtFT=list()
        self.Ct=list()
        self.aEq=None   #Storage for the equilibrium values
        self.bEq=list() #Storage for the equilibrium values      
        self.Eq=list()  #Storage for the equilibrium values 
        self[1]

#%% Functions for sparse sampling
def sparse_index(nt:int,n:int=-1,nr:int=10) ->np.ndarray:
    """
        Calculates a log-spaced sampling schedule for an MD time axis. Parameters are
nt, the number of time points, n, which is the number of time points to 
load in before the first time point is skipped, and finally nr is how many
times to repeat that schedule in the trajectory (so for nr=10, 1/10 of the
way from the beginning of the trajectory, the schedule will start to repeat, 
and this will be repeated 10 times)

    Parameters
    ----------
    nt : int
        Number of time points in the trajectory.
    n : int, optional
        Number of time points before the schedule skips a time point. Setting to
        -1 will return uniform sampling. The default is -1.
    nr : int, optional
        Number of repeats of the sampling schedule. The default is 10.

    Returns
    -------
    index : np.ndarray
        Array of integers indicating where the MD trajectory is sampled.
    """
    
    n=np.array(n).astype('int')
    nr=np.array(nr).astype('int')
    
    if n==-1:
        index=np.arange(nt)
        return index
    
    "Step size: this log-spacing will lead to the first skip after n time points"
    logdt0=np.log10(1.50000001)/n
    
    index=list()
    index.append(0)
    dt=0
    while index[-1]<nt:
        index.append(index[-1]+np.round(10**dt))
        dt+=logdt0
        
    index=np.array(index,dtype=int)

    "Repeat this indexing nr times throughout the trajectory"
    index=np.repeat(index,nr,axis=0)+np.repeat([np.arange(0,nt,nt/nr)],index.size,axis=0).reshape([index.size*nr])
    
    "Eliminate indices >= nt, eliminate repeats, and sort the index"
    "(repeats in above line lead to unsorted axis, unique gets rid of repeats and sorts)"
    index=index[index<nt]
    index=np.unique(index).astype('int')
    
    return index


def get_count(index):
    """
    Returns the number of averages for each time point in the sparsely sampled 
    correlation function
    """
    N=np.zeros(index[-1]+1)
    n=np.size(index)
   
    for k in range(n):
        N[index[k:]-index[k]]+=1
        
    return N


#%% Direct calculation with njit

# @njit(parallel=True)
def Ct_jit(ct,index,i1,x,y=None,c=1):
    """
    Calculate linear correlation function of x with itself or with y. Provide
    the pre-allocated correlation function (ct=np.zeros(...) if first call, otherwise
    one can accumulate the total correlatin function in from previous calls).
    index is also required (can be np.arange(ct.shape[0]))
    """
    if y is None:y=x
    if i1 is None:
        for k in range(x.shape[0]):
            ct[index[k:]-index[k]]+=c*x[k]*y[k:]
    else:
        for k in range(x.shape[0]):
            ct[i1[index[k:]-index[k]]]+=c*x[k]*y[k:]
    
#%% Rank-2 Correlation function (P2(cos(theta)))
def Ct(v):
    """
    Correlation function for rank 2 tensors. Input is a vector with x,y,z on
    the 0 axis and time on the -1 axis.
    """
    v/=np.sqrt((v**2).sum(0))
    ctc=Ctcalc()
    for k in range(3):
        for j in range(k,3):
            ctc.a=v[k]*v[j]
            ctc.c=1.5 if k==j else 3
            ctc.add()
    return ctc.Return(offset=-1/2)


    