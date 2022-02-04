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

Note, ctf.A, ctf.B, ctf.c may always be assigned. ctf.B, ctf.c are just getting
assigned to different elements of a list. Which element is edited and returned
depends on the state of ctf._index. When we call ctf[k], it's just editing the
state of ctf._index, and then returning itself.

Also note, we do not store A, B, but rather its FT-unless we're not in FT mode
                                                    

Finally, we'll use class variables for storage of A and B so that they may
be accessed globally to accelerate parallelization

Created on Thu Feb  3 11:24:10 2022

@author: albertsmith
"""

import numpy as np
import multiprocessing as mp

try:
    # assert 0,"Disable"
    import pyfftw.interfaces.scipy_fft as fft
    from pyfftw.interfaces import cache
    cache.enable()
    cache.set_keepalive_time(60)
    fftw=True
except:
    print('Warning: pyfftw not installed, reverting to scipy')
    from scipy import fft

from pyDR import Defaults
dtype=Defaults['dtype']


def nexpow2(n):
    n=int(n)
    return 1 if n==0 else 1<<(n-1).bit_length()

"""
A few notes on Fourier transforms and correlation functions
    1) The FT of a real signal is Hermitian
    2) The complex conjugate of a Hermitian FT remains Hermitian
    3) The project of two Hermitian FTs is still Hermitian
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
"""

class Ct_fast():
    def __init__(self,mode='auto',index=None):
        self.A=None
        self.B=list()
        self.a=None
        self.b=list()
        self.c=list()
        self.CtFT=list()
        self._index=0
        self.index=index
        self.in_shape=None
        self.N=None
        
        self.flag='fast'
        # self.flag='slow'
        
        self.nc=mp.cpu_count()
        self.parallel=self.nc>1
        self[0]
        
    def __setattr__(self,name,value):
        if value is None or (isinstance(value,list) and value.__len__()==0):
            super().__setattr__(name,value)
        elif name=='a':
            if self.flag=='slow':
                self.A=fft.fft(value,n=value.shape[1]<<1,axis=-1,workers=self.nc)
            else:
                self.A=fft.rfft(value,n=value.shape[-1]<<1,axis=-1)

            self.N=np.arange(value.shape[-1],0,-1)
        elif name=='b':
            if self.flag=='slow':
                self.B[self._index]=(value.shape[-1]<<1)*\
                    fft.ifft(value,n=value.shape[-1]<<1,axis=-1,workers=self.nc)
                #We use ifft to get the complex conjugate directly, 
                #but have to scale the result accordingly
            else:
                if np.iscomplexobj(value):
                    self.B[self._index]=[fft.ihfft(value.real,n=value.shape[-1]<<1,workers=self.nc)*(value.shape[-1]<<1),\
                                          fft.ihfft(value.imag,n=value.shape[-1]<<1,workers=self.nc)*(value.shape[-1]<<1)]
                else:
                    self.B[self._index]=fft.ihfft(value.real,n=value.shape[-1]<<1,workers=self.nc)*(value.shape[-1]<<1)
        elif name=='c':
            self.c[self._index]=value
        else:
            super().__setattr__(name,value)
            
            
    def add(self):
        CtFT=self.CtFT
        if self.CtFT[self._index] is None:
            if self.B[self._index] is not None and len(self.B[self._index])==2:
                self.CtFT[self._index]=[np.zeros(self.A.shape,dtype=complex) for _ in range(2)]
            else:
                self.CtFT[self._index]=np.zeros(self.A.shape,dtype=complex)                
        
        if self.B[self._index] is not None:
            if self.B[self._index].__len__()==2:
                self.CtFT[self._index][0]+=self.A*self.B[self._index][0]*self.c[self._index]
                self.CtFT[self._index][1]+=self.A*self.B[self._index][1]*self.c[self._index]
            else:
                self.CtFT[self._index]+=self.A*self.B[self._index]*self.c[self._index]
            self.B[self._index]=None
        else:
            CtFT[self._index]+=self.A.conj()*self.A*self.c[self._index]
            
    def Return(self,offset=0):
        if self.flag=='slow':
            out=fft.ifft(self.CtFT[self._index])[:,:self.CtFT[self._index].shape[1]>>1]
        else:        
            if len(self.CtFT[self._index])==2:
                out=fft.irfft(self.CtFT[self._index][0]).T[:self.N.size].T+\
                    1j*fft.irfft(self.CtFT[self._index][1]).T[:self.N.shape[-1]].T
            else:
                out=fft.irfft(self.CtFT[self._index]).T[:self.N.size].T
            
        self.CtFT[self._index]=None
        
        return out/self.N+offset
        
    def __getitem__(self,k):
        self._index=k
        while not(k<self.B.__len__()):
            self.B.append(None)
            self.b.append(None)
            self.c.append(1)
            self.CtFT.append(None)
        return self
    
    def reset(self):
        self.A=None
        self.B=list()
        self.a=None
        self.b=list()
        self.c=list()
        self.CtFT=list()
        self._index=0
        self.N=None
        self[0]
    
        

def Ct00(v):
    """
    Correlation function for rank 2 tensors. Input is a vector with x,y,z on
    the 0 axis and time on the -1 axis.
    """
    v/=np.sqrt((v**2).sum(0))
    ctf=Ct_fast()
    for k in range(3):
        for j in range(k,3):
            ctf.a=v[k]*v[j]
            ctf.c=1.5 if k==j else 3
            ctf.add()
    return ctf.Return(offset=-1/2)
