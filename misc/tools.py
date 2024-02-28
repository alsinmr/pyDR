#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 11:11:31 2021

@author: albertsmith
"""

from pyDR.Sens import Info
import os
import re
import numpy as np
from copy import copy



#%% Some tools for getting NMR information
class NucInfo(Info):
    """ Returns the gyromagnetic ratio for a given nucleus. Usually, should be 
    called with the nucleus and mass number, although will default first to 
    spin 1/2 nuclei if mass not specified, and second to the most abundant 
    nucleus. A second argument, info, can be specified to request the 
    gyromagnetic ratio ('gyro'), the spin ('spin'), the abundance ('abund'), or 
    if the function has been called without the mass number, one can return the 
    default mass number ('mass'). If called without any arguments, a pandas 
    object is returned containing all nuclear info ('nuc','mass','spin','gyro',
    'abund')
    """
    def __init__(self):
        h=6.6260693e-34
        muen=5.05078369931e-27
        super().__init__()
        dir_path = os.path.dirname(os.path.realpath(__file__))    
        with open(dir_path+'/GyroRatio.txt','r') as f:
            for line in f:
                line=line.strip().split()
                self.new_exper(Nuc=line[3],mass=int(line[1]),spin=float(line[5]),\
                               gyro=float(line[6])*muen/h,abundance=float(line[7])/100)
    
    def __call__(self,Nuc=None,info='gyro'):
        if Nuc is None:
            return self
        
        if Nuc=='D':
            Nuc='2H'
 
        #Separate the mass number from the nucleus type       
        mass=re.findall(r'\d+',Nuc)
        if not mass==[]:
            mass=int(mass[0])
                   
        Nuc=re.findall(r'[A-Z]',Nuc.upper())
        
        #Make first letter capital
        if np.size(Nuc)>1:
            Nuc=Nuc[0].upper()+Nuc[1].lower()
        else:
            Nuc=Nuc[0]
                        
        ftd=self[self['Nuc']==Nuc]  #Filtered by nucleus input
        
        "Now select which nucleus to return"
        if not mass==[]:    #Use the given mass number
           ftd=ftd[ftd['mass']==mass]
        elif any(ftd['spin']==0.5): #Ambiguous input, take spin 1/2 nucleus if exists
            ftd=ftd[ftd['spin']==0.5] #Ambiguous input, take most abundant nucleus
        elif any(ftd['spin']>0):
            ftd=ftd[ftd['spin']>0]
            
        ftd=ftd[np.argmax(ftd['abundance'])]
        
        if info is None or info=='all':
            return ftd
        else:
            assert info in self.keys,"info must be 'gyro','mass','spin','abundance', or 'Nuc'"
            return ftd[info]
    
    def __repr__(self):
        out=''
        for k in self.keys:out+='{:7s}'.format(k)+'\t'
        out=out[:-1]
        fstring=['{:7s}','{:<7.0f}','{:<7.0f}','{:<3.4f}','{:<4.3f}']
        for nucs in self:
            out+='\n'
            for k,(v,fs) in enumerate(zip(nucs.values(),fstring)):
                out+=fs.format(v*(1e-6 if k==3 else 1))+'\t'
        return out

NucInfo=NucInfo() #We don't really want multiple instances of this class, just one initialized one

def dipole_coupling(r,Nuc1,Nuc2):
    """ Returns the dipole coupling between two nuclei ('Nuc1','Nuc2') 
    separated by a distance 'r' (in nm). Result in Hz (gives full anisotropy,
    not b12, that is 2x larger than b12)
    """
    
    gamma1=NucInfo(Nuc1)
    gamma2=NucInfo(Nuc2)
    
    h=6.6260693e-34 #Plancks constant in J s
    mue0 = 12.56637e-7  #Permeability of vacuum [T^2m^3/J]
    
    return h*2*mue0/(4*np.pi*(r/1e9)**3)*gamma1*gamma2


#%% Flexible program for linear extrapolation
def linear_ex(x0,I0,x,dim=None,mode='last_slope'):
    """
    Takes some initial data, I0, that is a function a function of x0 in some
    dimension of I0 (by default, we search for a matching dimension- if more than
    one dimension match, then the first matching dimension will be used)
    
    Then, we extrapolate I0 between the input points such that we return a new
    I with axis x. 
    
    This is a simple linear extrapolation– just straight lines between points.
    If points in x fall outside of points in x0, we will use the two end points
    to calculate a slope and extrapolate from there.
    
    x0 must be sorted in ascending or descending order. x does not need to be sorted.
    
    If values of x fall outside of the range of x0, by default, we will take the
    slope at the ends of the given range. Alternatively, set mode to 'last_value'
    to just take the last value in x0
    """
    
    assert all(np.diff(x0)>=0) or all(np.diff(x0)<=0),"x0 is not sorted in ascending/descending order"
      
    x0=np.array(x0)
    I0=np.array(I0)
    ndim=np.ndim(x)
    x=np.atleast_1d(x)
    
    "Determine what dimension we should extrapolate over"
    if dim is None:
        i=np.argwhere(x0.size==np.array(I0.shape)).squeeze()
        assert i.size!=0,"No dimensions of I0 match the size of x0"
        dim=i if i.ndim==0 else i[0]
    

    "Swap dimensions of I0"
    I0=I0.swapaxes(0,dim)
    if np.any(np.diff(x0)<0):
#        i=np.argwhere(np.diff(x0)<0)[0,0]    
#        x0=x0[:i]
#        I0=I0[:i]    
        x0,I0=x0[::-1],I0[::-1]
    
    "Deal with x being extend beyond x0 limits"
    if x.min()<=x0[0]:
        I0=np.insert(I0,0,np.zeros(I0.shape[1:]),axis=0)
        x0=np.concatenate(([x.min()-1],x0),axis=0)
        if mode.lower()=='last_slope':
            run=x0[2]-x0[1]
            rise=I0[2]-I0[1]
            slope=rise/run 
            I0[0]=I0[1]-slope*(x0[1]-x0[0])
        else:
            I0[0]=I0[1]
    if x.max()>=x0[-1]:
        I0=np.concatenate((I0,[np.zeros(I0.shape[1:])]),axis=0)
        x0=np.concatenate((x0,[x.max()+1]),axis=0)
        if mode.lower()=='last_slope':
            run=x0[-3]-x0[-2]
            rise=I0[-3]-I0[-2]
            slope=rise/run
            I0[-1]=I0[-2]-slope*(x0[-2]-x0[-1])
        else:
            I0[-1]=I0[-2]
        
    "Index for summing"
    i=np.digitize(x,x0)
    
    I=((I0[i-1].T*(x0[i]-x)+I0[i].T*(x-x0[i-1]))/(x0[i]-x0[i-1])).T
    
    if ndim==0:
        return I[0]
    else:
        return I.swapaxes(0,dim)

#%% Class to translate among amino acid names, symbols, and abbreviations
class AA():
    names=['alanine','arginine','asparagine','aspartic acid','cysteine',
           'glutamine','glutamic acid','glycine','histidine','isoleucine',
           'leucine','lysine','methionine','phenylalanine','proline',
           'serine','threonine','tryptophan','tyrosine','valine']
    symbols=['A','R','N','D','C',
             'Q','E','G','H','I',
             'L','K','M','F','P',
             'S','T','W','Y','V']
    codes=['Ala','Arg','Asn','Asp','Cys',
           'Gln','Glu','Gly','His','Ile',
           'Leu','Lys','Met','Phe','Pro',
           'Ser','Thr','Trp','Tyr','Val']
    weights=[89,174,132,133,121,
            145,147,75,155,131,
            131,146,149,165,115,
            105,119,204,181,117]
    #Here we put uncommon names/codes/etc translated into the above codes
    alternates={'hsd':'his'} 
    
    def __init__(self,aa:str=None):
        """
        Initialize an amino acid with a string (name, symbol, code). Not
        case sensitive. Alternatively, intialize without any arguments to use
        indexing to specify amino acid later.

        Parameters
        ----------
        aa : str
            Amino acid specification (name, symbol, or code).

        Returns
        -------
        None.

        """
        if aa is None:
            self._index=None
            return
        else:
            if aa.upper() in self.symbols:
                self._index=self.symbols.index(aa.upper())
            elif aa.capitalize() in self.codes:
                self._index=self.codes.index(aa.capitalize())
            elif aa.lower() in self.names:
                self._index=self.names.index(aa.lower())
            elif aa.lower() in self.alternates:
                out=AA(self.alternates[aa.lower()])
                self._index=out._index
            else:
                assert 0,'Unknown amino acid: {}'.format(aa)
    
    @property
    def name(self):
        if self._index is not None:
            return self.names[self._index]
    @property
    def symbol(self):
        if self._index is not None:
            return self.symbols[self._index]
    @property
    def code(self):
        if self._index is not None:
            return self.codes[self._index]
    @property
    def weight(self):
        if self._index is not None:
            return self.weights[self._index]
        
    def __getitem__(self,aa):
        return AA(aa)
    
    def __call__(self,aa):
        return AA(aa)
    
    def __repr__(self):
        return self.name+'/'+self.code+'/'+self.symbol
    
    def _ipython_key_completions_(self):
        out=self.codes
        out.extend(self.names)
        return out
    
    

class tools():
    NucInfo=NucInfo
    dipole_coupling=dipole_coupling
    linear_ex=linear_ex  
    AA=AA()