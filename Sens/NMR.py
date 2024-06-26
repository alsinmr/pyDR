#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 10:28:55 2021

@author: albertsmith
"""

import numpy as np
import os
from pyDR.Sens import Sens
from pyDR.Sens import NMRexper

class NMR(Sens):
    def __init__(self,tc=None,z=None,info=None,**kwargs):
        """
        Initialize an NMR sensitivity object. One may provide a list of parameters
        for the set of experiments (see NMR.new_exper for details)
        One may also define the correlation time axis by setting tc or z (z is
        the log-10 correlation time)
        
        tc or z should have 2,3 or N elements
            2: Start and end
            3: Start, end, and number of elements
            N: Provide the full axis
        """
        
        super().__init__(tc=tc,z=z)
        
        "Load the various parameters into info"
        pars=['Type','v0','v1','vr','offset','stdev','med_val','Nuc','Nuc1','dXY','CSA','eta','CSoff','QC','etaQ','theta']
        for name in dir(NMRexper):
            f=getattr(NMRexper,name)
            if hasattr(f,'__code__') and f.__code__.co_varnames[0]=='tc':
                for p0 in f.__code__.co_varnames[1:f.__code__.co_argcount]:
                    if p0 not in pars:
                        pars.append(p0)
        
        for p0 in pars:self.info.new_parameter(p0)
        
        self.new_exper(info,**kwargs)
        
    def new_exper(self,info=None,**kwargs):
        """
        Define new experiments using a list of parameters. 
        For each parameter, one should provide either
        a single entry or a list of entries corresponding to all experiments 
        (single entry will be copied to all experiments). Note that providing a
        list of entries for dXY and/or Nuc1 will be interpreted as all experiments
        having multiple dipole-coupled nuclei. A list of lists will correct this
        behavior if what is desired is a different set of couplings for each 
        experiment.
        

        Parameters
        ----------
        info : pyDR.Sens.Info
            Append experiments defined by an Info object.
        **kwargs : TYPE
            Parameters describing the individual experiments.

        Returns
        -------
        None.

        """
        if info is not None:
            self.info.append(info)
            return
        
        
        ne=0
        "First find out how many experiments are defined"
        for k,v in kwargs.items():
            if k in self.info.keys:
                if k=='dXY' or k=='Nuc1':           #These variables may have a length for just one experiment (coupling to multiple nuclei)
                    if hasattr(v,'__len__') and v.__len__() and \
                        hasattr(v[0],'__len__') and not(isinstance(v[0],str)):  
                        ne=max(ne,len(v[0]))
                elif hasattr(v,'__len__') and not(isinstance(v,str)):
                    ne=max(ne,len(v))
                elif v is not None:
                    ne=max(ne,1)

        "Edit kwargs so all entries have same length"  
        for k,v in kwargs.items():
            if k=='dXY' or k=='Nuc1':
                if hasattr(v,'__len__') and not(isinstance(v,str)): #List of values provided
                    if len(v)==ne:
                        kwargs[k]=v
                    else:
                        kwargs[k]=[v for _ in range(ne)]
                
                    # if hasattr(v[0],'__len__') and not(isinstance(v[0],str)):
                    #     assert len(v)==ne,"{0} should have one entry or match the number of experiments ({1})".format(k,ne)
                    # else:
                    #     print('checkpoint')
                    #     kwargs[k]=[v for _ in range(ne)]
                else:
                    kwargs[k]=[v for _ in range(ne)]
            else:
                if hasattr(v,'__len__') and not(isinstance(v,str)):
                    assert len(v)==ne,"{0} should have one entry or match the number of experiments ({1})".format(k,ne)
                else:
                    kwargs[k]=[v for _ in range(ne)]
        
        defaults(self.info,**kwargs)
        
        return self
    
        
    def plot_Rz(self,index=None,ax=None,norm=False,**kwargs):
        """
        Plot the sensitivity of the experiment
        """          
        hdl=super().plot_rhoz(index=index,ax=ax,norm=norm,**kwargs)
        ax=hdl[0].axes
        ax.set_ylabel(r'$R(z)$ (normalized)' if norm else r'$R(z)$/s$^{-1}$')
        return hdl
                
            
    def _rho(self):
        """
        Calculates and returns the sensitivities of all experiments stored in
        self.info
        """
        out=list()
        for m,exp in enumerate(self.info):
            assert exp['Type'] in dir(NMRexper) and getattr(NMRexper,exp['Type']).__code__.co_varnames[0]=='tc',\
                "Experiment {0} was not found in NMRexper.py".format(exp('Type'))
            f=getattr(NMRexper,exp.pop('Type'))
            for k in [k for k in exp.keys()]:
                if k not in f.__code__.co_varnames[:f.__code__.co_argcount]:
                    exp.pop(k)
            try:
                out.append(f(self.tc,**exp))
            except:
                assert 0,f"Loading experiment #{m} failed. Check parameters"
        return np.array(out)
        
        
def defaults(info,**kwargs):
    """
    Populates the info array with default values, followed by input values    
    """
    keys=info.keys        
    info_new=info.__class__()
    for k in keys:info_new.new_parameter(par=k)
    
    "End-of-file function"
    def eof(f):
        "Determines if we are at the end of the file"
        pos=f.tell()    #Current position in the file
        f.readline()    #Read out a line
        if pos==f.tell(): #If position unchanged, we're at end of file
            return True
        else:       #Otherwise, reset pointer, return False
            f.seek(pos)
            return False
    
    "Load the defaults into a dictionary"
    default_file=os.path.join(os.path.dirname(os.path.abspath(__file__)),'NMR_defaults.txt')
    defaults=dict()
    current=dict()
    with open(default_file,'r') as f:
        reading_default=False
        while not(eof(f)):
            a=f.readline().strip()
            if reading_default:   
                if a=='END DEFAULT':    #End of current default settings reached
                    reading_default=False
                else:
                    k,v=a.split(maxsplit=1) #Split into variable name and values
                    try:
                        v=np.array(v.split(),dtype=float)   #Split values if more than one
                    except:
                        v=np.array(v.split())   #If conversion to float fails, leave as string
                    if v.size==1:v=v[0] #Don't keep as an array if only one value
                    current[k]=v    #Store result
            else:
                if a=='BEGIN DEFAULT':
                    reading_default=True #Start reading a set of defaults
                    current=dict()
                    keys=f.readline().strip().split()
                    defaults.update({k:current for k in keys})
    
    "Now we populate our dataframe with the defaults"     
    if 'Nuc' in kwargs:
        Nuc=np.atleast_1d(kwargs.pop('Nuc'))
        for k,Nuc0 in enumerate(Nuc):
            if Nuc0 in defaults.keys():
                info_new.new_exper(**defaults[Nuc0])
            else:
                info_new.new_exper(Nuc=Nuc0)
    else:
        for _,value in zip([0],kwargs.values()):
            for _ in range(len(value)):info_new.new_exper()

    for k,i in enumerate(info_new):
        if i['Type']=='NOE':
            
            info_new['dXY',k]=np.atleast_1d(i['dXY'])[0]
            info_new['Nuc1',k]=np.atleast_1d(i['Nuc1'])[0]
    
    "We replace None with zeros except for nuclei"
    for k in info_new.keys:
        if k!='Nuc' and k!='Nuc1':
            v=info_new[k]
            for m in range(len(v)):
                if v[m] is None:
                    info_new[k,m]=0
                
                    
    "We override the defaults with our input values"
    for key,values in kwargs.items():
        for k,value in enumerate(np.atleast_1d(values)):
            info_new[key,k]=value
     
    "Finally, we delete extra entries in Nuc1 and dXY and delete CSA if Type is NOE"
    "Actually, I think this doesn't make sense. Let's keep extra Nuc1 and dXY for the NOE"

    for k,i in enumerate(info_new):
        if i['Type']=='NOE':
            info_new['dXY',k]=np.atleast_1d(i['dXY'])[0]
            if info_new['Nuc1'] is not None:
                info_new['Nuc1',k]=np.atleast_1d(i['Nuc1'])[0]
            # if hasattr(i['dXY'],'size') and i['dXY'].size>1:info_new['dXY',k]=info_new['dXY',k][0]
            # if hasattr(i['Nuc1'],'size') and i['Nuc1'].size>1:info_new['Nuc1',k]=info_new['Nuc1',k][0]
            info_new['CSA',k]=0
            
    info.append(info_new)
