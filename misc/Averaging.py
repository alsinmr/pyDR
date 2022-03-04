#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:37:35 2022

@author: albertsmith
"""

from pyDR import clsDict
import numpy as np
from copy import copy

def avg2sel(data,sel):
    """
    Averages data points in a data object together such that we may define a new
    data object whose selection matches a given input selection.

    Parameters
    ----------
    data : pyDR.Data
        Data object to be averaged. Must have its selection specified. We assume
        that this selection consists of single bonds (usually we're averaging 
        an MD trajectory, which does not have ambiguity in its assignment)
    
    sel : pyDR.MolSelect
        Selection specifying how to average the initial data object. Note that
        each selection in sel defines one or more bonds. If one of the bonds
        is not existing in the data object's selection, then that bond will be
        skipped entirely. Note that each group selection in sel is expected to
        have equal number of atoms in both sel1 and sel2, corresponding to
        each bond. If this is not the case, then the heavy atom will be ignored
        and we will just try to match the light atom (proton) in data.
    
    Returns
    -------
    pyDR.Data
    Averaged data object.

    """
    
    out=clsDict['Data'](sens=data.sens) #Create output data with sensitivity as input detectors
    out.source=copy(data.source)
    out.src_data=None   #We get some values from data for source, but it is not "fit"  data in the usual sense
    out.select=clsDict['MolSelect'](data.select.molsys)
    if out.source.additional_info is None:
        out.source.additional_info='Avg' 
    else:
        out.source.additional_info='Avg_'+out.source.additional_info 

    sel1=list()
    sel2=list()
    
    flds=['R','Rstd','S2','S2std','label']
    for f in flds:
        if hasattr(data,f) and getattr(data,f) is not None:
            setattr(out,f,np.zeros([len(sel),*getattr(data,f).shape[1:]],dtype=getattr(data,f).dtype))
    
    
    sel0=data.select
    sel10=list()
    sel20=list()
    skipped=list()
    
    for k,(s1,s2) in enumerate(zip(sel.sel1,sel.sel2)):
        if not hasattr(s1,'__len__'):s1=(s1+s1)[:1]  #Trick to make sure we can iterate over these
        if not hasattr(s2,'__len__'):s2=(s2+s2)[:1]
        sel1=None
        sel2=None
        count=0
        for s01,s02 in zip(s1,s2):
            if sel0.molsys==sel.molsys:
                i=np.logical_or(np.logical_and(s01.id==sel0.sel1.ids,s02.id==sel0.sel2.ids),
                    np.logical_and(s01.id==sel0.sel2.ids,s02.id==sel0.sel1.ids))
            else:                        
                i=np.logical_or(\
                  np.all([s01.segid==sel.sel1.segids,s01.resid==sel.sel1.resids,s01.name==sel.sel1.names,
                          s02.segid==sel.sel2.segids,s02.resid==sel.sel2.resids,s02.name==sel.sel2.names],axis=0),
                  np.all([s01.segid==sel.sel2.segids,s01.resid==sel.sel2.resids,s01.name==sel.sel2.names,
                          s02.segid==sel.sel1.segids,s02.resid==sel.sel1.resids,s02.name==sel.sel1.names],axis=0))
            if np.any(i):
                count+=1
                if sel1 is None:
                    sel1=s01
                    sel2=s02
                else:
                    sel1+=s01
                    sel2+=s02
                for f in flds:
                    if hasattr(data,f) and getattr(data,f) is not None:
                        getattr(out,f)[k]+=getattr(data,f)[i][0]
        if count:
            sel10.append(sel1)
            sel20.append(sel2)
            for f in flds[:-1]:
                if hasattr(data,f) and getattr(data,f) is not None:
                    getattr(out,f)[k]/=count
        else:
            skipped.append(k)
    while len(skipped):
        i=skipped.pop(-1)
        for f in flds:
            if hasattr(data,f) and getattr(data,f) is not None:
                setattr(out,f,np.delete(getattr(out,f),i,axis=0))
    out.select.sel1=sel10
    out.select.sel2=sel20
    
    if data.source.project is not None:data.source.project.append_data(out)
    
    return out
        
            
        
                
        
        