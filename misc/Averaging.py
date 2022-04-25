#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 14:37:35 2022

@author: albertsmith
"""

from pyDR import clsDict
import numpy as np
from copy import copy
from difflib import SequenceMatcher

Data=clsDict['Data']
MolSelect=clsDict['MolSelect']

def avg2sel(data:Data,sel:MolSelect) -> Data:
    """
    Averages data points in a data object together such that we may define a new
    data object whose selection matches a given input selection. 
    
    One may instead provide a project, in which case the operation is performed 
    on all data in that project (returns None, but appends results to project)

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
    
    if str(data.__class__)==str(clsDict['Project']):
        for d in data:avg2sel(d,sel)
        return
    
    _mdmode=data.select._mdmode
    data.select._mdmode=True
    
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
                  np.all([s01.segid==sel0.sel1.segids,s01.resid==sel0.sel1.resids,s01.name==sel0.sel1.names,
                          s02.segid==sel0.sel2.segids,s02.resid==sel0.sel2.resids,s02.name==sel0.sel2.names],axis=0),
                  np.all([s01.segid==sel0.sel2.segids,s01.resid==sel0.sel2.resids,s01.name==sel0.sel2.names,
                          s02.segid==sel0.sel1.segids,s02.resid==sel0.sel1.resids,s02.name==sel0.sel1.names],axis=0))
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
    
    
    out.details=data.details.copy()
    out.details.append('Data was averaged to match a new selection')
    out.details.append('Original data had length {0} and new data has length {1}'.format(len(data),len(out)))
    
    if data.source.project is not None:data.source.project.append_data(out)
    
    data.select._mdmode=_mdmode
    
    return out
        
            
def avgData(data:Data,index:list,wt:list=None)->Data:
    """
    Averages data points in a data object together according to an index.
    
    One may instead provide a project, in which case the operation is performed 
    on all data in that project (returns None, but appends results to project)

    Parameters
    ----------
    data : pyDR.Data
        Data object to be averaged. Must have its selection specified. We assume
        that this selection consists of single bonds (usually we're averaging 
        an MD trajectory, which does not have ambiguity in its assignment)
    
    index : list/np.ndarray
        List of indices to define which data points to average together. Should
        be the same length as the original data object, where the returned data
        will have length equal to the number of uniqe values in index.
        
    wt : list
        List of weightings for each unique element in index. For example, if
        index=[0,0,0,0,1,1,2], uniform weighting is given by
        wt=[[0.25,0.25,0.25,0.25],[0.5,0.5],[1]]
        Note that uniform weighting is implemented by default
        Default is None
    
    Returns
    -------
    pyDR.Data
        Averaged data object.

    """
    
    if str(data.__class__)==str(clsDict['Project']):
        for d in data:avgData(d,index=index,wt=wt)
        return
    
    out=clsDict['Data'](sens=data.sens) #Create output data with sensitivity as input detectors
    out.detect=data.detect
    out.source=copy(data.source)
    out.src_data=None   #We get some values from data for source, but it is not "fit"  data in the usual sense
    out.select=clsDict['MolSelect'](data.select.molsys) if data.select is not None else None
    if out.source.additional_info is None:
        out.source.additional_info='Avg' 
    else:
        out.source.additional_info='Avg_'+out.source.additional_info

    n=np.unique(index).size
    
    index=np.array(index)
    
    #Setup or check the weighting
    if wt is None:
        wt=list()
        for i in np.unique(index):
            ni=(index==i).sum()
            wt.append(np.repeat([1/ni],ni))
    else:
        for k,(i,wt0) in enumerate(zip(np.unique(index),wt)):
            ni=(index==i).sum()
            assert len(wt0)==0,'The number of weights provided for index {0} did not match the number of occurences of index {0}'.format(k)
            wt[k]=np.array(wt0)/np.sum(wt0) #Normalize weighting
            
            
            
    #Pre-allocate the data storage
    flds=['R','Rstd','S2','S2std','label']
    for f in flds:
        if hasattr(data,f) and getattr(data,f) is not None:
            dtype=getattr(data,f).dtype
            if getattr(data,f).ndim==2:
                nd=getattr(data,f).shape[1]
                setattr(out,f,np.zeros([n,nd],dtype=dtype))
            else:
                setattr(out,f,np.zeros(n,dtype=dtype))
    
    sel1=list() #Storage for the new selection
    sel2=list()
    for k,(i,wt0) in enumerate(zip(np.unique(index),wt)):
        for f in flds[:-1]:
            if hasattr(data,f) and getattr(data,f) is not None:
                getattr(out,f)[k]=(getattr(data,f)[index==i].T*wt0).sum(-1)
        
        ids=np.argwhere(i==index)[:,0]
        label=data.label[ids[0]]
        for q in ids[1:]:
            match=SequenceMatcher(a=label,b=data.label[q]).find_longest_match(0,len(label),0,len(data.label[q]))
            label=label[match.a:match.a+match.size]
            
        out.label[k]=label if len(label) else data.label[ids[0]]    
        
        if data.select is not None:
            s1,s2=data.select.uni.atoms[:0],data.select.uni.atoms[:0] #Empty atom groups
            for q in ids:
                s1+=data.select.sel1[q]
                s2+=data.select.sel2[q]
            sel1.append(s1)
            sel2.append(s2)
    
    if data.select is not None:
        out.select.sel1=sel1
        out.select.sel2=sel2
    
    
    out.details=data.details.copy()
    out.details.append('Data was averaged using an index')
    out.details.append('index=('+', '.join(str(i) for i in index)+')')
    
    if data.source.project is not None:data.source.project.append_data(out)
    
    return out
    
        
def avgMethyl(data:Data) -> None:
    """
    Average together every three data points in a data object.    
    
    One may instead provide a project, in which case the operation is performed 
    on all data in that project (returns None, but appends results to project)

    Parameters
    ----------
    data : TYPE
        Data to be averaged where each group of three data points are in principle
        equivalent, and therefore can be averaged together. An example would be
        H–C bonds in a methyl group.
        
    Returns
    -------
    pyDR.Data
        Averaged data object.

    """
    if str(data.__class__)==str(clsDict['Project']):    #Perform function on full project
        for d in data:avgMethyl(d)
        return
    
    index=np.repeat(np.arange(data.R.shape[0]//3),3)
    out=avgData(data,index)
    out.details.pop(-1)
    out.details[-1]='Data averaging was applied over every 3 data points (methyl averaging)'
    return out
    

    
        
                
        
        