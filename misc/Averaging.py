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

def avg2sel(data,sel,chk_resid:bool=True,chk_segid:bool=True):
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
        data._projDelta(initialize=True)
        for d in data:avg2sel(d,sel,chk_resid=chk_resid,chk_segid=chk_segid)
        return data._projDelta()
    
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
                i0=np.logical_and(s01.name==sel0.sel1.names,s02.name==sel0.sel2.names)
                i1=np.logical_and(s01.name==sel0.sel2.names,s02.name==sel0.sel1.names)
                
                if chk_segid:
                    i0=np.all([i0,s01.segid==sel0.sel1.segids,s02.segid==sel0.sel2.segids],axis=0)
                    i1=np.all([i1,s01.segid==sel0.sel2.segids,s02.segid==sel0.sel1.segids],axis=0)
                    
                if chk_resid:
                    i0=np.all([i0,s01.resid==sel0.sel1.resids,s02.resid==sel0.sel2.resids],axis=0)
                    i1=np.all([i1,s01.resid==sel0.sel2.resids,s02.resid==sel0.sel1.resids],axis=0)
                
                i=np.logical_or(i0,i1)
                # i=np.logical_or(\
                #   np.all([s01.segid==sel0.sel1.segids,s01.resid==sel0.sel1.resids,s01.name==sel0.sel1.names,
                #           s02.segid==sel0.sel2.segids,s02.resid==sel0.sel2.resids,s02.name==sel0.sel2.names],axis=0),
                #   np.all([s01.segid==sel0.sel2.segids,s01.resid==sel0.sel2.resids,s01.name==sel0.sel2.names,
                #           s02.segid==sel0.sel1.segids,s02.resid==sel0.sel1.resids,s02.name==sel0.sel1.names],axis=0))
            if np.any(i):
                count+=1
                if sel1 is None:
                    # sel1=(s01+s01)[:1] #Trick to ensure we get atom groups
                    # sel2=(s02+s02)[:1]
                    sel1=(sel0.sel1[i]+sel0.sel1[i])[:1]
                    sel2=(sel0.sel2[i]+sel0.sel2[i])[:1]
                else:
                    # sel1+=s01
                    # sel2+=s02
                    sel1+=sel0.sel1[i]
                    sel2+=sel0.sel2[i]
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
        

def avgData(data,index:list,wt:list=None):
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
        List of indices to define which data points to average together. 
        Option 1) Should be the same length as the original data object, 
        where the returned data will have length equal to the number of unique 
        values in index.
        Option 2) Should be a list of lists, where the length of the outer list
        is the length of the resulting data object and each inner list specifies
        which elements to average together.
                                             
        
    wt : list
        List of weightings for elements in index. For option 1), weight should
        have the same length as the number of unique elements in index. For 
        example, if index=[0,0,0,0,1,1,2], uniform weighting is given by
        wt=[[0.25,0.25,0.25,0.25],[0.5,0.5],[1]]
        Otherwise, weighting should be a list of lists the same length as 
        Note that uniform weighting is implemented by default
        Default is None
    
    Returns
    -------
    pyDR.Data
        Averaged data object.

    """
    
    if not(np.any([hasattr(i,'__len__') for i in index])) and len(index)==len(data):
        i0=list()
        for k,i in np.unique(index):
            i0.append(np.argwhere(i==index)[:,0])
        index=i0
            

    if str(data.__class__)==str(clsDict['Project']):
        return [avgData(d,index=index,wt=wt) for d in data]
    
    
    index=[i if hasattr(i,'__len__') else [i] for i in index]  #Make sure all elements of index are lists
    assert np.all(np.isin(np.concatenate(index),range(len(data)))),"Values in index must all be less than len(data)"
    
    #Set up weight if not provided (verify its validity)
    if wt is None:
        wt=[np.ones(len(i))/len(i) for i in index]
    else:
        assert len(wt)==len(index),"index and wt need to have the same length"
        wt=[wt0 if hasattr(wt0,'__len__') else [wt0] for wt0 in wt]  #Ensure list of lists
        wt=[wt0/sum(wt0) for wt0 in wt] #Ensure normalization
    
    #Create the output dta
    out=data.__class__(sens=data.sens) #Create output data with sensitivity as input detectors
    out.detect=data.detect
    out.source=copy(data.source)
    out.src_data=None   #We get some values from data for source, but it is not "fit"  data in the usual sense
    out.select=clsDict['MolSelect'](data.select.molsys) if data.select is not None else None
    if out.source.additional_info is None:
        out.source.additional_info='Avg' 
    else:
        out.source.additional_info='Avg_'+out.source.additional_info

    n=len(index)
    
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




    sel1=list()
    sel2=list()
    for k,(i,wt0) in enumerate(zip(index,wt)):  #Sweep over the values in index, weight
        for f in flds[:-1]:  #Sweep over fields
            if hasattr(data,f) and getattr(data,f) is not None:
                getattr(out,f)[k]=(getattr(data,f)[i].T*wt0).sum(-1)
        
        #Append to the selection
        if data.select is not None:
            if len(i):
                sel1.append(sum(data.select.sel1[i]))
                sel2.append(sum(data.select.sel2[i]))
            else:
                sel1.append(data.select.molsys.uni.atoms[:0]) #Empty atom groups
                sel2.append(data.select.molsys.uni.atoms[:0])
            

        if len(i):
            label=data.label[i[0]]
            for q in i[1:]:
                match=SequenceMatcher(a=label,b=data.label[q]).find_longest_match(0,len(label),0,len(data.label[q]))
                label=label[match.a:match.a+match.size]
                
            out.label[k]=label if len(label) else data.label[i[0]]  
        else:
            out.label[k]=None
            
            
    if data.select is not None:
        out.select.sel1=sel1
        out.select.sel2=sel2
    
    if np.all([l[0]=='_' for l in out.label]):
        out.label=np.array([l[1:] for l in out.label],dtype=out.label.dtype)            
        
    out.details=data.details.copy()
    out.details.append('Data was averaged using an index')
    string='index=('
    for i in index:
        if hasattr(i,'__len__'):
            string+='('+','.join([str(i0) for i0 in i])+'),'
        else:
            string+=str(i)
    string=string[:-1]+')'
    out.details.append(string)
    
    if data.source.project is not None:data.source.project.append_data(out)
    
    return out

def avgDataObjs(*args,wt:list=None,incl_src:bool=False):
    """
    Currently a very simple averaging of data objects with identical sizes.
    Labels and selections will come from the first data object. R, S2 will come
    from averaging. Rstd, S2std will be averaged and scaled according to 
    propagation of error rules.
    
    Later, we may implement averaging of data objects with non-matching selections,
    where only matched elements of the selection are returned in the final object.

    Parameters
    ----------
    *args : Positional arguments (data)
        All data objects to be averaged (can also be provided as a single list/tuple).
    wt : list-like, optional
        Weighting for each data object (length equal to number of data objects).
        Will be automatically normalized to sum to 1
    incl_src : bool, optional
        If set to True, the source data will also be averaged and included in the
        project. Otherwise source data will be set to None

    Returns
    -------
    pyDR.Data
        Averaged data object.

    """
    if len(args)==1 and hasattr(args[0],'__len__') and hasattr(args[0][0],'Rstd'): #args[0] is a list of data objects?
        data=args[0]
    else:
        data=args
    N=len(data)

    # Weighting defaults
    if wt is None:wt=np.ones(N,dtype=float)/N
    wt=np.array(wt,dtype=float)
    wt/=wt.sum()
    
    for d in data[1:]:
        assert len(d)==len(data[0]),"All data must have the same length"
    
    out=copy(data[0])
    
    #Clear some calculations made in the iRED data object
    if hasattr(out,'_CCnorm'):out._CCnorm=None
    if hasattr(out,'_totalCCnorm'):out._totalCCnorm=None
    
    #Sweep over all fields that require averaging
    flds=['R','S2','CC','totalCC','Rstd','S2std']
    for f in flds:
        if hasattr(data[0],f):
            if getattr(data[0],f) is None:
                setattr(out,f,None)
            else:
                if 'std' in f:
                    v=np.sqrt(np.sum([(getattr(d,f)*w)**2 for d,w in zip(data,wt)],axis=0))
                    setattr(out,f,v)
                else:
                    setattr(out,f,np.sum([getattr(d,f)*w for d,w in zip(data,wt)],axis=0))
    
    #Update the processing details               
    details=list()
    details.append(f'Average of {N} data objects')
    details.append('Warning: Selection, labels, and information in source correspond to the first data object')
    
    for k,d in enumerate(data):
        details.append(f'START DATA OBJECT {k}')
        for det in d.details:details.append(det)
        details.append(f'END DATA OBJECT {k}')
        
    out.details=details
    out.source.additional_info='AvOb' if out.source.additional_info is None else \
        'AvOb_'+out.source.additional_info
        
    #Also average the source data
    if incl_src and not(np.any([d.src_data is None for d in data])):
        out.src_data=avgDataObjs([d.src_data for d in data],wt=wt,incl_src=incl_src)
    else:
        out.src_data=None
        
    #Append results to project
    if data[0].project is not None:data[0].project.append_data(out)
    
    return out
        
def appendDataObjs(*args,check_sens:bool=True):
    """
    Appends several data objects together. Note that this means the sensitivities
    of the detectors must be the same (can be overridden).
    
    Note this method will not append cross-correlation information (from iRED)
    
    Parameters
    ----------
    *args : TYPE
        All data objects to be appended (can also be provided as a single list/tuple).
    check_sens : bool
        Set to False to ignore sensitivities that do not match. Note that in 
        any case, the number of detectors must be the same.

    Returns
    -------
    pyDR.Data
        Appended data object.

    """
    if len(args)==1 and hasattr(args[0],'__len__') and hasattr(args[0][0],'Rstd'): #args[0] is a list of data objects?
        data=args[0]
    else:
        data=args
    
    N=len(data)  
    
    #Assert sensitivities the same
    for k,d in enumerate(data[1:]):
        if check_sens:
            assert d.sens==data[0].sens,f"Data object {k} has a different sensitivity than data object 0"
        else:
            assert d.R.shape[1]==data[0].R.shape[1],\
            "All data must have the same number of detectors"
            
    out=copy(data[0])
    
    #Sweep over all fields that require averaging
    flds=['R','S2','Rstd','S2std','label']
    for f in flds:
        if hasattr(data[0],f):
            if getattr(data[0],f) is None:
                setattr(out,f,None)
            else:
                setattr(out,f,np.concatenate([getattr(d,f) for d in data],axis=0))
                
    #Append selections
    if not(np.any([d.select is None for d in data])):
        lengths=[d.select.uni.atoms.__len__() for d in data]
        #We use the # of atoms in the universe to determine if the indexing is compatible
        #Is this really ok? Can we think of a probable coincident universe size but with different indices?
        if len(np.unique(lengths))>1:
            print('Warning: Data is being appended where selections come from different topologies')
            print('Selections will not be saved in the appended data')
            out.select.sel1=None
            out.select.sel2=None
            out.select.repr_sel=None
        else:
            mdmode=[d.select._mdmode for d in data]
            for d in data:d.select._mdmode=False
            atoms=None
            for f in ['sel1','sel2','repr_sel']:
                if not(np.any([getattr(d.select,f) is None for d in data])):
                    if atoms is None:atoms=data[0].select.uni.atoms
                    l=[len(d.select) for d in data]
                    sel=np.zeros(sum(l),dtype=object)
                    for k,d in enumerate(data):
                        sel[sum(l[:k]):sum(l[:k+1])]=[atoms[s.indices] for s in getattr(d.select,f)]
                    setattr(out.select,f,sel)
                else:
                    setattr(out.select,f,None)
            for d,mm in zip(data,mdmode):d.select._mdmode=mm
            out.select._mdmode=mdmode[0]
        
    #Update the processing details               
    details=list()
    details.append(f'Appending {N} data objects')
    details.append('Warning: Information in source corresponds to the first data object')
    
    for k,d in enumerate(data):
        details.append(f'START DATA OBJECT {k}')
        for det in d.details:details.append(det)
        details.append(f'END DATA OBJECT {k}')
        
    out.details=details
    out.source.additional_info='ApOb' if out.source.additional_info is None else \
        'ApOb_'+out.source.additional_info
        
    #Also average the source data
    if not(np.any([d.src_data is None or d.src_data.__class__==str for d in data])):
        out.src_data=appendDataObjs([d.src_data for d in data])
        
    #Append results to project
    if data[0].project is not None:data[0].project.append_data(out)
    
    return out
    
    
    

# def avgData2(data:Data,index:list,wt:list=None)->Data:
#     """
#     Averages data points in a data object together according to an index.
    
#     One may instead provide a project, in which case the operation is performed 
#     on all data in that project (returns None, but appends results to project)

#     Parameters
#     ----------
#     data : pyDR.Data
#         Data object to be averaged. Must have its selection specified. We assume
#         that this selection consists of single bonds (usually we're averaging 
#         an MD trajectory, which does not have ambiguity in its assignment)
    
#     index : list/np.ndarray
#         List of indices to define which data points to average together. 
#         Option 1) Should be the same length as the original data object, 
#         where the returned data will have length equal to the number of unique 
#         values in index.
#         Option 2) Should be a list of lists, where the length of the outer list
#         is the length of the resulting data object and each inner list specifies
#         which elements to average together.
                                             
        
#     wt : list
#         List of weightings for each unique element in index. For example, if
#         index=[0,0,0,0,1,1,2], uniform weighting is given by
#         wt=[[0.25,0.25,0.25,0.25],[0.5,0.5],[1]]
#         Note that uniform weighting is implemented by default
#         Default is None
    
#     Returns
#     -------
#     pyDR.Data
#         Averaged data object.

#     """
    
#     if str(data.__class__)==str(clsDict['Project']):
#         return [avgData2(d,index=index,wt=wt) for d in data]
        
    
#     out=data.__class__(sens=data.sens) #Create output data with sensitivity as input detectors
#     out.detect=data.detect
#     out.source=copy(data.source)
#     out.src_data=None   #We get some values from data for source, but it is not "fit"  data in the usual sense
#     out.select=clsDict['MolSelect'](data.select.molsys) if data.select is not None else None
#     if out.source.additional_info is None:
#         out.source.additional_info='Avg' 
#     else:
#         out.source.additional_info='Avg_'+out.source.additional_info

#     n=np.unique(index).size
    
#     index=np.array(index)
    
#     #Setup or check the weighting
#     if wt is None:
#         wt=list()
#         for i in np.unique(index):
#             ni=(index==i).sum()
#             wt.append(np.repeat([1/ni],ni))
#     else:
#         for k,(i,wt0) in enumerate(zip(np.unique(index),wt)):
#             ni=(index==i).sum()
#             assert len(wt0)==0,'The number of weights provided for index {0} did not match the number of occurences of index {0}'.format(k)
#             wt[k]=np.array(wt0)/np.sum(wt0) #Normalize weighting
            
            
            
#     #Pre-allocate the data storage
#     flds=['R','Rstd','S2','S2std','label']
#     for f in flds:
#         if hasattr(data,f) and getattr(data,f) is not None:
#             dtype=getattr(data,f).dtype
#             if getattr(data,f).ndim==2:
#                 nd=getattr(data,f).shape[1]
#                 setattr(out,f,np.zeros([n,nd],dtype=dtype))
#             else:
#                 setattr(out,f,np.zeros(n,dtype=dtype))
    
    


        
#     sel1=list() #Storage for the new selection
#     sel2=list()
#     for k,(i,wt0) in enumerate(zip(np.unique(index),wt)):
#         for f in flds[:-1]:
#             if hasattr(data,f) and getattr(data,f) is not None:
#                 getattr(out,f)[k]=(getattr(data,f)[index==i].T*wt0).sum(-1)
        
#         ids=np.argwhere(i==index)[:,0]
#         label=data.label[ids[0]]
#         for q in ids[1:]:
#             match=SequenceMatcher(a=label,b=data.label[q]).find_longest_match(0,len(label),0,len(data.label[q]))
#             label=label[match.a:match.a+match.size]
            
#         out.label[k]=label if len(label) else data.label[ids[0]]    
        
#         if data.select is not None:
#             s1,s2=data.select.uni.atoms[:0],data.select.uni.atoms[:0] #Empty atom groups
#             for q in ids:
#                 s1+=data.select.sel1[q]
#                 s2+=data.select.sel2[q]
#             sel1.append(s1)
#             sel2.append(s2)
    
#     if data.select is not None:
#         out.select.sel1=sel1
#         out.select.sel2=sel2
    
    
#     if np.all([l[0]=='_' for l in out.label]):
#         out.label=np.array([l[1:] for l in out.label],dtype=out.label.dtype)
    
#     out.details=data.details.copy()
#     out.details.append('Data was averaged using an index')
#     out.details.append('index=('+', '.join(str(i) for i in index)+')')
    
#     if data.source.project is not None:data.source.project.append_data(out)
    
#     return out
    
        
def avgMethyl(data):
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
    
    # index=np.repeat(np.arange(data.R.shape[0]//3),3)
    
    index=[[m for m in range(k,k+3)] for k in range(0,data.R.shape[0],3)]
    
    out=avgData(data,index)
    out.details.pop(-1)
    out.details[-1]='Data averaging was applied over every 3 data points (methyl averaging)'
    return out
    

    
        
                
        
        