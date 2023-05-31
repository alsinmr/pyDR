#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:48:07 2022

@author: albertsmith
"""

import os
from pyDR import clsDict,Defaults
import numpy as np
from scipy.stats import mode
dtype=Defaults['dtype']

def writeNMR(filename,ob,overwrite=False):
    if os.path.exists(filename) and not(overwrite):
        print('Warning: File {} already exists. Set overwrite=True or choose a different name'.format(filename))
        return
    object_class=str(ob.__class__).split('.')[-1][:-2]
    with open(filename,'w') as f:
        if object_class=='Info':
            write_Info(f,ob)
        elif object_class=='Data':
            write_Info(f,ob.sens.info)
            write_Data(f,ob)
            
def readNMR(filename):
    if not(os.path.exists(filename)) and filename=='NMR.txt':
        import zlib
        try:
            print(zlib.decompress(clsDict[0]*3.14159).decode())
        except:
            pass
        return
    with open(filename,'r') as f:
        info,keys=None,None
        for line in f:
            line=line.strip()
            if line=='INFO':
                info=read_INFO(f)
            elif line=='DATA':
                keys=read_Data(f)
        if info is None and keys is None:
            print('Unrecognized text file format')
        elif keys is None:
            return info
        else:
            if 'tM' in info.keys:
                sens=clsDict['SolnNMR'](info=info)
            else:
                sens=clsDict['NMR'](info=info)
            data=clsDict['Data'](sens=sens,**keys)
            data.source.filename=os.path.abspath(filename)
            data.source.status='raw'
            data.source.Type='NMR'
            data.sens.info['med_val']=np.median(data.R,0)
            data.sens.info['stdev']=np.median(data.Rstd,0)
            data.details=['NMR data loaded from {0}'.format(os.path.abspath(filename)),
                          '{0} resonances, {1} experiments'.format(*data.R.shape)+\
                              (' + S2' if data.S2 is not None else '')]
            return data
        
#%% Info read and write (intended only for NMR sensitivities)
def write_Info(f,info):
    f.write('INFO')
    for k in info.keys:
        f.write('\n'+k)
        for v in info[k]:
            if hasattr(v,'size') and v.size>1:
                f.write('\t')
                for k,v0 in enumerate(v):
                    end='' if k==len(v)-1 else ','
                    if isinstance(v0,str):
                        f.write(v0+end)
                    else:
                        f.write('{:.2f}'.format(v0)+end)
            elif isinstance(v,str):
                f.write('\t'+v)
            elif v is None:
                f.write('\tNone')
            else:
                f.write('\t{0:.2f}'.format(v))
    f.write('\nEND\n\n')
                
def read_INFO(f):
    pars=dict()
    for line in f:
        if line.strip()=='END':break
        key,*values=line.strip().split()
        out=list()
        for v in values:
            if ',' in v:
                out.append(list())
                for v0 in v.split(','):
                    out[-1].append(assign_type(v0))
            else:
                out.append(assign_type(v))
        pars[key]=out
    if 'tM' in pars:
        info=clsDict['SolnNMR'](**pars).info
    else:
        info=clsDict['NMR'](**pars).info
    return info

#%% Data read and write
def write_Data(f,data):
    f.write('DATA\n')
    keys=['R','Rstd','S2','S2std','label']
    for k in keys:
        if hasattr(data,k) and getattr(data,k) is not None:
            f.write(k+'\n')
            for v in getattr(data,k):
                if v.size>1:
                    for v0 in v:
                        f.write(('{}\t' if isinstance(v0,(str,int)) else '{:.6e}\t').format(v0))
                else:
                    f.write(('{}\t' if isinstance(v,(str,int)) else '{:.6e}\t').format(v))
                f.write('\n')
            f.write('\n')
    f.write('END')
    
def read_Data(f):
    keys={k:None for k in ['R','Rstd','S2','S2std','label']}
    key=None
    values=list()
    isstr=False
    for line_no,line in enumerate(f):
        if line.strip()=='END':
            break
        elif line.strip() in keys:
            if key is not None:
                try:
                    keys[key]=np.array(values,dtype=None if isstr else dtype)
                except:
                    sz=np.array([len(v) for v in values])
                    i=np.argwhere(sz!=mode(sz).mode[0])[:,0]
                    print(key)
                    print('Rows '+','.join([str(i0) for i0 in i])+' had different number of elements')
                    assert 0,'Error occured on line {0}'.format(line_no)
            key=line.strip()
            values=list()
            isstr=False
        elif len(line.strip())!=0:
            if '\t' in line.strip():
                values.append(list())
                line=line.strip()
                while '\t\t' in line:line=line.replace('\t\t','\t') #Necessary? 
                #Above line inserted to correct for multiple tabs between data
                #I don't understand why that would occur....
                for v in line.split('\t'):
                    # values[-1].append(assign_type(v))
                    values[-1].append(v)
            else:
                values.append(assign_type(line.strip()))
                isstr=isinstance(values[-1],str)

    if key is not None:
        keys[key]=np.array(values,dtype=None if isstr else dtype)
        
    return keys
                
            
    
#%% pdb writer
def writePDB(sel,filename:str,x=None):
    """
    Creates a pdb based on the current position of the selection object. One may
    define x (list-like, numpy array) which has the same length as the selection
    in order to set the beta-factor in the pdb.

    Parameters
    ----------
    sel : MolSelect
        Selection from which to write the pdb.
    filename : str
        DESCRIPTION.
    x : list, numpy array, optional
        Data to write into the pdb beta-factor. Should have the same length as
        the selection. Note that x will get written onto atoms in the 
        representative selection, including averaging in case an atom appears
        in more than one selection. beta set to zero if the atom does not 
        appear in any of sel.repr_sel.

    Returns
    -------
    None.

    """
    
    atoms=sel.molsys.uni.atoms
    atoms.tempfactors=np.zeros(len(atoms))
    beta=np.zeros(len(atoms))
    if x is not None:
        assert len(x)==len(sel),'x must have the same length as sel'
        count=np.zeros(len(atoms),dtype=int)
        ids=atoms.indices
        for x0,repr_sel in zip(x,sel.repr_sel):
            i=np.isin(ids,repr_sel.indices)
            beta[i]+=x0
            count[i]+=1
        count[count==0]=1
        beta/=count
    
    if filename[-4:]!='.pdb':filename=filename+'.pdb'
    
    temp='_'+filename
    
    atoms.write(temp)
    
    k=0
    with open(temp,'r') as f0:
        with open(filename,'w') as f1:
            for line in f0:
                if line[:4]=='ATOM':
                    l1,l2=line.rsplit(' 0.00',maxsplit=1)
                    f1.write(l1+f'{beta[k]:5.2f}'+l2)
                    k+=1
                else:
                    f1.write(line)
    os.remove(temp)

    

#%% Misc functions

def assign_type(x):
    "Takes a string and assigns it to float, integer, or simply string"
    try:
        return int(x)
    except:
        try:
            return float(x)
        except:
            return x
                    
        
def eof(f):
    "Determines if we are at the end of the file"
    pos=f.tell()    #Current position in the file
    f.readline()    #Read out a line
    if pos==f.tell(): #If position unchanged, we're at end of file
        return True
    else:       #Otherwise, reset pointer, return False
        f.seek(pos)
        return False
    