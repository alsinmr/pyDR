#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 16:50:54 2022

@author: albertsmith
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:16:06 2022

@author: albertsmith
"""

import os
import numpy as np
from pyDR.Sens import Info
from pyDR import Sens
from pyDR import Data

from pyDR import MolSys,MolSelect
decode=bytes.decode

def read_file(filename):
    with open(filename,'rb') as f:
        l=decode(f.readline())[:-1]
        if l=='OBJECT:INFO':
            return read_Info(f)
        if l=='OBJECT:NUMPY':
            return read_np_object(f)
        if l=='OBJECT:SENS':
            return read_Sens(f)
        if l=='OBJECT:DETECTOR':
            return read_Detector(f)
        if l=='OBJECT:DATA':
            return read_Data(f)
        if l=='OBJECT:MOLSELECT':
            return read_MolSelect(f)

def read_Info(f):
    keys=list()
    values=list()
    for l in f:
        if str(l)[2:-3]=='END:OBJECT':break
        keys.append(str(l)[2:-3])
        pos=f.tell()
        line=f.readline()
        if str(line)[2:-3]=='OBJECT:NUMPY':
            values.append(read_np_object(f))
        else:
            f.seek(pos)
            values.append(np.load(f,allow_pickle=False))
    
    return Info(**{k:v for k,v in zip(keys,values)})

    
def read_Sens(f):
    object_class=decode(f.readline())[:-1]
    line=decode(f.readline())[:-1]
    if line!='OBJECT:INFO':print('Warning: Info object should be only data stored in sensitivity after object type')
    info=read_Info(f)
    line=decode(f.readline())[:-1]
    if line!='END:OBJECT':print('Warning: Sens object did not terminate correctly')
    return getattr(Sens,object_class)(info=info)
    


def read_Detector(f):
    line=decode(f.readline())[:-1]
    if line=='OBJECT:SENS':
        detect=Sens.Detector(read_Sens(f))  #Get input sensitivity, initialize detector
    elif line=='OBJECT:DETECTOR':
        detect=Sens.Detector(read_Detector(f))  #Get input sensitivity, initialize detector
    else:
        print('Warning:Sensitivity should be first entry in detector file')
    line=decode(f.readline())[:-1]
    if line=='Unoptimized detector':
        if decode(f.readline()[:-1])!='END:OBJECT':print('Detector file not terminated correctly')
        return detect
    opt_pars={}
    opt_pars['n']=int(line[2:])
    opt_pars['Type']=decode(f.readline())[5:-1]
    opt_pars['Normalization']=decode(f.readline())[14:-1]
    opt_pars['NegAllow']=decode(f.readline())[9:-1]=='True'
    opt_pars['options']=list()
    if decode(f.readline())[:-1]!='OPTIONS:':print('Options not correctly initialized')
    for l in f:
        if decode(l)[:-1]=='END:OPTIONS':break
        opt_pars['options'].append(decode(l)[:-1])
        
    target=np.load(f,allow_pickle=False)
    detect.r_target(target)
    detect.opt_pars=opt_pars.copy()
    detect.opt_pars['options']=list()
    if np.max(np.abs(target-detect.rhoz))>1e-6:
        print('Warning: Detector reoptimization failed')
    for o in opt_pars['options']:
        getattr(detect,o)()
    
    if decode(f.readline()[:-1])!='END:OBJECT':print('Detector file not terminated correctly')
    
    return detect



    
def read_Data(f):
    flds=['R','Rstd','S2','S2std','Rc']
    line=decode(f.readline())[:-1]
    if line!='OBJECT:DETECTOR':print('Warning: First entry of data object should be the detector')
    detect=read_Detector(f)
    
    data=Data(sens=detect.sens)
    data.detect=detect
    
    pos=f.tell()
    if decode(f.readline())[:-1]=='src_data':
        line=f.readline()
        if decode(line)[:-1]=='OBJECT:DATA':
            data.source._src_data=read_Data(f)
        else:
            data.source._src_data=decode(line)[:-1]
    else:
        f.seek(pos)
    if decode(f.readline())[:-1]!='LABEL':print('Warning: Data label is missing')
    data.label=np.load(f,allow_pickle=False)
    if decode(f.readline())[:-1]!='END:LABEL':print('Warning: Data label terminated incorrectly')
    for l in f:
        k=decode(l)[:-1]
        if k=='END:OBJECT':break
        if k in flds:
            setattr(data,k,np.load(f,allow_pickle=False))
    return data


def read_MolSelect(f):
    line=decode(f.readline())[:-1]
    if line!='TOPO':print('Warning: First entry of MolSelect object should be topo')
    topo=decode(f.readline())[:-1]
    line=decode(f.readline())[:-1]
    tr_files=list()
    t0,tf,step,dt=0,-1,1,None
    if line=='TRAJ':
        line=decode(f.readline())[:-1]
        t0,tf,step,dt=[float(line.split(':')[k+1] if k==3 else line.split(':')[k+1].split(',')[0]) for k in range(4)]
        line=decode(f.readline())[:-1]
        while line!='END:TRAJ':
            tr_files.append(line)
            line=decode(f.readline())[:-1]
    molsys=MolSys(topo,tr_files,t0=t0,tf=tf,step=step,dt=dt)
    select=MolSelect(molsys)
    line=decode(f.readline())[:-1]
    if line=='LABEL':
        select.label=np.load(f,allow_pickle=False)
        line=decode(f.readline())[:-1]
    while line!='END:OBJECT':
        fld=line.split(':')[0]
        if len(line.split(':'))==3:
            nr=int(line.split(':')[-1])
            out=np.zeros(nr,dtype=object)
            for k in range(nr):
                out[k]=molsys.uni.atoms[np.load(f,allow_pickle=False)]
            setattr(select,fld,out)
        else:
            setattr(select,fld,molsys.uni.atoms[np.load(f,allow_pickle=False)])
        line=decode(f.readline())[:-1]
    return select
    
    
def read_np_object(f):
    shape=np.load(f,allow_pickle=False)
    out=list()
    pos=f.tell()
    for k,l in enumerate(f):
        if str(l)[2:-3]=='END:OBJECT':break
        elif str(l)[2:-3]=='OBJECT:NUMPY':
            "Nested NP object"
            out.append(read_np_object(f))
        elif 'NUMPY' in str(l):
            "Numpy array"
            f.seek(pos)
            out.append(np.load(f,allow_pickle=False))
        else:
            "String"
            out.append(decode(l)[:-1])
        pos=f.tell()
    return np.array(out,dtype=object).reshape(shape)




