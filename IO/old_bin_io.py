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
try:    #Not happy about this....sometimes loads as module, sometimes as Data class (!!!???)
    from pyDR.Data.Data import Data
except:
    from pyDR.Data import Data    
from pyDR import MolSys,MolSelect
decode=bytes.decode

print('Why are we here?')

def write_file(filename,ob,overwrite=False):
    if os.path.exists(filename) and not(overwrite):
        print('Warning: File {} already exists. Set overwrite=True or choose a different name'.format(filename))
        return
    with open(filename,'wb') as f:
        object_class=str(ob.__class__).split('.')[-1][:-2]
        object_parent=str(ob.__class__.__base__).split('.')[-1][:-2]
        if object_class=='Info':
            write_Info(f,ob)
        if object_class=='ndarray':
            write_np_object(f,ob)
        if object_class=='MolSelect':
            write_MolSelect(f,ob)
        if object_class=='Detector':
            write_Detector(f,ob)
        elif object_parent=='Sens':
            write_Sens(f,ob)
        elif object_class=="Data":
            write_Data(f,ob)

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


def write_Info(f,info):
    f.write(b'OBJECT:INFO\n')
    for k in info.keys:
        f.write(bytes(k+'\n','utf-8'))
        value=info[k]
        if hasattr(value,'dtype') and value.dtype=='O':
            try:
                value=value.astype(float)
                if np.all(value==value.astype(int)):value=value.astype(int)
                np.save(f,value,allow_pickle=False)
            except:
                write_np_object(f,value)
        elif isinstance(value,str):
            f.write(bytes(value+'\n','utf-8'))
        else:
            np.save(f,value,allow_pickle=False)
    f.write(b'END:OBJECT\n')
    
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
    
def write_Sens(f,sens):
    f.write(b'OBJECT:SENS\n')
    object_class=str(sens.__class__).split('.')[-1][:-2]
    f.write(bytes(object_class+'\n','utf-8'))
    write_Info(f,sens.info)
    f.write(b'END:OBJECT\n')
    
def read_Sens(f):
    object_class=decode(f.readline())[:-1]
    line=decode(f.readline())[:-1]
    if line!='OBJECT:INFO':print('Warning: Info object should be only data stored in sensitivity after object type')
    info=read_Info(f)
    line=decode(f.readline())[:-1]
    if line!='END:OBJECT':print('Warning: Sens object did not terminate correctly')
    return getattr(Sens,object_class)(info=info)
    

def write_Detector(f,detect,src_fname=None):
    f.write(b'OBJECT:DETECTOR\n')
    if str(detect.sens.__class__).split('.')[-1][:-2]=='Detector':
        write_Detector(f,detect.sens)
    else:
        write_Sens(f,detect.sens)
    
    if detect.opt_pars.__len__()==5:
        op=detect.opt_pars
        for k in ['n','Type','Normalization','NegAllow']:
            f.write(bytes('{0}:{1}\n'.format(k,op[k]),'utf-8'))
        f.write(b'OPTIONS:\n')
        for o in op['options']:
            f.write(bytes('{}\n'.format(o),'utf-8'))
        f.write(b'END:OPTIONS\n')
        target=detect.rhoz
        if 'inclS2' in op['options']:
            target=target[1:]
        if 'R2ex' in op['options']:
            target=target[:-1]
        np.save(f,target,allow_pickle=False)
    elif detect.opt_pars.__len__()==0:
        f.write(b'Unoptimized detector\n')
    else:
        assert 0,'opt_pars of detector object has the wrong number of entries'
            
    f.write(b'END:OBJECT\n')

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


def write_Data(f,data):
    flds=['R','Rstd','S2','S2std','Rc']
    f.write(b'OBJECT:DATA\n')
    if data.detect is None:data.detect=Sens.Detector(data.sens)
    write_Detector(f,data.detect)
    if data.src_data is not None:
        f.write(b'src_data\n')
        if isinstance(data.src_data,str):
            f.write(bytes(data.src_data+'\n','utf-8'))
        else:
            write_Data(f,data.src_data)    
    
    f.write(b'LABEL\n')
    np.save(f,data.label,allow_pickle=False)
    f.write(b'END:LABEL\n')
    
    for k in flds:
        if hasattr(data,k) and getattr(data,k) is not None:
            f.write(bytes('{0}\n'.format(k),'utf-8'))
            np.save(f,getattr(data,k),allow_pickle=False)
    f.write(b'END:OBJECT\n')
    
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


def write_MolSelect(f,select):
    f.write(b'OBJECT:MOLSELECT\n')
    molsys=select.molsys
    traj=molsys.traj
    #Start with the molsys information
    f.write(b'TOPO\n')
    f.write(bytes('{0}\n'.format(molsys.topo),'utf-8'))
    #Next the trajectory if included
    if traj is not None:
        f.write(b'TRAJ\n')
        f.write(bytes('t0:{0},tf:{1},step:{2},dt:{3}\n'.format(traj.t0,traj.tf,traj.step,traj.dt),'utf-8'))
        for file in traj.files:
            f.write(bytes(file+'\n','utf-8'))
        f.write(b'END:TRAJ\n')
    if select.label is not None:
        f.write(b'LABEL\n')
        np.save(f,select.label,allow_pickle=False)
    for fld in ['sel1','sel2','repr_sel']:
        v=getattr(select,fld)
        if v is not None:
            if isinstance(v,np.ndarray):
                f.write(bytes('{0}:list:{1}\n'.format(fld,len(v)),'utf-8'))
                for v0 in v:
                    np.save(f,v0.indices,allow_pickle=False)
            else:
                f.write(bytes('{0}\n'.format(fld),'utf-8'))
                np.save(f,v.indices,allow_pickle=False)
    f.write(b'END:OBJECT\n')

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
    
    
def write_Source(f,select):
    f.write(b'OBJECT:SOURCE\n')
    flds=['Type','filename','saved_filename','_title','_status']
        
            
    

def write_np_object(f,ob):
    """
    Read and write of numpy objects have three options for each element:
        1) Another numpy object
        2) A string
        3) A numpy array
    
    Numpy objects are written with this function (possibly recursively)
    Strings are encoded/decoded with bytes and bytes.decode
    Numpy arrays are encoded with the np.save option (allow_pickle=False)
    """
    f.write(b'OBJECT:NUMPY\n')
    np.save(f,np.array(ob.shape),allow_pickle=False)
    ob1=ob.reshape(np.prod(ob.shape))
    for o in ob1:
        if hasattr(o,'dtype') and o.dtype=='O':
            write_np_object(f,o)
        elif isinstance(o,str):
            f.write(bytes(o+'\n','utf-8'))
        else:
            np.save(f,o,allow_pickle=False)
    f.write(b'END:OBJECT\n')
            
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



