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
decode=bytes.decode

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
        np.save(f,target)
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

flds=['R','Rstd','S2','S2std','Rc']
def write_Data(f,data):
    f.write(b'OBJECT:DATA\n')
    if data.detect is None:data.detect=Sens.Detector(data.sens)
    write_Detector(f,data.detect)
    if data.src_data is not None:
        #TODO
        """
        We need an option here to test if src_data is a string. If it is, then
        we need to simply write the string into the save file instead of writing the
        whole data object. We also need to update read_Data to read in the string
        """
        f.write(b'src_data\n')
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
    line=decode(f.readline())[:-1]
    if line!='OBJECT:DETECTOR':print('Warning: First entry of data object should be the detector')
    detect=read_Detector(f)
    
    data=Data(sens=detect.sens)
    data.detect=detect
    
    pos=f.tell()
    if decode(f.readline())[:-1]=='src_data':
        f.readline()
        data.src_data=read_Data(f)
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
            
def write_np_object(f,ob):
    """
    Read and write of numpy objects have three options for each element:
        1) Another numpy object
        2) A string
        3) A numpy array
    
    Numpy objects are written with this function (possibly recursively)
    Strings are encoded/decoded with bytes and bytes.decode
    Numpy arrays are encoded with the np.save option (allow_pick=False)
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




