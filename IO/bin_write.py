#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 12:16:06 2022

@author: albertsmith
"""

import os
import numpy as np
#from pyDR.Sens import Info
from pyDR import Sens
#from pyDR import Data
#from pyDR import MolSys,MolSelect
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
        if object_class=='MolSelect':
            write_MolSelect(f,ob)
        if object_class=='Detector':
            write_Detector(f,ob)
        elif object_parent=='Sens':
            write_Sens(f,ob)
        elif object_class=="Data":
            write_Data(f,ob)

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
    

def write_Sens(f,sens):
    f.write(b'OBJECT:SENS\n')
    object_class=str(sens.__class__).split('.')[-1][:-2]
    f.write(bytes(object_class+'\n','utf-8'))
    write_Info(f,sens.info)
    f.write(b'END:OBJECT\n')


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




