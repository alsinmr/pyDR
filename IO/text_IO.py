#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 17:48:07 2022

@author: albertsmith
"""

import os
from pyDR import clsDict,Defaults
import numpy as np
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
    with open(filename,'r') as f:
        for line in f:
            line=line.strip()
            if line=='INFO':
                out=read_INFO(f)
                while not(eof(f)):
                    if f.readline().strip()=='DATA':
                        info=out
                        out=read_Data(f)
                        out.sens=clsDict['NMR'](info)
                        out.source.filename=os.path.abspath(filename)
                        out.source.status='raw'
                        out.source.Type='NMR'
            elif line=='DATA':
                out=read_Data(f)
                out.sens=clsDict['NMR']()
                out.source.filename=os.path.abspath(filename)
                out.source.status='raw'
                out.source.Type='NMR'
            elif eof(f):
                print('Unrecognized text file type')
    return out
        
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
    line=f.readline().strip()
    for line in f:
        if line.strip()=='END':break
        key,*values=line.strip().split('\t')
        out=list()
        for v in values:
            if ',' in v:
                out.append(list())
                for v0 in v.split(','):
                    out[-1].append(assign_type(v0))
            else:
                out.append(assign_type(v))
        pars[key]=out
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
    data=clsDict['Data']()
    keys=['R','Rstd','S2','S2std','label']
    key=None
    isstr=False
    values=list()
    for line in f:
        if line.strip() in keys:
            if key is not None:
                setattr(data,key,np.array(values,dtype=None if isstr else dtype))
            key=line.strip()
            values=list()
        elif len(line.strip())!=0:
            if '\t' in line.strip():
                values.append(list())
                for v in line.strip().split('\t'):
                    # values[-1].append(assign_type(v))
                    values[-1].append(v)
            else:
                values.append(assign_type(line.strip()))
                isstr=isinstance(values[-1],str)
    return data
                
            
    
    
                    
                        

#%% Misc functions

def assign_type(x):
    "Takes a string and assigns it to float, integer, or simply string"
    
    if np.char.isnumeric(x.split('.')[0]):
        a,b=[int(x0) if np.char.isnumeric(x0) else None for x0 in x.split('.')]
        if b is None or b==0:
            return a
        else:
            return a+b/(10**int(np.log10(b)+1))
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
    