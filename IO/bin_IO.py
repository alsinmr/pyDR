#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 11:09:41 2022

@author: albertsmith
"""

import os
import typing

import numpy as np
import pyDR.Sens

from ..Defaults import Defaults
from pyDR import clsDict
from pyDR.Sens.MD import MDsens_from_pars

# from ..IO.bin_write import write_file

dtype=Defaults['dtype']

#%% Input/Output functions

from pyDR import Sens
decode=bytes.decode


def isbinary(filename):
    textchars = bytearray({7,8,9,10,12,13,27} | set(range(0x20, 0x100)) - {0x7f})
    is_binary_string = lambda bytes: bool(bytes.translate(None, textchars))
    with open(filename, 'rb') as f:
        return is_binary_string(f.read(1024))

#%% Main Input/Output
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
        elif object_class=="Data" or object_class=="Data_PCA":
            write_Data(f,ob)
        elif object_class=='Data_iRED':
            write_Data_iRED(f,ob)
        elif object_class=='Source':
            write_Source(f,ob)
        elif object_class=='EntropyCC':
            write_EntropyCC(f,ob)
            
def read_file(filename:str,directory:str='')->object:
    with open(filename,'rb') as f:
        l=decode(f.readline()).strip()
        if l=='OBJECT:INFO':
            return read_Info(f)
        if l=='OBJECT:NUMPY':
            return read_np_object(f)
        if l=='OBJECT:SENS':
            return read_Sens(f)
        if l=='OBJECT:DETECTOR':
            return read_Detector(f)
        if l=='OBJECT:DATA':
            out=read_Data(f,directory=directory)
            out.source.saved_filename=os.path.abspath(filename)
            return out
        if l=='OBJECT:DATA_IRED':
            out=read_Data_iRED(f,directory=directory)
            out.source.saved_filename=os.path.abspath(filename)
            return out
        if l=='OBJECT:MOLSELECT':
            return read_MolSelect(f,directory=directory)
        if l=='OBJECT:ENTROPYCC':
            return read_EntropyCC(f)


#%% Info Input/Output
def write_Info(f: typing.BinaryIO, info: dict):
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
    
def read_Info(f: typing.TextIO):
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
            if np.any(np.isnan(values[-1])):
                i=np.isnan(values[-1])
                values[-1]=values[-1].astype(object)
                values[-1][i]=None
    
    out=clsDict['Info'](**{k:v for k,v in zip(keys,values)})
    return out

#%% Sens and detector Input/Output    
def write_Sens(f: typing.BinaryIO, sens:pyDR.Sens.Sens):
    f.write(b'OBJECT:SENS\n')
    object_class=str(sens.__class__).split('.')[-1][:-2]
    f.write(bytes(object_class+'\n','utf-8'))
    if hasattr(sens,'sampling_info'):
        f.write(bytes('SAMPLINGINFO\ntf:{tf},dt:{dt},n:{n},nr:{nr}\n'.format(**sens.sampling_info),'utf-8'))
    else:
        write_Info(f,sens.info)
    f.write(b'END:OBJECT\n')


def read_Sens(f: typing.TextIO):
    object_class=decode(f.readline())[:-1]
    line=decode(f.readline()).strip()
    if 'SAMPLINGINFO' in line:
        line=decode(f.readline()).strip()
        tf=int(line.split('tf:')[1].split(',')[0])
        dt=float(line.split('dt:')[1].split(',')[0])
        n=int(line.split('n:')[1].split(',')[0])
        nr=int(line.split('nr:')[1])
        line=decode(f.readline())[:-1]
        return MDsens_from_pars(tf=tf,dt=dt,n=n,nr=nr)
    else:
        if line!='OBJECT:INFO':print('Warning: Info object should be only data stored in sensitivity after object type')
        info=read_Info(f)
    line=decode(f.readline())[:-1]
    if line!='END:OBJECT':print('Warning: Sens object did not terminate correctly')
    return clsDict[object_class](info=info)


def write_Detector(f: typing.BinaryIO, detect: pyDR.Sens.Detector, src_fname=None):
    f.write(b'OBJECT:DETECTOR\n')
    if str(detect.sens.__class__).split('.')[-1][:-2]=='Detector':
        write_Detector(f,detect.sens)
    else:
        write_Sens(f,detect.sens)
    
    if 'n' in detect.opt_pars:
        op=detect.opt_pars
        for k in ['n','Type','Normalization','NegAllow']:
            f.write(bytes('{0}:{1}\n'.format(k,op[k]),'utf-8'))
        f.write(b'OPTIONS:\n')
        for o in op['options']:
            f.write(bytes('{}\n'.format(o),'utf-8'))
        f.write(b'END:OPTIONS\n')
        target=detect.rhoz
#        if 'inclS2' in op['options']:
#            target=target[1:]
#        if 'R2ex' in op['options']:
#            target=target[:-1]
        f.write(b'TARGET\n')
        np.save(f,target,allow_pickle=False)
        f.write(b'NORM\n')
        np.save(f,detect.norm,allow_pickle=False)
    elif detect.opt_pars.__len__()==0:
        f.write(b'Unoptimized detector\n')
    else:
        assert 0,'opt_pars of detector object has the wrong number of entries'
            
    f.write(b'END:OBJECT\n')


def read_Detector(f):
    line=decode(f.readline())[:-1]
    if line=='OBJECT:SENS':
        detect=clsDict['Detector'](read_Sens(f))  #Get input sensitivity, initialize detector
    elif line=='OBJECT:DETECTOR':
        detect=clsDict['Detector'](read_Detector(f))  #Get input sensitivity, initialize detector
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
    
    if decode(f.readline()).strip()!='TARGET':print('Target not correctly saved')    
    target=np.load(f,allow_pickle=False)
    detect._Sens__rho=target
    #TODO The below line is a little risky- can be fixed with r.reload(), but how to check if this gets run?
    detect._Sens__rhoCSA=np.zeros(target.shape)
    if decode(f.readline()).strip()!='NORM':print('Norm not correctly saved')
    detect._Sens__norm=np.load(f,allow_pickle=False)
    
    dz=detect.z[1]-detect.z[0]
    info=detect.info
    for k in info.keys.copy():info.del_parameter(k)
    info.new_parameter(z0=np.array([(detect.z*rz).sum()/rz.sum() for rz in detect.rhoz]))
    info.new_parameter(zmax=np.array([detect.z[np.argmax(rz)] for rz in detect.rhoz]))
    info.new_parameter(Del_z=np.array([rz.sum()*dz/rz.max() for rz in detect.rhoz]))
    
    if detect.norm.size==detect.rhoz.shape[0]:
        info.new_parameter(stdev=1/detect.norm)


    
#    detect.r_target(target)
    detect.opt_pars=opt_pars.copy()
#    detect.opt_pars['options']=list()
    
    if decode(f.readline()[:-1])!='END:OBJECT':print('Detector file not terminated correctly')
    
    return detect



#%% Data input and output   
def write_Data(f,data):
    flds=['R','Rstd','S2','S2std']
    f.write(b'OBJECT:DATA\n')
    if data.detect is None:data.detect=Sens.Detector(data.sens)
    write_Detector(f,data.detect)
    write_Source(f,data.source)    
    
    f.write(b'LABEL\n')
    np.save(f,data.label,allow_pickle=False)
    f.write(b'END:LABEL\n')
    
    for k in flds:
        if hasattr(data,k) and getattr(data,k) is not None:
            f.write(bytes('{0}\n'.format(k),'utf-8'))
            np.save(f,getattr(data,k),allow_pickle=False)
    f.write(b'END:OBJECT\n')

def read_Data(f,directory:str=''):
    flds=['R','Rstd','S2','S2std','Rc']
    line=decode(f.readline())[:-1]
    if line!='OBJECT:DETECTOR':print('Warning: First entry of data object should be the detector')
    detect=read_Detector(f)
    assert decode(f.readline())[:-1]=='OBJECT:SOURCE','Source entry not initiated correctly'
    source=read_Source(f,directory=directory)

    if decode(f.readline())[:-1]!='LABEL':print('Warning: Data label is missing')
    kwargs={}
    kwargs['label']=np.load(f,allow_pickle=False)
    if decode(f.readline())[:-1]!='END:LABEL':print('Warning: Data label terminated incorrectly')
    for l in f:
        k=decode(l)[:-1]
        if k=='END:OBJECT':break
        if k in flds:
            kwargs[k]=np.load(f,allow_pickle=False)
    
    kwargs['sens']=detect.sens
    data=clsDict['Data'](**kwargs)
    data.source=source
    data.detect=detect
    return data

def write_Data_iRED(f,data):
    flds=['R','Rstd','S2','S2std','CC','totalCC']
    f.write(b'OBJECT:DATA_IRED\n')
    if data.detect is None:data.detect=Sens.Detector(data.sens)
    write_Detector(f,data.detect)
    write_Source(f,data.source)    
    
    f.write(b'LABEL\n')
    np.save(f,data.label,allow_pickle=False)
    f.write(b'END:LABEL\n')
    
    f.write(b'IREDdict\n')
    if len(data.iRED.keys()):
        for k in ['rank','M','m','Lambda']:
            f.write(bytes('{0}\n'.format(k),'utf-8'))
            np.save(f,data.iRED[k],allow_pickle=False)
    f.write(b'END:IREDdict\n')
    
    for k in flds:
        if hasattr(data,k) and getattr(data,k) is not None:
            f.write(bytes('{0}\n'.format(k),'utf-8'))
            np.save(f,getattr(data,k),allow_pickle=False)
    f.write(b'END:OBJECT\n')
    
def read_Data_iRED(f,directory:str=''):
    flds=['R','Rstd','S2','S2std','Rc','CC','totalCC']
    line=decode(f.readline())[:-1]
    if line!='OBJECT:DETECTOR':print('Warning: First entry of data object should be the detector')
    detect=read_Detector(f)
    assert decode(f.readline())[:-1]=='OBJECT:SOURCE','Source entry not initiated correctly'
    source=read_Source(f,directory=directory)

    if decode(f.readline())[:-1]!='LABEL':print('Warning: Data label is missing')
    kwargs={}
    kwargs['label']=np.load(f,allow_pickle=False)
    if decode(f.readline())[:-1]!='END:LABEL':print('Warning: Data label terminated incorrectly')
    
    if decode(f.readline()).strip()!='IREDdict':print('Warning: IREDdict missing')
    line=decode(f.readline()).strip()
    iRED={}
    while line!='END:IREDdict':
        iRED[line]=np.load(f,allow_pickle=False)
        line=decode(f.readline()).strip()
    
    for l in f:
        k=decode(l)[:-1]
        if k=='END:OBJECT':break
        if k in flds:
            kwargs[k]=np.load(f,allow_pickle=False)
    
    kwargs['sens']=detect.sens
    data=clsDict['Data_iRED'](**kwargs)
    data.source=source
    data.detect=detect
    data.iRED=iRED
    return data

def write_Source(f,source):
    f.write(b'OBJECT:SOURCE\n')
    flds=['Type','filename','saved_filename','_title','status','n_det','additional_info']
    for fld in flds:
        f.write(bytes('{0}:{1}\n'.format(fld,getattr(source,fld)),'utf-8'))
    f.write(b'Details\n')
    for l in source.details:
        f.write(bytes(l+'\n','utf-8'))
    if source._src_data is not None:    #Check _src_data, not src_data, because src_data will re-load the data object!!
        f.write(b'src_data\n')
        if isinstance(source._src_data,str):
            f.write(bytes(source._src_data+'\n','utf-8'))
        else:
            write_Data(f,source._src_data)
    if source.select is not None:
        write_MolSelect(f,source.select)
    f.write(b'END:OBJECT\n')
    
def read_Source(f,directory:str=''):
    source=clsDict['Source']()
    flds=['Type','filename','saved_filename','_title','status','n_det','additional_info']
    
    for fld in flds:
        line=decode(f.readline()).strip()
        assert line.split(':')[0]==fld,"Error: expected '{0}' but found '{1}'.".format(fld,line.split(':')[0])
        v=line.split(':',maxsplit=1)[1]
        if v=='None':continue
        if fld=='filename' and v[0]=='[' and v[-1]==']':
            v=v[2:-2].split(',')
        setattr(source,fld,int(v) if fld=='n_det' else v)
    line=decode(f.readline()).strip()
    if line=='Details':
        details=list()
        line=decode(f.readline()).strip()
        while line not in ['src_data','OBJECT:MOLSELECT','END:OBJECT']:
            details.append(line)
            line=decode(f.readline()).strip()
        source.details=details
    if line=='src_data':
        line=decode(f.readline())[:-1]
        if line=='OBJECT:DATA':
            source._src_data=read_Data(f)
        else:
            source._src_data=line
        line=decode(f.readline())[:-1]
    if line=='OBJECT:MOLSELECT':
        source.select=read_MolSelect(f,directory=directory)
        line=decode(f.readline())[:-1]
    if line!='END:OBJECT':print('Warning: Source object not terminated correctly')
    return source
    
        

#%% Selection object Input/Output
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
    
def read_MolSelect(f,directory=''):
    line=decode(f.readline())[:-1]
    if line!='TOPO':print('Warning: First entry of MolSelect object should be topo')
    topo=decode(f.readline())[:-1]  #First try provided full path
    topo=find_file(topo,directory=directory,f=f)   #Find the topology file
    
    line=decode(f.readline())[:-1]
    tr_files=list()
    t0,tf,step,dt=0,-1,1,None
    if line=='TRAJ':   #Collect the trajectory files
        line=decode(f.readline())[:-1]
        t0,tf,step,dt=[float(line.split(':')[k+1] if k==3 else line.split(':')[k+1].split(',')[0]) for k in range(4)]
        dt/=step
        line=decode(f.readline())[:-1]
        while line!='END:TRAJ':
            tr_files.append(find_file(line,directory=directory,f=f))
            line=decode(f.readline())[:-1]
        if None in tr_files:
            tr_files=None
            tf=1
    molsys=clsDict['MolSys'](topo,tr_files,t0=t0,tf=tf,step=step,dt=dt)
    select=clsDict['MolSelect'](molsys)
    line=decode(f.readline())[:-1]
    if line=='LABEL':  #Load the label
        select.label=np.load(f,allow_pickle=False)
        line=decode(f.readline())[:-1]
    while line!='END:OBJECT':
        fld=line.split(':')[0]
        if len(line.split(':'))==3:
        
            nr=int(line.split(':')[-1])
            out=np.zeros(nr,dtype=object)
            for k in range(nr):
                if topo is not None:  #Next line throws error if topo not found
                    out[k]=molsys.uni.atoms[np.load(f,allow_pickle=False)]
                else:
                    _=np.load(f,allow_pickle=False)
            if topo is not None:
                setattr(select,fld,out)
        else:
            if topo is not None:
                setattr(select,fld,molsys.uni.atoms[np.load(f,allow_pickle=False)])
            else:
                _=np.load(f,allow_pickle=False) #We still need to get past the binary numpy array
        line=decode(f.readline())[:-1]
    return select

#%% EntropyCC
def write_EntropyCC(f,ECC):
    f.write(b'OBJECT:ENTROPYCC\n')
    write_MolSelect(f, ECC.select)
    flds=['Sres','Scc']
    
    for fld in flds:
        f.write(bytes(f'\n{fld}\n','utf-8'))
        np.save(f,getattr(ECC,fld),allow_pickle=True)
    f.write(b'END:OBJECT\n')
    
def read_EntropyCC(f):
    EntropyCC=clsDict['EntropyCC']
    line=decode(f.readline())[:-1]
    assert line=='OBJECT:MOLSELECT','First object in EntropyCC should be the selection object'
    ECC=EntropyCC(read_MolSelect(f))
    
    flds=['Sres','Scc']
    for l in f:
        k=decode(l)[:-1]
        if k=='END:OBJECT':break
        if k in flds:
            setattr(ECC,f'_{k}',np.load(f,allow_pickle=False))
    return ECC

#%% Numpy object Input/Ouput    
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
    out.append(None)
    out=np.array(out,dtype=object)[:-1].reshape(shape)
    return np.array(out,dtype=object).reshape(shape)

def find_file(filename,directory:str='',f=None):
    if os.path.exists(filename):    #Full path correctly given (could be in current folder)
        return os.path.abspath(filename)
    if f is not None and len(directory) and '.pdb' in filename:  #See if topo is in pdbs
        pdb_dir=os.path.join(directory,'pdbs')  #Directory for storing pdbs
        data_file=os.path.split(f.name)[1]
        if os.path.exists(os.path.join(pdb_dir,'pdb_list.txt')):
            with open(os.path.join(pdb_dir,'pdb_list.txt'),'r') as f:
                for line in f:
                    if line.strip().split(':')[0]==data_file:
                        return os.path.join(pdb_dir,line.strip().split(':')[1]+'.pdb')
        
    if os.path.exists(os.path.split(filename)[1]): #File in current folder (but wrong full path given)
        return os.path.abspath(os.path.split(filename)[1])
    if os.path.exists(os.path.join(directory,os.path.split(filename)[1])): #File in the project directory
        return os.path.abspath(os.path.join(directory,os.path.split(filename)[1]))
    print('{0} could not be found'.format(os.path.split(filename)[1]))
    return



# def write_PCA(f: typing.BinaryIO, pca: pyDR.PCA):
#     f.write(b'OBJECT:PCA\n')
#     write_MolSelect(pca.select)
    
#     pca_flds=[]
    
#     for k in pca_flds:
#         f.write(bytes('{0}\n'.format(k),'utf-8'))
#         np.save(f,getattr(pca,k),allow_pickle=False)
    
            
#     f.write(b'END:OBJECT\n')




    









    

        
            
    

