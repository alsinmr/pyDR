#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 15:58:40 2022

@author: albertsmith
"""
import os
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from ..Defaults import Defaults
from ..Sens import Detector
# from ..IO.bin_write import write_file
from .data_plots import plot_rho,plot_fit
from ..misc.disp_tools import set_plot_attr

dtype=Defaults['dtype']

#%% Input/Output functions

from pyDR.Sens import Info
from pyDR import Sens
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
    # if hasattr(Data,'Data'):
    #     data=Data.Data(sens=detect.sens)
    # else:
    #     data=Data(sens=detect.sens)
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
        if isinstance(data.source._src_data,str):
            f.write(bytes(data.source._src_data+'\n','utf-8'))
        else:
            write_Data(f,data.source._src_data)    
    
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




#%% Data object


class Data():
    def __init__(self,R=None,Rstd=None,label=None,sens=None,select=None,src_data=None,Type=None):
        """
        Initialize a data object. Optional inputs are R, the data, R_std, the 
        standard deviation of the data, sens, the sensitivity object which
        describes the data, and src_data, which provides the source of this
        data object.
        """
        
        if R is not None:R=np.array(R,dtype=dtype)
        if Rstd is not None:Rstd=np.array(Rstd,dtype=dtype)
        
        "We start with some checks on the data sizes, etc"
        if R is not None and Rstd is not None:
            assert R.shape==Rstd.shape,"Shapes of R and R_std must match when initializing a data object"
        if R is not None and sens is not None:
            assert R.shape[1]==sens.rhoz.shape[0],"Shape of sensitivity object is not consistent with shape of R"
        
        self.R=np.array(R) if R is not None else np.zeros([0,sens.rhoz.shape[0] if sens else 0],dtype=dtype)
        self.Rstd=np.array(Rstd) if Rstd is not None else np.zeros(self.R.shape,dtype=dtype)
        if label is None:
            if select is not None and select.label is not None and len(select.label)==self.R.shape[0]:
                self.label=select.label
            else:
                self.label=np.arange(self.R.shape[0],dtype=object)
        self.sens=sens
        self.detect=Detector(sens) if sens is not None else None
        self.source=Source(src_data=src_data,select=select,Type=Type)
#        self.select=select #Stores the molecule selection for this data object
        self.vars=dict() #Storage for miscellaneous variable
        
        
    
    def __setattr__(self, name, value):
        """Special controls for setting particular attributes.
        """

        if name=='src_data':
            self.source._src_data=value
            return
        if name in ['select','project']:
            setattr(self.source,name,value)
            return
        super().__setattr__(name, value)

    @property
    def title(self):
        return self.source.title

    @property
    def select(self):
        return self.source.select
    
    @property
    def src_data(self):
        return self.source.src_data
    
    @property
    def n_data_pts(self):
        return self.R.shape[0]
    
    @property
    def ne(self):
        return self.R.shape[1]
    
    @property
    def info(self):
        return self.sens.info if self.sens is not None else None
    
    @property
    def _hash(self):
        flds=['R','Rstd','S2','S2std','sens','detect']
        out=0
        for f in flds:
            if hasattr(self,f) and getattr(self,f) is not None:
                x=getattr(self,f)
                out+=x._hash if hasattr(x,'_hash') else hash(x.data.tobytes())
        return out
    
    
    def __eq__(self,data):
        "Using string means this doesn't break when we do updates. Maybe delete when finalized"
        assert str(self.__class__)==str(data.__class__),"Object is not the same type. == not defined"
        return self._hash==data._hash
    
    def fit(self,bounds=True,parallel=False):
        return fit(self,bounds=bounds,parallel=parallel)
    
    def save(self,filename,overwrite=False,save_src=True,src_fname=None):
        if not(save_src):
            src=self.src_data
            self.src_data=src_fname  #For this to work, we need to update bin_io to save and load by filename
        write_file(filename=filename,ob=self,overwrite=overwrite)
        if not(save_src):self.src_data=src
        
    def plot(self,errorbars=False,style='plot',fig=None,index=None,rho_index=None,plot_sens=True,split=True,**kwargs):
        """
        Plots the detector responses for a given data object. Options are:
        
        errorbars:  Show the errorbars of the plot (False/True or int)
                    (default 1 standard deviation, or insert a constant to multiply the stdev.)
        style:      Plot style ('plot','scatter','bar')
        fig:        Provide the desired figure object (matplotlib.pyplot.figure)
        index:      Index to specify which residues to plot (None or logical/integer indx)
        rho_index:  Index to specify which detectors to plot (None or logical/integer index)
        plot_sens:  Plot the sensitivity as the first plot (True/False)
        split:      Break the plots where discontinuities in data.label exist (True/False)  
        """
        if rho_index is None:rho_index=np.arange(self.R.shape[1])
        if index is None:index=np.arange(self.R.shape[0])
        index,rho_index=np.array(index),rho_index
        if rho_index.dtype=='bool':rho_index=np.argwhere(rho_index)[:,0]
        assert rho_index.__len__()<=50,"Too many data points to plot (>50)"
        
        if fig is None:fig=plt.figure()
        ax=[fig.add_subplot(2*plot_sens+rho_index.size,1,k+2*plot_sens+1) for k in range(rho_index.size)]
        if plot_sens:
            ax.append(fig.add_subplot(2*plot_sens+rho_index.size,1,1))
            bbox=ax[-1].get_position()
            bbox.y0-=0.5*(bbox.y1-bbox.y0)
            ax[-1].set_position(bbox)
        # ax=[fig.add_subplot(rho_index.size+plot_sens,1,k+1) for k in range(rho_index.size+plot_sens)]
        
        if plot_sens:
            # ax=[*ax[1:],ax[0]] #Put the sensitivities last in the plotting
            hdl=self.sens.plot_rhoz(index=rho_index,ax=ax[-1])
            colors=[h.get_color() for h in hdl]
            for a in ax[1:-1]:a.sharex(ax[0])
        else:
            for a in ax[1:]:a.sharex(ax[0])
            colors=plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        not_rho0=self.sens.rhoz[0,0]/self.sens.rhoz[0].max()<.98
        for k,a,color in zip(rho_index,ax,colors):
            hdl=plot_rho(self.label[index],self.R[index,k],self.Rstd[index,k]*errorbars if errorbars else None,\
                     style=style,color=color,ax=a,split=split)
            set_plot_attr(hdl,**kwargs)
            if not(a.is_last_row()):plt.setp(a.get_xticklabels(), visible=False)
            a.set_ylabel(r'$\rho_'+'{}'.format(k+not_rho0)+r'^{(\theta,S)}$')
        # fig.tight_layout()    
        return ax
    
    def plot_fit(self,index=None,exp_index=None,fig=None):
        assert self.src_data is not None and hasattr(self,'Rc') and self.Rc is not None,"Plotting a fit requires the source data(src_data) and Rc"
        info=self.src_data.info.copy()
        lbl,Rin,Rin_std=[getattr(self.src_data,k) for k in ['label','R','Rstd']]
        Rc=self.Rc
        if 'inclS2' in self.sens.opt_pars['options']:
            info.new_exper(Type='S2')
            Rin=np.concatenate((Rin,np.atleast_2d(1-self.src_data.S2).T),axis=1)
            Rc=np.concatenate((Rc,np.atleast_2d(1-self.S2c).T),axis=1)
            Rin_std=np.concatenate((Rin_std,np.atleast_2d(self.src_data.S2std).T),axis=1)
        
        return plot_fit(lbl=lbl,Rin=Rin,Rc=Rc,Rin_std=Rin_std,\
                    info=info,index=index,exp_index=exp_index,fig=fig)
        
            
            
        
#%% Functions for fitting the data
from .Source import Source
from .fitfun import fit0
def fit(data,bounds=True,parallel=False):
    """
    Performs a detector analysis on the provided data object. Options are to
    include bounds on the detectors and whether to utilize parallelization to
    solve the detector analysis. 
    
    Note that instead of setting parallel=True, one can set it equal to an 
    integer to specify the number of cores to use for parallel processing
    (otherwise, defaults to the number of cores available on the computer)
    """
    detect=data.detect.copy()
    out=Data(sens=detect,src_data=data) #Create output data with sensitivity as input detectors
    out.label=data.label
    out.sens.lock() #Lock the detectors in sens since these shouldn't be edited after fitting
    out.select=data.select
    
    
    "Prep data for fitting"
    X=list()
    for k,(R,Rstd) in enumerate(zip(data.R.copy(),data.Rstd)):
        r0=detect[k] #Get the detector object for this bond
        UB=r0.rhoz.max(1)#Upper and lower bounds for fitting
        LB=r0.rhoz.min(1)
        R-=data.sens[k].R0 #Offsets if applying an effective sensitivity
        if 'inclS2' in r0.opt_pars['options']: #Append S2 if used for detectors
            R=np.concatenate((R,[data.S2[k]]))
            Rstd=np.concatenate((Rstd,[data.S2std[k]]))
        R/=Rstd     #Normalize data by its standard deviation
        r=(r0.r.T/Rstd).T   #Also normalize R matrix
        
        X.append((r,R,(LB,UB) if bounds else None,Rstd))
    
    "Perform fitting"
    if parallel:
        nc=parallel if isinstance(parallel,int) else mp.cpu_count()
        with mp.Pool(processes=nc) as pool:
            Y=pool.map(fit0,X)
    else:
        Y=[fit0(x) for x in X]
    
    "Extract data into output"
    out.R=np.zeros([len(Y),detect.r.shape[1]],dtype=dtype)
    out.R_std=np.zeros(out.R.shape,dtype=dtype)
    out.Rc=np.zeros([out.R.shape[0],detect.r.shape[0]],dtype=dtype)
    for k,y in enumerate(Y):
        out.R[k],out.R_std[k],Rc0=y
        out.R[k]+=detect[k].R0
        out.Rc[k]=Rc0*X[k][3]
        
    if 'inclS2' in detect.opt_pars['options']:
        out.S2c,out.Rc=out.Rc[:,-1],out.Rc[:,:-1]
    if 'R2ex' in detect.opt_pars['options']:
        out.R2,out.R=out.R[:,-1],out.R[:,:-1]
        out.R2std,out.Rstd=out.Rstd[:,-1],out.Rstd[:,:-1]
    
    if data.source.project is not None:data.source.project.append_data(out)
    return out
    
    

 

    


