#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2021 Albert Smith-Penzel

This file is part of pyDIFRATE

pyDIFRATE is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

pyDIFRATE is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with pyDIFRATE.  If not, see <https://www.gnu.org/licenses/>.


Questions, contact me at:
albert.smith-penzel@medizin.uni-leipzig.de



Created on Tue Oct  6 10:46:10 2020

@author: albertsmith
"""


import numpy as np
from copy import copy,deepcopy
from pyDR.MDtools import vft
from pyDR.MDtools.Ctcalc import sparse_index,get_count,Ctcalc
from pyDR.misc import ProgressBar
from .vec_funs import new_fun,print_frame_info
from . import FramesPostProc as FPP
from pyDR.iRED.iRED import iRED
from pyDR import Defaults,clsDict


#%%Functions for returning just the correlation function (to ct or to ired)
def md2data(select,rank=2):
    return FrameObj(select).md2data(rank=rank)

def md2iRED(select,rank=2):
    return FrameObj(select).md2iRED(rank=rank)


flags={'ct_finF':True,'ct_m0_finF':False,'ct_0m_finF':False,'ct_0m_PASinF':False,\
        'A_m0_finF':False,'A_0m_finF':False,'A_0m_PASinF':False,\
        'ct_prod':True,'ct':True,'S2':True}

class ReturnIndex():
    _flags={'ct_finF':True,'ct_m0_finF':False,'ct_0m_finF':False,'ct_0m_PASinF':False,\
            'A_m0_finF':False,'A_0m_finF':False,'A_0m_PASinF':False,\
            'ct_prod':True,'ct':True,'S2':True}
    for k in flags.keys():
        locals()[k]=property(lambda self,k=k:self.flags[k]) #Default flag settings
    
    def __init__(self,ret_in=None,**kwargs):
        self.flags=copy(self._flags)
        

        if ret_in is not None:
            if hasattr(ret_in,'return_index') and len(ret_in.return_index)==10:
                self=ret_in
            else:
                assert isinstance(ret_in,list) and len(ret_in)==10,'ret_in must be a list of 10 elements'
                for k,v in zip(flags.keys(),ret_in):
                    flags[k]=bool(v)
                
        for k,v in kwargs.items():
            if k in flags.keys():
                flags[k]=v
        
        self.flags=flags.copy() #This makes the class and instance values independent
    
    
    
    def __getitem__(self,k):
        if isinstance(k,int):
            return [v for v in self.flags.values()][k]
        else:
            return self.flags[k]
    
    def __repr__(self):
        out=''
        for k,v in self.flags.items():
            out+=k+': {0}'.format(v)+'\n'
        return out
    def __str__(self):
        return self.__repr__()
    
    def copy(self):
        return ReturnIndex(**self.flags)
        
    @property            
    def return_index(self):
        """
        Returns an array of logicals determining which terms to calculate
        """
        return np.array([v for v in self.flags.values()],dtype=bool)
    
    @property
    def calc_ct_m0_finF(self):
        "Determines if we should calculate ct_m0_finF"
        if self.ct_finF or self.ct_m0_finF or self.ct_0m_PASinF or self.ct_prod:return True
        return False
    @property
    def calc_A_m0_finF(self):
        "Determines if we should calculate A_m0_finF"
        if self.A_m0_finF or self.A_0m_finF:return True
        return False
    @property
    def calc_A_0m_PASinF(self):
        "Determines if we should calculate A_m0_PASinF"
        if self.ct_finF or self.ct_prod or self.A_0m_PASinF:return True
        return False
    @property
    def calc_ct_finF(self):
        "Determines if we should calculate ct_finF"
        if self.ct_finF or self.ct_prod:return True
        return False
    @property
    def calc_any_ct(self):
        "Determines if any correlation functions should be calculated"
        if self.ct_finF or self.ct_m0_finF or self.ct_0m_finF or self.ct_0m_PASinF or \
            self.ct_prod or self.ct:return True
        return False
    
    def set2sym(self):
        "De-activates the storage of terms that cannot by calculated in symmetric mode"  
        self.set2auto()
        # self.flags['ct_m0_finF']=False
    
    def set2auto(self):
        "De-activates the storage of terms that cannot by calculated in auto mode"
        if self.ct_0m_finF or self.ct_0m_PASinF or self.A_m0_finF or self.A_m0_finF:
            print('Warning: Individual components of the correlation functions or tensors will not be returned in auto or sym mode')
        self.flags=copy(self._flags)
        self.flags.update({'ct_0m_finF':False,'ct_0m_PASinF':False,\
                           'A_m0_finF':False,'A_0m_finF':False})
    def set2direct(self):
        "De-activates all calculations except the direct calculation of ct"
        for k in self.flags:
            self.flags[k]=False
        self.flags['ct']=True
        
        

class FrameObj():
    def __init__(self,select=None,molecule=None):
        self.select=copy(select) if select is not None else copy(molecule) #I want to phase out using the name molecule
        self.select._mdmode=True
        self.vft=None
        self.vf=list()
        self.rank=2 #Rank of calculation
        self.frame_info={'frame_index':list(),'label':None,'info':list()}
        self.defaults={'t0':0,'tf':-1,'dt':None,'n':-1,'nr':10,'mode':'auto',\
                       'squeeze':True}
        self.terms={'ct_finF':True,'ct_m0_finF':False,'ct_0mPASinF':False,\
                    'A_m0_finF':True,'A_0m_finF':True,'A_0m_PASinF':True,\
                    'ct_prod':True,'ct':True,'S2':True}
        self.__frames_loaded=False #Flag to check if frames currently loaded
        self.include=None #Record of which frames were included in calculation
        self.mode=self.defaults['mode']
        self.sampling_info={'tf':None,'dt':None,'n':None,'nr':None,'t0':None} #Record the sampling info
        
        self.return_index=ReturnIndex(**self.terms)   
        self.t=None
        self.Ct={}
        self.A={}
        self.S2=None
        self.__return_index=None
        self._project=None
        self.reduced_mem=Defaults['reduced_mem']
    
    
    def __setattr__(self,name,value):
        if name=='project':
            if str(value.__class__).split('.')[-1][:-2]=='Project':
                self._project=value
            return
        if name=='rank':
            if hasattr(self,'rank') and value!=self.rank: #Check if the rank has changed
                self.include=None #This will force the re-calculation of correlation functions
                if hasattr(self,'ct_out'):delattr(self,'ct_out')
            super().__setattr__(name,value)
            return
        super().__setattr__(name,value)
    
    
    @property
    def molecule(self):
        return self.select
    
    @property
    def project(self):
        """
        Returns the associated project if one exists

        Returns
        -------
        Project
            pyDR Project object

        """
        return self.select.project if self._project is None else self._project
    
    @property
    def description_of_terms(self):
        out="""n=number of frames, nr=number of residues, nt=number of time points
        ct_finF:
              n x nr x nt array of the real correlation functions for each
              motion (after scaling by residual tensor of previous motion)
              ct_m0_finF:
              5 x n x nr x nt array, with the individual components of each
              motion (f in F)
        ct_0m_finF:
              5 x n x nr x nt array, with the individual components of each
              motion (f in F)
        ct_0m_PASinF:
              5 x n x nr x nt array, with the individual components of each
              motion (PAS in F)
        A_m0_finF:
              Value at infinite time of ct_m0_finF
        A_0m_finF:
              Value at infinite time of ct_0m_finF
        A_0m_PASinF:
              Value at infinite time of ct_0m_PASinF 
        ct_prod:
              nr x nt array, product of the elements ct_finF
        ct:
              Directly calculated correlation function of the total motion
        S2:
              Final value of ct
        """
        print(out)
    
    
    @property
    def details(self):
        out=self.select.details
        tf,n,nr=(self.sampling_info[k] for k in ['tf','n','nr'])
        out.append('Processed data sampling: tf={0}, n={1}, nr={2}'.format(tf,n,nr))
        out.append('Frame processing mode is {0}'.format(self.mode))
        return out
    
    @property
    def traj(self):
        return self.select.traj
    
    @property
    def nf(self):
        """
        Number of frames

        Returns
        -------
        int

        """
        return len(self.frame_info['info'])
    
    def new_frame(self,Type=None,frame_index=None,**kwargs):
        """
        Create a new frame, where possible frame types are found in vec_funs.
        Note that if the frame function produces a different number of reference
        frames than there are bonds (that is, vectors produced by the tensor 
        frame), then a frame_index is required, to map the frame to the appropriate
        bond. The length of the frame_index should be equal to the number of 
        vectors produced by the tensor frame, and those elements should have 
        values ranging from 0 to one minus the number of frames defined by this
        frame. 
        
        To get a list of all implemented frames and their arguments, call this
        function without any arguments. To get arguments for a particular frame,
        call this function with only Type defined.
        """
        mol=self.select
        if Type is None:
            print_frame_info()
        elif len(kwargs)==0:
            print_frame_info(Type)
        else:
            assert self.vft is not None,'Define the tensor frame first (run mol.tensor_frame)'
            vft=self.vft()
            nb=vft[0].shape[1] if len(vft)==2 else vft.shape[1] #Number of bonds in the tensor frame
            fun,fi,info=new_fun(Type,mol,**kwargs)
            if frame_index is None:frame_index=fi #Assign fi to frame_index if frame_index not provided
            f=fun()    #Output of the vector function (test its behavior)
            nf=f[0].shape[1] if isinstance(f,tuple) or isinstance(f,list) else f.shape[1]
            if fun is not None:
                "Run some checks on the validity of the frame before storing it"
                if frame_index is not None:
                    assert frame_index.size==nb,'frame_index size does not match the size of the tensor_fun output ({0} bonds,{1} elements in frame_index)'.format(nb,frame_index.size)
                    assert frame_index[np.logical_not(np.isnan(frame_index))].max()<nf,'frame_index contains values that exceed the number of frames'
                    self.frame_info['frame_index'].append(frame_index)
                else:
                    assert nf==nb or nf==1,'No frame_index was provided, but the size of the tensor_fun and the frame_fun do not match'
                    if nf==1:
                        self.frame_info['frame_index'].append(np.zeros(nb))
                    else:
                        self.frame_info['frame_index'].append(np.arange(nb))
                self.frame_info['info'].append(info)
                self.vf.append(fun)    #Append the new function
                self.__frames_loaded=False
                self.__return_index=None    #This is the return index that was actually used
    
    def tensor_frame(self,Type='bond',label=None,**kwargs):
        """
        Creates a frame that defines the NMR tensor orientation. Usually, this
        is the 'bond' frame (default Type). However, other frames may be used
        in case a dipole coupling is not the relevant interaction. The chosen
        frame should return vectors defining both a z-axis and the xz-plane. A
        warning will be returned if this is not the case.
        """
        mol=self.select
        if Type is None:
            print_frame_info()
        elif len(kwargs)==0:
            print_frame_info(Type)
        else:
            if Type=='bond' and 'sel3' not in kwargs:
                kwargs['sel3']='auto'     #Define sel3 for the bond frame (define vXZ)
            
            self.vft,*_=new_fun(Type,mol,**kwargs) #New tensor function
            if len(self.vft())!=2:
                print('Warning: This frame only defines vZ, and not vXZ;')
                print('In this case, correlation functions may not be properly defined')
            if label is not None:
                self.frame_info['label']=label
            self.__frames_loaded=False
    
    def clear_frames(self):
        """
        Deletes all stored frame functions
        """
        self.vft=None
        self.vf=list()
        self.frame_info={'frame_index':list(),'label':None,'info':list()}
    
    def load_frames(self,n=-1,nr=10,index=None):
        """
        Sweeps through the trajectory and loads all frame and tensor vectors, 
        storing the result in vecs
        """
        tf=len(self.select.traj)
        index=sparse_index(tf,n,nr)
        if self.__frames_loaded:
            if np.all(self.vecs['index']==index) and self.sampling_info['t0']==self.traj.t0:return
        self.sampling_info={'tf':tf,'dt':self.select.traj.dt/1e3,'n':n,'nr':nr,'t0':self.traj.t0}
        self.vecs=mol2vec(self,index=index)
        self.__frames_loaded=True
        self.include=None   #If new frames loaded, we should re-set what frames used for correlation functions
        
    
    def select_frames(self,include:list=None):
        """
        Allows excluding some frames from the analysis, by specifying which frames
        to include based on a list of logicals

        Parameters
        ----------
        include : list, optional
            List of True/False the same length as the number of frames included.
            The default is None, which will include all frames.

        Returns
        -------
        dict.
            Dictionary containing the truncated set of vectors.

        """
        if include is None:return self.vecs
        vecs=self.vecs.copy()
        vecs['frame_index']=self.vecs['frame_index'].copy()
        vecs['v']=self.vecs['v'].copy()
        for k,v in zip(range(len(include)-1,-1,-1),include[::-1]):
            if not(v):
                vecs['frame_index'].pop(k)
                vecs['v'].pop(k)
        vecs['n_frames']=len(vecs['v'])
        return vecs
    
    def frame_names(self,include:list=None)->list:
        """
        Returns a list of strings which describing what motion each output data
        set corresponds to (ex. PAS>methylCC)

        Parameters
        ----------
        include : list, optional
            List of True/False the same length as the number of frames included.
            The default is None, which will include all frames.

        Returns
        -------
        list
            List of strings describing what motion is described by each output
            data set.

        """
        frame_names=[vf.__str__().split(' ')[1].split('.')[0] for vf in self.vf]
        out=list()
        for fr0,fr1 in zip(['PAS',*frame_names],[*frame_names,'LF']):
            out.append('{0}>{1}'.format(fr0,fr1))
        return out
    
    def frames2ct(self,mode='auto',return_index=None,include=None):
        """
        Converts vectors loaded by load_frames into correlation functions and
        tensors. One may specify the use of only specific frames by setting the
        variable 'include', which should be a list of logicals the same length
        as the number of frames (length of self.vf)
        """
        
        
        if mode=='auto':self.return_index.set2auto()
        if mode=='sym':self.return_index.set2sym()
        if mode=='direct':self.return_index.set2direct()
        
        
        
        assert include is  None or len(include)==len(self.vf),\
        "include index must have the same length ({0}) as the number of frames({1})".format(len(include),len(self.vf))
        include=np.ones(self.nf,dtype=bool) if include is None else np.array(include,dtype=bool)
        return_index=ReturnIndex(return_index) if return_index else self.return_index
        self.return_index=return_index
        
        "In the next lines, we determine whether the function needs to be run"
        run=not(hasattr(self,'ct_out'))
        if self.include is None or np.logical_not(np.all(include==self.include)):run=True #Not run before/new frames loaded/different frames used
        if not(self.mode==mode):run=True
        if self.__return_index is not None:
            for k in range(10):
                if return_index[k]!=self.__return_index[k]:run=True

        
        if run:
            if not(self.__frames_loaded):self.load_frames() #Load the frames if not already done
            
            self.__return_index=return_index.copy()
            
            "Here we check if post-processing is REQUIRED"
            for k,v in enumerate(self.vecs['v']):
                if v.shape[0]>2:
                    print('Post processing is required for some frames (running post-processing)')
                    self.post_process()
            
            
            "Here, we take out frames that aren't used"
            vecs=self.select_frames(include)

            out=frames2ct(v=vecs,return_index=return_index,mode=mode,rank=self.rank)
            if self.reduced_mem:
                self.vecs={}
                self.__frames_loaded=False
            
            self.mode=mode
            self.include=include
            
            self.ct_out=out
            self.Ct={}
            self.A={}
            self.t=None
            self.S2=None
            for k in out.keys():
                if k[:2]=='ct':self.Ct[k]=out[k]
                elif k[:1]=='A':self.A[k]=out[k]
                elif k=='t':self.t=out[k]
                elif k=='S2':self.S2=out[k]
            
        return self.ct_out
                
    def frames2data(self,mode:str='auto',return_index=None,include:list=None,rank:int=None):
        """
        Transfers the frames results to a list of data objects

        Parameters
        ----------
        mode : str, optional
            Determine whether to assume the motions have an axis of symmetry,
            where the full frames calculation is run in 'full' mode, we assume
            an axis of symmetry in 'sym' mode, and toggle between the two modes
            depending on a pre-calculation of eta in 'auto' mode. The default is 'auto'.
        return_index : TYPE, optional
            Index or object which determines what calculationg are run. 
            The default is None.
        include : list, optional
            List of which loaded frames to include in the calculation. 
            The default is None.
        rank : int, optional
            Rank of the correlation function (1 or 2). Rank 1 is only valid in
            symmetric mode. Rank 1 is primarily used for comparison of results
            to a rank-1 iRED calculation. If not specified, the stored value of
            rank will be used (usually 2). The default is None.

        Returns
        -------
        out : list
            List of data objects corresponding to the full correlation function,
            the product of frame correlation functions, and rotation between each
            frame.

        """
        if mode=='auto':self.return_index.set2auto()
        if mode=='sym':self.return_index.set2sym()
        
        if rank is not None:self.rank=rank #User-sepecified rank.
        
        self.frames2ct(mode=mode,return_index=return_index,include=include)
        out=ct2data(self.ct_out,self.select)  
        for o in out:o.details=self.details.copy()
        out[0].source.additional_info='Direct'        
        out[0].details.append('Direct analysis of the correlation function')
        
        if len(out)>1:
            out[1].source.additional_info='Product'
            out[1].details.append('Product of correlation functions from frame analysis')
                
            for o,fn in zip(out[2:],self.frame_names(include=include)):
                o.source.additional_info=fn
                o.details=self.details.copy()
                o.details.append('Rotation between frames '+' and '.join(fn.split('>')))
        
        out[0].sens.sampling_info=self.sampling_info
        
        "This allows us to turn _mdmode back off for the returned data objects"
        out[0].select=copy(out[0].select)
        out[0].select._mdmode=False
        for o in out:o.select=out[0].select
        if self.rank==1:
            for o in out:
                o.source.additional_info='rk1_'+o.source.additional_info if \
                    o.source.additional_info is not None else 'rk1'
        
        if self.project is not None:
            for o in out:
                self.project.append_data(o)
        
        if self.reduced_mem:
            delattr(self,'ct_out')
            self.Ct={}
        return out
    
    def md2data(self,rank:int=None):
        """
        Calculates only the direct correlation function and returns this as a
        data object

        Parameters
        ----------
        rank : TYPE, optional
            Rank of the correlation function. Usually set to 2 for the tensor
            correlation functions. However, can be set to 1, primarily for 
            comparing results to iRED analysis. The default is 2.

        Returns
        -------
        None.

        """
        if rank is not None:self.rank=rank
        
        if self.vft is None:self.tensor_frame(Type='bond',sel1=1,sel2=2)
        
        if hasattr(self,'ct_out') and 'ct' in self.ct_out:
            out=ct2data(self.ct_out,self.select)[0]
            out.details=self.details.copy()
            out.details.append('Direct analysis of the correlation function')
            out.source.Type='MD'
            out.source.additional_info='rk1' if self.rank==1 else None
            out.sens.sampling_info=self.sampling_info
            if self.project is not None:self.project.append_data(out)
        else:
            self.return_index.set2direct()
            include=[False for _ in range(len(self.vf))]
            out=self.frames2data(mode='direct',include=include,rank=rank)[0]
            out.source.Type='MD'
            out.source.additional_info='rk1' if self.rank==1 else None
            out.sens.sampling_info=self.sampling_info
            if self.project is not None:self.project.update_info()
        return out
    
    def frames2iRED(self, rank:int=2, include: list = None) -> list:
        """
        Sets the frames mode to symmetric and extracts vectors for each frame
        required to perform iRED analysis.

        Parameters
        ----------
        rank : int, optional
            1 or 2, giving the rank of the iRED analysis. The default is 2.
        include : list, optional
            List of logicals the same length as the number of loaded frames, 
            used to determine whether or not to include each frame.
            The default is None, which will include all frames

        Returns
        -------
        list
            Contains a list of the iRED objects corresponding to the total 
            motion, and the frame-separated motions (total motion first).

        """
        if not(self.__frames_loaded):self.load_frames() #Load the frames if not already done
        self.return_index.set2sym() #Set to symmetric mode
        self.mode='sym'
        include=np.ones(len(self.vf),dtype=bool) if include is None else np.array(include,dtype=bool)
        assert len(include)==len(self.vf),\
        "include index must have the same length ({0}) as the number of frames({1})".format(len(include),len(self.vf))
    
        v=self.select_frames(include)
        index=v['index']
        source=clsDict['Source'](Type='Fr2iREDmode',select=copy(self.select),filename=self.select.traj.files,
                      status='raw')
        source.select._mdmode=False #Turn off md mode for export!
        source.details=self.details.copy()
        source.project=self.project
        
        if len(v['v']):
            vZ,vXZ,nuZ,nuXZ,_=apply_fr_index(v)
            nf=len(nuZ)
        else:
            vZ=v['vT'][0] if v['vT'].shape[0]==2 else v['vT']
            vZ/=np.sqrt((vZ**2).sum(0))
            
            source.details.append('Direct analysis of the correlation function')
            source.details.append('Analyzed with iRED')
            out=[iRED({'v':vZ,'t':v['t'],'index':index,'source':source,'sampling_info':self.sampling_info},rank=rank)]
            return out
        
    
        
        nr,nt=vZ.shape[1:]
    
        A_0m_PASinf=list()
        for k in range(nf):
            vZ_inf=vft.applyFrame(vft.norm(vZ),nuZ_F=nuZ[k],nuXZ_F=nuXZ[k])
            A_0m_PASinf.append(vft.D2vec(vZ_inf).mean(axis=-1))
        
        source.details.append('Direct analysis of the correlation function')
        source.select.molsys=copy(source.select.molsys)
        source.select.molsys.make_pdb(ti=0)
        
        
        out=[iRED({'v':vZ,'t':v['t'],'index':index,'source':source,'sampling_info':self.sampling_info},rank=rank)]
        for k,fn in zip(range(nf+1),self.frame_names(include)):
            if k==0:
                v0=vft.applyFrame(vft.norm(vZ),nuZ_F=nuZ[k],nuXZ_F=nuXZ[k])
            elif k==nf:
                A0,nuZ_f,nuXZ_f,nuZ_F,nuXZ_F=A_0m_PASinf[k-1],nuZ[k-1],nuXZ[k-1],None,None
                v0=sym_nuZ_f(A_0m_PASinf=A0,nuZ_f=nuZ_f,nuXZ_f=nuXZ_f,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F)
            else:
                A0,nuZ_f,nuXZ_f,nuZ_F,nuXZ_F=A_0m_PASinf[k-1],nuZ[k-1],nuXZ[k-1],nuZ[k],nuXZ[k] 
                v0=sym_nuZ_f(A_0m_PASinf=A0,nuZ_f=nuZ_f,nuXZ_f=nuXZ_f,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F)
        
            
            source=copy(source)
            source.details[-1]='Rotation between frames '+' and '.join(fn.split('>'))
            source.details.append('Analyzed with iRED')
            source.additional_info=fn
            out.append(iRED({'v':v0,'t':v['t'],'index':index,'source':source,'sampling_info':self.sampling_info},rank=rank))
            out[-1].md2data=self.md2data
            
        return out
            
    def md2iRED(self,rank:int=2):
        """
        Extracts vectors describing only the full motion for use in the iRED 
        analysis

        Returns
        -------
        dict
            Contains a list of the vectors for further analysis with iRED, as
            well as information about sampling of the time axis.

        """
        if self.vft is None:self.tensor_frame(Type='bond',sel1=1,sel2=2)
        include=[False for _ in range(len(self.vf))]
        out=self.frames2iRED(rank=rank,include=include)[0]
        out.source.Type='iREDmode'
        out.source.select.molsys=copy(out.source.select.molsys)
        out.source.select.molsys.make_pdb(ti=0)
        out.md2data=self.md2data
        return out

    #TODO add back in some version of draw tensors        
    # def draw_tensors(self,fr_num,tensor_name='A_0m_PASinF',sc=2.09,tstep=0,disp_mode=None,index=None,scene=None,\
    #              fileout=None,save_opts=None,chimera_cmds=None,\
    #              colors=[[255,100,100,255],[100,100,255,255]],marker=None,\
    #              marker_color=[[100,255,100,255],[255,255,100,255]]):
        
    #     assert tensor_name in self.A.keys(),'Tensors not found, set tensor_name to one of {0}'.format(self.A.keys())
        
    #     draw_tensors(self.A[tensor_name][fr_num],mol=self.molecule,sc=sc,tstep=tstep,\
    #                  disp_mode=disp_mode,index=index,scene=scene,fileout=fileout,\
    #                  save_opts=save_opts,chimera_cmds=chimera_cmds,colors=colors,\
    #                  marker=marker,marker_color=marker_color,vft=self.vft)
        
    def post_process(self,Type=None,*args,**kwargs):
        if Type is None:            
            if 'fr_ind' in kwargs:
                k=kwargs['fr_ind']
                info=self.frame_info['info'][k]
                if 'PPfun' in info:
                    print('Applying default post processing to frame {0}'.format(k))
                    Type=info['PPfun']
                    assert hasattr(FPP,Type),'Unknown post-processing method set in defaults'
                    info_in=info.copy()
                    info_in.pop('PPfun')
                    self.post_process(Type=Type,fr_ind=k,**info_in)
                else:
                    print('Warning: No default post processing for frame {0}'.format(k))
            else:
                print('Applying default post processing (only active for frames that define their own post processing)')
                for k,info in enumerate(self.frame_info['info']):
                    if 'PPfun' in info:
                        Type=info['PPfun']
                        assert hasattr(FPP,Type),'Unknown post-processing method set in defaults'
                        info_in=info.copy()
                        info_in.pop('PPfun')
                        self.post_process(Type=Type,fr_ind=k,**info_in)
        else:
            assert hasattr(self,'vecs'),'No frames have been loaded (load frames before post processing)'
            assert hasattr(FPP,Type),'Unknown post-processing method'
            if not(hasattr(self,'_vecs')) and not(self.reduced_mem):
                self._vecs=deepcopy(self.vecs)
            getattr(FPP,Type)(self.vecs,*args,**kwargs)
            self.include=None

    
    def remove_post_process(self):
        if hasattr(self,'_vecs'):
            self.vecs=self._vecs
            delattr(self,'_vecs')
            self.include=None
        else:
            print('No post-processing to remove (Is reduced_mem set to True?)')
        
        
    

#%% Output functions
"This is the usual output– go from a MolSelect object to a data object"
def frames2data(mol=None,v=None,mode='full',n=100,nr=10,tf=None,dt=None):
    """
    Calculates the correlation functions (frames2ct) and loads the result into
    data objects (ct2data)
    """
    
    if mode=='full':
        return_index=[True,False,False,False,True,False,True,True,True,True]
    else:
        return_index=[True,False,False,False,False,False,True,True,True,True]
    
    ct_out=frames2ct(mol=mol,v=v,return_index=return_index,mode=mode,n=n,nr=nr,tf=tf,dt=dt)
    
    
    out=ct2data(ct_out,mol)    
    
    return out

def frames2tensors(mol=None,v=None,n=100,nr=10,tf=None,dt=None):
    """
    Calculates the various residual tensors for a set of frames, returned in a
    dictionary object. (This function is simply running frames2ct with 
    return_index set to True only for the time-independent terms)
    """
    return_index=[False,False,False,False,True,True,True,False,False,True]
    
    return frames2ct(mol=mol,v=v,return_index=return_index,n=n,nr=nr,tf=tf,dt=dt)

"Here we go from the output of frames2ct and load it into a data object"
def ct2data(ct_out,mol=None):
    """
    Takes the results of a frames2ct calculation (the ct_out dict) and loads 
    the results into a data object(s) for further processing. One data object 
    will be returned for ct, if included in the ct_out dict, one for ct_prod,
    also if included, and the subsequent results are for each frame of 
    ct_finF.

    Within ct_finF, we also include A_m0_finF, and also A_0m_PASinF (A_0m_PASinF 
    is calculated in the previous, so that a given frame contains the equilibrium
    values used to construct ct_finF). The equilibrium tensor for all motion
    can be found in ct_prod (technically, this is A_m0_PASinF where F is
    the lab frame).
    """
    
    out=list()
    
    md=clsDict['MD'](t=ct_out['t'])
    stdev=np.repeat([md.info['stdev'].astype(Defaults['dtype'])],ct_out['ct'].shape[0],axis=0)
    if 'ct' in ct_out:
        data=clsDict['Data'](R=ct_out['ct'],Rstd=stdev,sens=md,select=mol,Type='Frames')
#        data.Rstd[:]=stdev #Copy stdev for every data point
        data.source.additional_info='Direct'
        data.source.filename=mol.traj.files
        data.source.status='raw'
        if 'S2' in ct_out:
            data.S2=ct_out['S2']
        out.append(data)
    
    if 'ct_prod' in ct_out:
        data=clsDict['Data'](R=ct_out['ct_prod'],Rstd=stdev,sens=md,select=mol,Type='Frames')
#        data.Rstd[:]=stdev #Copy stdev for every data point
        data.source.additional_info='Product'
        data.source.filename=mol.traj.files
        data.source.status='raw'
        data.tensors=dict()
        if 'A_0m_PASinF' in ct_out:
            data.tensors['A_0m_PASinF']=ct_out['A_0m_PASinF'][-1]
        data.detect=out[0].detect
        out.append(data)
    
    if 'ct_finF' in ct_out:
        for k,ct0 in enumerate(ct_out['ct_finF']):
            data=clsDict['Data'](R=ct0,Rstd=stdev,sens=md,select=mol,Type='Frames')
#            data.Rstd[:]=stdev #Copy stdev for every data point
            data.source.filename=mol.traj.files
            data.source.status='raw'
            data.tensors=dict()
            if 'A_m0_finF' in ct_out:
                data.tensors['A_m0_finF']=ct_out['A_m0_finF'][k]
            if 'A_0m_finF' in ct_out:
                data.tensors['A_0m_finF']=ct_out['A_0m_finF'][k]
            if 'A_0m_PASinF' in ct_out:
                if k==0:
                    A0=np.zeros([5,out[-1].R.shape[0]])
                    A0[2]=1
                else:
                    A0=ct_out['A_0m_PASinF'][k-1]
                data.tensors['A_0m_PASinF']=A0
            data.detect=out[0].detect
            out.append(data)

    ms=copy(out[0].select.molsys)
    ms.make_pdb(ti=0)
    for d in out:
        d.select.molsys=ms

    return out


#%% Main calculations
"Applies indices for frames"
def apply_fr_index(v,squeeze=True):
    """
    Expands the output of mol2vec such that all frames have the same number of
    elements (essentially apply the frame_index). This is done in such a way
    that if a frame is missing for one or more residues (frame_index set to nan),
    then the motion appears in the next frame out.
    """
    nu=[(v0,None) if len(v0)!=2 else v0 for v0 in v['v']]     #Make sure all frames have 2 elements
    vZ,vXZ=(v['vT'],np.ones(v['vT'].shape)*np.nan) if len(v['vT'])!=2 else (v['vT'][0],v['vT'][1])    #Bond vector (and XZ vector) in the lab frame
    nf=len(v['v'])
    nr,nt=vZ.shape[1:]
    
    fi=v['frame_index']
    fiout=list()
    nuZ=list()
    nuXZ=list()

    iF0=list()

    for k in range(nf):
        iF=np.isnan(fi[k])
        iF0.append(iF)
        iT=np.logical_not(iF)
        nuZ.append(np.zeros([3,nr,nt]))
        nuXZ.append(np.zeros([3,nr,nt]))
        nuXZ[-1][:]=np.nan
        
        
        nuZ[-1][:,iT]=nu[k][0][:,fi[k][iT].astype(int)]
        nuZ[-1][:,iF]=vZ[:,iF] if k==0 else nuZ[k-1][:,iF]
        if nu[k][1] is not None:
            nuXZ[-1][:,iT]=nu[k][1][:,fi[k][iT].astype(int)]
        
        nuXZ[-1][:,iF]=vXZ[:,iF] if k==0 else nuXZ[k-1][:,iF]
        
        fiout.append(np.zeros(nr,dtype=int))
        fiout[-1][iT]=fi[k][iT]
        fiout[-1][iF]=np.arange(nr)[iF] if k==0 else -1
    
    k=0
    while k<nf-1:
        test1=np.logical_and(iF0[k],iF0[k+1])
        test2=np.logical_or(iF0[k],iF0[k+1])
        if np.all(np.logical_not(test1)) and np.all(test2):
            "Fill in values of nuZ[k],nuXZ[k], from nuZ[k+1], move all values down, delete last element"
            nuZ[k][:,iF0[k]]=nuZ[k+1][:,iF0[k]]
            nuXZ[k][:,iF0[k]]=nuXZ[k+1][:,iF0[k]]
            
            fiout[k][iF0[k]]=fiout[k+1][iF0[k]]+np.max(fiout[k])+1
            nuZ[k+1:-1]=nuZ[k+2:]
            nuXZ[k+1:-1]=nuXZ[k+2:]
            fiout[k+1:-1]=fiout[k+2:]
            nuZ.pop()
            nuXZ.pop()
            fiout.pop()
        k+=1
        
    "Fix the frame_index"
    fiout0=fiout
    fiout=list()
    for k in range(len(fiout0)+1):
        if k==0:
            fiout.append(np.zeros(nr,dtype=int))
            fiout[-1][fiout0[k]<0]=-1
            fiout[-1][fiout[k]>=0]=np.arange(np.sum(fiout[k]>=0))
        elif k==len(fiout0):
            fiout.append(np.array(fiout0[-1],dtype=int))
            m=k-2
            while np.any(fiout[-1]<0):
                fiout[-1][fiout[-1]<0]=fiout0[m][fiout[-1]<0]+np.max(fiout0)
                m+=-1
        else:
            fiout.append(np.array(fiout0[k-1],dtype=int))
            m=k-2
            while np.any(fiout[-1]<0):
                fiout[-1][fiout[-1]<0]=fiout0[m][fiout[-1]<0]+np.max(fiout0)
                m+=-1
            fiout[-1][fiout0[k]<0]=-1
        
        
    
    "Make sure vectors are normalized"
    vZ=vft.norm(vZ)
    nuZ=[vft.norm(nuz) for nuz in nuZ]
    
    return vZ,vXZ,nuZ,nuXZ,fiout


"This function handles the organization of the output, determines which terms to calculate"
def frames2ct(mol=None,v=None,return_index=None,mode='full',n=100,nr=10,t0=0,tf=None,dt=None,rank:int=2):
    """
    Calculates correlation functions for frames (f in F), for a list of frames.
    One may provide the MolSelect object, containing the frame functions, or
    the output of mol2vec (or ini_vec_load). frames2data returns np arrays with
    the following data
    
    If we have n frames (including the tensor frame), nr residues (tensors), and
    nt time points in the resulting correlation function, we can calculate any
    of the following functions:
    
    ct_finF     :   n x nr x nt array of the real correlation functions for each
                    motion (after scaling by residual tensor of previous motion)
    ct_m0_finF  :   5 x n x nr x nt array, with the individual components of 
                    each motion (f in F)
    ct_0m_finF  :   5 x n x nr x nt array, with the individual components of 
                    each motion (f in F)
    ct_0m_PASinF:   5 x n x nr x nt array, with the individual components of 
                    each motion (PAS in F)
    A_m0_finF   :   Value at infinite time of ct_m0_finF
    A_0m_finF   :   Value at infinite time of ct_0m_finF
    A_0m_PASinF :   Value at infinite time of ct_0m_PASinF 
    ct_prod     :   nr x nt array, product of the elements ct_finF
    ct          :   Directly calculated correlation function of the total motion
    S2          :   Final value of ct
    
    Include a logical index to select which functions to return, called return_index
    
    Default is
    return_index=[True,False,False,False,False,False,False,True,True,False]
    that is, ct_finF, ct_prod, and ct are included in the default.
    
    That is, we calculate the individual correlation functions, the product of
    those terms, and the directly calculated correlation function by default.
    
    frames2ct(mol=None,v=None,return_index=None,n=100,nr=10,nf=None,dt=None)
    
    We may also take advantage of symmetry in the various motions. This option
    is obtained by changing mode from 'full' to either 'sym' (assume all motions
    result in symmetric residual tensors, A_0m_PASinF has eta=0), or we set mode
    to 'auto', where a threshold on eta determines whether or not we treat the
    residual tensor as symmetric. By default, the threshold is 0.2, but the user
    may set the mode to autoXX, where XX means eta should be less than 0.XX (one
    may use arbitrary precision, autoX, autoXX, autoXXX, etc.). 
    """
    
    
    if v is None and mol is None:
        print('mol or v must be given')
        return
    elif v is None:
        v=mol2vec(mol,n=n,nr=nr,t0=t0,tf=tf,dt=dt)

    ri=ReturnIndex(return_index)
    
    index=v['index']
    
    if len(v['v']):
        vZ,vXZ,nuZ,nuXZ,_=apply_fr_index(v)
        nf=len(nuZ)
    else:
        nf=0
        vZ=v['vT'][0] if v['vT'].shape[0]==2 else v['vT']
        for k in ri.flags.keys():
            if k not in ['ct','S2']:ri.flags[k]=False
    

    assert rank==2 or (mode=='direct' or mode=='sym'),'rank 1 calculations can only be done in "sym" mode'
    
    nr,nt=vZ.shape[1:]

    "Initial calculations/settings required if using symmetry for calculations"
    if mode.lower()=='sym' or 'auto' in mode.lower():
    
        if mode.lower()=='sym':ri.set2sym()
        if mode.lower()=='auto':ri.set2auto()
        
        A_0m_PASinf=list()
        for k in range(nf):
            vZ_inf=vft.applyFrame(vft.norm(vZ),nuZ_F=nuZ[k],nuXZ_F=nuXZ[k])
            A_0m_PASinf.append(vft.D2vec(vZ_inf).mean(axis=-1))
    else:
        A_0m_PASinf=[None for _ in range(nf)]
    
    
    if mode=='sym':
        threshold=1
    elif 'auto' in mode.lower():
        threshold=float(mode[4:])/10**(len(mode)-4) if len(mode)>4 else 0.2   #Set threshold for eta (default 0.2)
    else:
        threshold=0  
        

    if ri.calc_ct_m0_finF:
        "Calculate ct_m0_finF if requested, if ct_prod requested, if ct_finF requested, or if ct_0m_finF requested"
        ct_m0_finF=list()
        A_m0_finF=list()
        for k in range(nf+1):
            if k==0:
                a,b=Ct_D2inf(vZ=vZ,vXZ=vXZ,nuZ_F=nuZ[k],nuXZ_F=nuXZ[k],cmpt='m0',mode='both',index=index,rank=rank)
            elif k==nf:
                a,b=sym_full_swap(vZ=vZ,threshold=threshold,A_0m_PASinf=A_0m_PASinf[k-1],vXZ=vXZ,\
                              nuZ_f=nuZ[k-1],nuXZ_f=nuXZ[k-1],cmpt='m0',mode='both',index=index,rank=rank)
            else:
#                a,b=Ct_D2inf(vZ=vZ,vXZ=vXZ,nuZ_f=nuZ[k-1],nuXZ_f=nuXZ[k-1],nuZ_F=nuZ[k],nuXZ_F=nuXZ[k],cmpt='m0',mode='both',index=index)
                a,b=sym_full_swap(vZ=vZ,threshold=threshold,A_0m_PASinf=A_0m_PASinf[k-1],vXZ=vXZ,\
                              nuZ_f=nuZ[k-1],nuXZ_f=nuXZ[k-1],nuZ_F=nuZ[k],nuXZ_F=nuXZ[k],\
                              cmpt='m0',mode='both',index=index,rank=rank)
            ct_m0_finF.append(a)
            A_m0_finF.append(b)
        # ct_m0_finF=np.array(ct_m0_finF)
        if ri.calc_A_m0_finF:
            A_m0_finF=np.array(A_m0_finF)

    elif ri.calc_A_m0_finF:
        "Calculate A_m0_finF if requested, or A_0m_finF requested"
        A_m0_finF=list()
        for k in range(nf+1):
            if k==0:
                b=Ct_D2inf(vZ=vZ,vXZ=vXZ,nuZ_F=nuZ[k],nuXZ_F=nuXZ[k],cmpt='m0',mode='d2',index=index)
            elif k==nf:
                b=Ct_D2inf(vZ=vZ,vXZ=vXZ,nuZ_f=nuZ[k-1],nuXZ_f=nuXZ[k-1],cmpt='m0',mode='d2',index=index)
            else:
                b=Ct_D2inf(vZ=vZ,vXZ=vXZ,nuZ_f=nuZ[k-1],nuXZ_f=nuXZ[k-1],nuZ_F=nuZ[k],nuXZ_F=nuXZ[k],cmpt='m0',mode='d2',index=index)
            A_m0_finF.append(b)
        A_m0_finF=np.array(A_m0_finF)

    if ri.ct_0m_finF:
        "ct_0m_finF are just the conjugates of ct_m0_finF"
        # ct_0m_finF=np.array([ct0.conj() for ct0 in ct_m0_finF])
        ct_0m_finF=[ct0.conj() for ct0 in ct_m0_finF]
    
    if ri.A_0m_finF:
        "A_0m_finF are just the conjugates of A_m0_finF"
        A_0m_finF=np.array([a0.conj() for a0 in A_m0_finF])
      
    if ri.ct_0m_PASinF:  #This option is deactivated for sym and auto modes
        "Calculate ct_0m_PASinF if requested"
        ct_0m_PASinF=list()
        A_0m_PASinF=list()
        for k in range(nf+1):
            if k==nf:
                a,b=Ct_D2inf(vZ=vZ,vXZ=vXZ,cmpt='0m',mode='both',index=index)
            else:
                a,b=Ct_D2inf(vZ=vZ,vXZ=vXZ,nuZ_F=nuZ[k],nuXZ_F=nuXZ[k],cmpt='0m',mode='both',index=index)
            ct_0m_PASinF.append(a)
            A_0m_PASinF.append(b)
        # ct_0m_PASinF=np.array(ct_0m_PASinF)
        A_0m_PASinF=np.array(A_0m_PASinF)
    elif ri.calc_A_0m_PASinF:
        "Calculate A_0m_PASinF if requested, if ct_prod requested, or if ct_finF requested"
        A_0m_PASinF=list()
        for k in range(nf+1):
            if k==nf:
                b=Ct_D2inf(vZ=vZ,vXZ=vXZ,cmpt='0m',mode='D2',index=index)
            else:
                b=Ct_D2inf(vZ=vZ,vXZ=vXZ,nuZ_F=nuZ[k],nuXZ_F=nuXZ[k],cmpt='0m',mode='D2',index=index)
            A_0m_PASinF.append(b)
        A_0m_PASinF=np.array(A_0m_PASinF)
    
    if ri.calc_ct_finF:
        "Calculate ct_finF if requested, or if ct_prod requested"
        ct_finF=list()
        for k in range(nf+1):
            if k==0:
                ct_finF.append(ct_m0_finF[0][2].real)
            else:
                ct_finF.append((np.moveaxis(ct_m0_finF[k],-1,0)*A_0m_PASinF[k-1]/A_0m_PASinF[k-1][2].real).sum(1).real.T)
        # ct_finF=np.array(ct_finF)

    if ri.ct_prod:
        "Calculate ct_prod"
        # ct_prod=ct_finF.prod(0)
        ct_prod=np.prod(ct_finF,0)
        
    if ri.ct:
        "Calculate ct if requested"
        ct,S2=Ct_D2inf(vZ,cmpt='00',mode='both',index=index,rank=rank)
        ct=ct.real
        S2=S2.real
    elif ri.S2:
        "Calculate S2 if requested"
        S2=Ct_D2inf(vZ,cmpt='00',mode='d2',index=index,rank=rank)
        S2=S2.real
    
    out=dict()
    for k in ri.flags.keys():
        if getattr(ri,k):out[k]=locals()[k]

    if ri.calc_any_ct:
        if index is None:
            index=np.arange(v['vT'].shape[-1])
        out['index']=index
        N=get_count(index)
        i=N!=0
        N=N[i]
        dt=(v['t'][1]-v['t'][0])/(index[1]-index[0])
        t=(np.cumsum(i)-1)*dt
        out['N']=N
        out['t']=t[i]
    
    return out

"This function extracts various frame vectors from trajectory"
def mol2vec(fr_obj,n=100,nr=10,index=None):
    """
    Extracts vectors describing from the frame functions found in the MolSelect
    object. Arguments are mol, the molecule object, n and nr, which are parameters
    specifying sparse sampling, and dt, which overrides dt found in the trajectory
    """
    
    traj=fr_obj.select.traj    
    tf=len(traj)
    if index is None:
        index=sparse_index(tf,n,nr)
    
    return ini_vec_load(traj,fr_obj.vf,fr_obj.vft,fr_obj.frame_info['frame_index'],index=index,info=fr_obj.frame_info['info'])
    
"This function takes care of the bulk of the actual calculations"
def Ct_D2inf(vZ,vXZ=None,nuZ_F=None,nuXZ_F=None,nuZ_f=None,nuXZ_f=None,cmpt='0p',mode='both',index=None,rank:int=2):
    """
    Calculates the correlation functions and their values at infinite time
    simultaneously (reducing the total number of calculations)
    
    To perform the calculation in reference frame F, provide nuZ_F and 
    optionally nuXZ_F
    
    To calculate the effect of the motion of frame f on the correlation function
    for the bond, provide nuZ_f and optionally nuXZ_f
    
    To only return the correlation function, or only return the values at infinite
    time, set mode to 'Ct' or 'D2inf', respectively.
    
    To determine what terms to calculate, set cmpt:
        '0p' yields the 5 terms, C_0p (default)
        'p0' yields the 5 terms, C_p0
        '01','20','00','-20', etc. all will return the requested component
        
    Setting m OR mp will automatically set the other term to 0. Default is for
    mp=0 (starting component), and m is swept from -2 to 2. 
    
    Currently, m or mp must be zero
    
    index should be provided if the trajectory has been sparsely sampled.
    
    ct,d2=Ct_D2inf(vZ,vXZ=None,nuZ_F=None,nuXZ_F=None,nuZ_f=None,nuXZ_f=None,cmpt='0p',mode='both',index=index)
    
    if mode is 'd2', only d2 is returned (even if index is provided). F
    if mode is 'ct', d2 is not returned
    if mode is 'both', then ct and d2 are returned
    """
    
    
    """Rather than having a bunch of if/then statements, we're just going to make
    a logical array to determine which terms get calculated in this run. Note
    symmetry relations: we will use mmpswap to get p0 terms
    
    calc=[0-2,0-1,00,01,02]
    
    Note that we'll skip terms that can be obtained based on their relationship
    to other terms, and fill these in at the end (ex. if C_01 and C_10 are required,
    we'll only calculate one, and get the other from the negative conjugate)
    """
    calc=np.zeros(5,dtype=bool)
    mmpswap=False

    if cmpt in ['0m','m0','0p','p0']:
        calc[:3]=True
        if cmpt in ['m0','p0']:mmpswap=True
    elif cmpt in ['0-2','0-1','00','01','02']:
        calc=np.array(['0-2','0-1','00','01','02'])==cmpt
    elif cmpt in ['-20','-10','10','20']:
        calc=np.array(['-20','-10','00','10','20'])==cmpt
        mmpswap=True
    

     
    #Flags for calculating correlation function or not, and how to calculate
    calc_ct=True if (mode[0].lower()=='b' or mode[0].lower()=='c') else False
    ctc=Ctcalc(mode='a',index=index,calc_ct=calc_ct,length=5)

    "Here we create a generator that contains each term in the correlation function"
    l=loops(vZ=vZ,vXZ=vXZ,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F,nuZ_f=nuZ_f,nuXZ_f=nuXZ_f,calc=calc,rank=rank)
    for k,l0 in enumerate(l):
        "These terms appear in all correlation functions"
        if rank==1:
            zzp=l0['az']
        elif 'eag' in l0.keys():
            zzp=l0['eag']*l0['ebd']
        else:
            zzp=l0['az']*l0['bz']

        
        """zz is the common component for all correlation functions, so we assign
        it to a of the correlation function calculator
        """
        ctc.a=zzp
        "Loop over all terms C_0p"
        for k,ctc0 in enumerate(ctc):
            if calc[k]:             #Loop over all terms
                p=ct_prods(l0,k,rank=rank)
                "Assigning p to b to correlate it with the zz component"
                ctc[k].b=p      
                ctc[k].add()     
       
    """
    Previously, we had implemented a complex conjugate before inverse FT, in
    order to account for non-symmetric correlation functions. Currently, this
    is not impolemented, although would be possible with the CtCalc object
    """    
    #Calculate results
    ct,d2=list(),list()
    offsets=[0,0,-1/2 if rank==2 else 0,0,0]
    for ctc0,offset in zip(ctc,offsets):
        out=ctc0.Return(offset=offset)
        ct.append(out[0])
        d2.append(out[1])
    
    "Add offsets to terms C_pp"

    "If a particular value selected with m=0,mp!=0 (C_p0), apply the m/mp swap"
    if mmpswap:     
        d2=[m_mp_swap(d2,0,k-2,k-2,0) for k,d2 in enumerate(d2)]
        ct=[m_mp_swap(ct0,0,k-2,k-2,0) for k,ct0 in enumerate(ct)]
    
    "Remove extra dimension if input was one dimensional"
    if vZ.ndim==2:
        d2=[None if d21 is None else d21.squeeze() for d21 in d2]
        ct=[None if ct1 is None else ct1.squeeze() for ct1 in ct]
        
    """Extract only desired terms of ct,d2 OR"
    fill in terms of ct, d2 that are calculated with sign swaps/conjugates"""   
    if cmpt in ['0m','m0','0p','p0']:
        d2[3],d2[4]=-d2[1].conj(),d2[0].conj()
        d2=np.array(d2)
        if calc_ct:
            ct[3],ct[4]=-ct[1].conj(),ct[0].conj()
            ct=np.array(ct)
    elif cmpt in ['0-2','0-1','00','01','02','-20','-10','10','20']:
        i=np.argwhere(calc).squeeze()
        d2=d2[i]
        if calc_ct:ct=ct[i]
    
    "Return the requested terms"
    if mode[0].lower()=='b':    #both (just check match on first letter)
        return ct,d2
    elif mode[0].lower()=='c':  #ct only
        return ct
    else:                       #D2inf only
        return d2

"Used in conjunction with loops to calculate the required terms for the correlation functions"
def ct_prods(l,n,rank:int=2):
    """
    Calculates the appropriate product (x,y,z components, etc.) for a given
    correlation function's component. Provide l, the output of a generator from
    loops. Mean may be taken, or FT, depending on if final value or correlation 
    function is required. 
    
    n determines which term to calculate. n is 0-8, indexing the following terms
    
    [0-2,0-1,00,01,02,-2-2,-1-1,11,22]
    
    """


    if rank==1:
        return l['az']

    if n==0:
        p=np.sqrt(3/8)*(l['ax']*l['bx']-l['ay']*l['by']+1j*2*l['ax']*l['by'])
    if n==1:
        p=-np.sqrt(3/2)*(l['ax']*l['bz']+1j*l['ay']*l['bz'])
    if n==2:
        p=3/2*l['az']*l['bz']
    if n==3:
        p=np.sqrt(3/2)*(l['ax']*l['bz']-1j*l['ay']*l['bz'])
    if n==4:
        p=np.sqrt(3/8)*(l['ax']*l['bx']-l['ay']*l['by']-1j*2*l['ax']*l['by'])
    if n==5 or n==8:
        p=l['az']*l['bz']
        p1=1/2*l['az']*l['gz']
    if n==6 or n==7:
        p=1/4*l['az']*l['bz']
        p1=1/2*l['az']*l['gz']
    
    if 'eag' in l.keys():
        p*=l['gz']*l['dz']
    
    if n>4:
        return p,p1
    else:
        return p
        
        
        
"Generator object to loop over when calculating correlation functions/residual tensors"
def loops(vZ,vXZ=None,nuZ_F=None,nuXZ_F=None,nuZ_f=None,nuXZ_f=None,calc=None,rank:int=2):
    """
    Generator that calculates the elements required for the loop over components
    for each correlation function. 
    
    All arguments must be provided in the same frame (typically, the lab frame,
    although other frames may be used)
    
    Vectors
    vZ:     Direction of the bond
    vXZ:    Vector in XZ plane of the bond frame (usually another bond). Required
            if calculating bond motion (warning produced if frame F is defined
            but frame f is not, and vXZ is omitted).
    nuZ_F:  Z-axis of frame F (motion of F is removed). Optional
    nuXZ_F: Vector in XZ plane of frame F (if F defined with two vectors). Optional
    nuZ_f:  Z-axis of frame f. Used if calculating motion of f in F. Optional
    nuXZ_f: Vector in XZ plane of frame f (if f defined with two vectors). Optional
    
    Arguments:
    calc:   Logical of 9 elements, to determine which terms to return. If set to
            None, all elements will be returned
            
    loops(vZ,vXZ=None,nuZ_F=None,nuXZ_F=None,nuZ_f=None,nuXZ_f=None,calc=None)
    """
    
    if calc is None:calc=np.ones(9,dtype=bool)
    
    vZ,nuZ_F,nuZ_f=vft.norm(vZ),vft.norm(nuZ_F),vft.norm(nuZ_f) #Make sure terms are normalized
    
    "Apply frame F (remove motion of frame F)"
    vZF,vXZF,nuZ_fF,nuXZ_fF=vft.applyFrame(vZ,vXZ,nuZ_f,nuXZ_f,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F)
    
    if np.any(calc[[0,1,3,4]]): #Do we need X and Y axes for the bond frame?
        sc=vft.getFrame(vZF,vXZF)   
        vXF,vYF=vft.R([1,0,0],*sc),vft.R([0,1,0],*sc)
    else:
        vXF=[None,None,None]    #Just set to None if not required
        vYF=[None,None,None]
    
    if rank==1: #Rank 1 symmetric calculation. Only z-components, correlated with self (3 loop elements)
        for az in vZF:
            out={'az':az}
            yield out
    elif nuZ_f is None:   #This is a bond in frame calculation (9 loop elements)
        for ax,ay,az in zip(vXF,vYF,vZF):
            for bx,by,bz in zip(vXF,vYF,vZF):
                out={'az':az,'bz':bz}
                
                if calc[0] or calc[4]:  #All terms required
                    out.update({'ax':ax,'bx':bx,'ay':ay,'by':by})
                elif calc[1] or calc[3]:    #Some terms required
                    out.update({'ax':ax,'ay':ay})
                yield out   #Only z terms required

    else: #This is a frame (f) in frame (F) calculation (81 loop elements)
        
        scfF=vft.getFrame(nuZ_fF,nuXZ_fF)
        vZf=vft.R(vZF,*vft.pass2act(*scfF))

        eFf=[vft.R([1,0,0],*vft.pass2act(*scfF)),\
             vft.R([0,1,0],*vft.pass2act(*scfF)),\
             vft.R([0,0,1],*vft.pass2act(*scfF))]
        
        for ea,ax,ay,az in zip(eFf,vXF,vYF,vZF):
            for eb,bx,by,bz in zip(eFf,vXF,vYF,vZF):
                for eag,gz in zip(ea,vZf):
                    for ebd,dz in zip(eb,vZf):
                        if calc[0] or calc[4]:  #All terms required
                            out={'eag':eag,'ebd':ebd,'ax':ax,'ay':ay,'az':az,\
                                 'bx':bx,'by':by,'bz':bz,'gz':gz,'dz':dz}
                        elif calc[1] or calc[3]: #Some terms required
                            out={'eag':eag,'ebd':ebd,'ax':ax,'ay':ay,'az':az,\
                                 'bz':bz,'gz':gz,'dz':dz}
                        else:   #Only z-terms required
                            out={'eag':eag,'ebd':ebd,'az':az,'bz':bz,'gz':gz,'dz':dz}
                        
                        yield out
    

"Swap indices, using appropriate symmetry relationships"
def m_mp_swap(X,mpi=0,mi=0,mpf=0,mf=0): 
    """
    Performs the appropriate sign changes to switch between components
    of correlation functions or their values at infinite time. One should provide
    the initial component indices (mi,mpi) and the final component indices (mf,mpf)
    
    Currently, one of the components must be 0 or both mi=mpi, mf=mpf
    """
    
    if X is None:
        return None
    
    if not((np.abs(mi)==np.abs(mf) and np.abs(mpi)==np.abs(mpf)) or (np.abs(mi)==np.abs(mpf) and np.abs(mpi)==np.abs(mf))):
        print('Invalid m values')
        print('Example: initial components (0,2) can have final components (0,-2),(2,0),(-2,0)')
        return
    
    if mi!=0 and mpi!=0 and mi!=mpi:
        print('m or mp must be 0, or m=mp')
        return
    
    if mi==mpi and mf==mpf:
        return X
    
    if np.abs(mi)!=np.abs(mf):      #Test for a position swap
        X=np.conj(X)
    if (mi+mpi)!=(mf+mpf):  #Test for a sign swap
        if np.abs(mi)==1 or np.abs(mf)==1:
            X=-np.conj(X)   #Sign change and conjugate
        elif np.abs(mi)==2 or np.abs(mf)==2:
            X=np.conj(X)    #Conjugate 
    return X
    

#%% Calculations in case of symmetry axis in motion
def sym_full_swap(vZ,threshold=0,A_0m_PASinf=None,vXZ=None,nuZ_F=None,nuXZ_F=None,
                  nuZ_f=None,nuXZ_f=None,cmpt='0p',mode='both',index=None,rank:int=2):
    """
    Swaps between calculating all components of the correlation function or
    assuming the correlation function is symmetric
    
    sym_full_swap(vZ,threshold=0,A_0m_PASinf=None,vXZ=None,nuZ_F=None,nuXZ_F=None,\
                  nuZ_f=None,nuXZ_f=None,cmpt='0p',mode='both',index=None)
    
    Setting the threshold to 0 will force a full calculation, and setting the 
    threshold to 1 will force a symmetric calculation. In case threshold is set
    to 0, A_0m_PASinf is not required.
    """
    A=None
    if A_0m_PASinf is None or threshold==0:
        ct,A=Ct_D2inf(vZ=vZ,vXZ=vXZ,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F,nuZ_f=nuZ_f,nuXZ_f=nuXZ_f,cmpt=cmpt,mode=mode,index=index)
    elif threshold==1:
        ct0=Ctsym(A_0m_PASinf,nuZ_f=nuZ_f,nuXZ_f=nuXZ_f,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F,index=index,rank=rank)
        ct=np.zeros([5,ct0.shape[0],ct0.shape[1]],dtype=complex)
        ct[2]=ct0
    else:
        sym=vft.Spher2pars(A_0m_PASinf)[1]<threshold   #Returns eta for the residual tensor
        nsym=np.logical_not(sym)
        sym_args={'A_0m_PASinf':A_0m_PASinf,'nuZ_f':nuZ_f,'nuXZ_f':nuXZ_f,'nuZ_F':nuZ_F,'nuXZ_F':nuXZ_F}
        full_args={'vZ':vZ,'vXZ':vXZ,'nuZ_f':nuZ_f,'nuXZ_f':nuXZ_f,'nuZ_F':nuZ_F,'nuXZ_F':nuXZ_F}
        
        for k,v in sym_args.items():
            sym_args[k]=None if v is None else v[:,sym]
        for k,v in full_args.items():
            full_args[k]=None if v is None else v[:,nsym]
        if np.any(sym):
            print('Using symmetric calculation for {0} correlation functions'.format(sym.sum()))
            out_sym=Ctsym(**sym_args,index=index)
        if np.any(nsym):
            print('Using full calculation for {0} correlation functions'.format(nsym.sum()))
            out_full=Ct_D2inf(**full_args,cmpt=cmpt,mode='ct',index=index)
        nt=out_sym.shape[-1] if np.any(sym) else out_full.shape[-1]
        ct=np.zeros([5,sym.shape[0],nt],dtype=complex)
        if np.any(sym):ct[2,sym]=out_sym
        if np.any(nsym):ct[:,nsym]=out_full

    return ct,A if mode.lower()=='both' else ct  
            

def sym_nuZ_f(A_0m_PASinf,nuZ_f,nuXZ_f=None,nuZ_F=None,nuXZ_F=None):
    """
    Assuming motion within frame f is symmetric, this function returns a 
    vector that moves with frame f, and points in the direction of the residual
    tensor resulting from motion within frame f. If nuZ_F (optionally nuXZ_F),
    then motion of frame f will have motion of frame F removed.
    
    Note that A_0m_PASinf is the residual tensor of motion within frame f (not 
    frame F)
    
    sym_nuZ_f(A_0m_PASinf,nuZ_f,nuXZ_f=None,nuZ_F=None,nuXZ_F=None)
    
    Note that the results from this calculation could, in principle, be used in 
    the iRED analysis
    """
    
    _,_,*euler=vft.Spher2pars(A_0m_PASinf)
    vZ=vft.R(np.array([0,0,1]),*euler)  #Direction of residual tensor in frame f
    nuZ_f,nuXZ_f=vft.applyFrame(nuZ_f,nuXZ_f,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F)
    sc=[sc0.T for sc0 in vft.getFrame(nuZ_f,nuXZ_f)]   
    out=vft.R(vZ,*sc)
    if out.ndim==3:
        return out.swapaxes(1,2)
    else:
        return out

def Ctsym(A_0m_PASinf,nuZ_f,nuXZ_f=None,nuZ_F=None,nuXZ_F=None,index=None,rank:int=2):
    """
    Calculates the correlation function of the motion of a frame, assuming that
    motion within that frame has a symmetry axis. This is achieved by providing
    the residual tensor of motion within frame f, obtained by calculating:
        
        A_PASinf=Ct_D2inf(vZ,vXZ,nuZ_F=nuZ_f,nuXZ_F=nuXZ_f,mode='d2')
    
    Note that we calculate the correlation function of motion due to motion of 
    frame f within frame F. Then, for the above call, the alignment frame is 
    defined by nuZ_f, but needs to be assigned to nuZ_F.
    
    This result can then be used to obtain the correlation function of motion for
    f in F. Note that this correlation function only has a (0,0) component, so
    the result is just a numpy array
    
    Ct=Ctsym(A_PASinf,nuZ_f=nuZ_f,nuXZ_f=nuXZ_f,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F,index=index)
    """
    
    nuZ_fsym=sym_nuZ_f(A_0m_PASinf=A_0m_PASinf,nuZ_f=nuZ_f,nuXZ_f=nuXZ_f,nuZ_F=nuZ_F,nuXZ_F=nuXZ_F)
    return Ct_D2inf(nuZ_fsym,cmpt='00',mode='ct',index=index,rank=rank)

def ini_vec_load(traj,frame_funs,tensor_fun,frame_index=None,index=None,info=None):
    """
    Loads vectors corresponding to each frame, defined in a list of frame functions.
    Each element of frame_funs should be a function, which returns one or two
    vectors.
    
    traj should be the trajectory from MDanalysis

    frame_index is a list of indices, allowing one to match each bond to the correct
    frame (length of each list matches number of bonds, values in list should be between
    zero and the number of frames-1)

    index (optional) is used for sparse sampling of the trajectory
    
    dt gives the step size of the MD trajectory (use to override dt found in traj)
    """
    
    if hasattr(frame_funs,'__call__'):frame_funs=[frame_funs]  #In case only one frame defined (unusual usage)

    nf=len(frame_funs)
    nt=len(traj)
    
    if index is None: index=np.arange(nt)
    dt=traj.dt/1e3
    
    t=index*dt
    v=[list() for _ in range(nf)]
    vT=list()
    """v is a list of lists. The outer list runs over the number of frames (length of frame_funs)
    The inner list runs over the timesteps of the trajectory (that is, the timesteps in index)
    The inner list contains the results of executing the frame function (outer list) at that
    time point (inner list)
    """
      
    for c,i in enumerate(index):
        traj[i] #Go to current frame
        for k,f in enumerate(frame_funs):
            v[k].append(f())

        vT.append(tensor_fun())
        "Print the progress"
        try:
            if c%int(len(index)/100)==0 or c+1==len(index):
                ProgressBar(c+1, len(index), prefix = 'Loading Ref. Frames:', suffix = 'Complete', length = 50) 
        except:
            pass

    for k,v0 in enumerate(v):
        v[k]=np.array(v0)
        """Put the vectors in order such that if two vectors given, each vector
        is an element of the first dimension, and the second dimension is X,Y,Z
        (If only one vector, X,Y,Z is the first dimension)
        """
        v[k]=np.moveaxis(v[k],0,-1)

        
    vT=np.moveaxis(vT,0,-1)
    
    return {'n_frames':nf,'v':v,'vT':vT,'t':t,'index':index,'frame_index':frame_index,'info':info} 
