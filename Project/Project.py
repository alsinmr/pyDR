#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 14:50:19 2022

@author: albertsmith
"""

import os
import numpy as np
from pyDR.IO import read_file, readNMR, isbinary, write_PDB
from pyDR import Defaults
from pyDR import clsDict
import re
from copy import copy
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import gc
from shutil import copyfile
decode=bytes.decode


class DataMngr():
    def __init__(self, project):
        self.project = project

        self._saved_files = list()
        
        if self.directory:
            if not(os.path.exists(self.directory)):
                os.mkdir(self.directory)
            for fname in os.listdir(self.directory):
                if '.data' in fname:
                    self._saved_files.append(fname)
            self._saved_files.sort()
        self.data_objs = [None for _ in self.saved_files]
        self._hashes = [None for _ in self.saved_files]

    @property
    def directory(self):
        if self.project.directory is None:return
        return os.path.join(self.project.directory, 'data')
        
    @property
    def saved_files(self):
        return self._saved_files
                
    def load_data(self,filename=None,index=None):
        if self.directory is None:return
        "Loads a saved file from the project directory"
        if filename is not None:
            assert filename in self.saved_files,"{} not found in project. Use 'append_data' for new data".format(filename)
            index=self.saved_files.index(filename)
        fullpath=os.path.join(self.directory,self.saved_files[index])
        self.data_objs[index]=read_file(fullpath,directory=self.project.directory)
        self._hashes[index]=self.data_objs[index]._hash
        self.data_objs[index].source.project=self.project
        if self.data_objs[index].source.select is not None:
            self.data_objs[index].source.select.molsys.project=self.project
            self.data_objs[index].source.select.project=self.project
        
        
    def append_data(self,data,incl_source=True):
        filename=None
        if isinstance(data,str):
            filename,data=data,None
        "Adds data to the project (either from file, set 'filename' or loaded data, set 'data')"
        if filename is not None:
            if not(os.path.exists(filename)):
                if self.directory is not None and \
                    os.path.exists(os.path.join(os.path.dirname(self.directory),filename)):  #Check in the project directory
                    filename=os.path.join(os.path.dirname(self.directory),filename) 
                    """Note that adding data from within the project/data directory is discouraged,
                    so we do not check that directory here.
                    """
            if not(os.path.exists(filename)):          
                try:
                    data=readNMR(filename)
                except:
                    pass
            else:                    
                assert os.path.exists(filename),"{} does not exist".format(filename)
                data=read_file(filename,directory=self.project.directory) if isbinary(filename) else readNMR(filename)

        if data in self.data_objs:
            print("Data already in project (index={})".format(self.data_objs.index(data)))
            data.project=self.project
            #TODO is the above line really a good idea?
            return
                
        self.data_objs.append(data)
        
        self._hashes.append(None)   #We only add the hash value if data is saved
        self._saved_files.append(None)
        
        self.data_objs[-1].source.project=self.project
        
        if self.data_objs[-1].source.select is not None:
            self.data_objs[-1].source.select.molsys.project=self.project
            self.data_objs[-1].source.select.project=self.project

        if self.project is not None:
            flds=['Type','status','short_file','title','additional_info']
            dct={f:getattr(self.data_objs[-1].source,f) for f in flds}
            dct['filename']=None
            self.project.pinfo.new_exper(**dct)
            self.project._index=np.append(self.project._index,len(self.data_objs)-1)

        if data.src_data is not None:
            if data.src_data in self.data_objs:
                data.src_data=self[self.data_objs.index(data.src_data)]
            elif incl_source:
                self.append_data(data=data.src_data) #Recursively append source data
    
    def remove_data(self,index,delete=False):
        """
        Remove a data object from project. Set delete to True to also delete 
        saved data in the project folder. If delete is not set to True, saved 
        data will re-appear upon the next project load even if removed here.
        """
            
        if not(hasattr(index,'__len__')):
            self.data_objs.pop(index)
            self._hashes.pop(index)
            if delete and self.saved_files[index] is not None:
                os.remove(os.path.join(self.directory,self.saved_files[index]))
            self._saved_files.pop(index)
            self.project.pinfo.del_exp(index)
        else:
            for i in np.sort(index)[::-1]:self.remove_data(i,delete=delete)
                
            
    
    def __getitem__(self,i):
        if hasattr(i,'__len__'):
            return [self[i0] for i0 in i]
        else:
            assert i<len(self.data_objs),"Index exceeds number of data objects in this project"
            if self.data_objs[i] is None:
                self.load_data(index=i)
            return self.data_objs[i]
            
    
    
    def __len__(self):
        return len(self.data_objs)
    
    def __iter__(self):
        def gen():
            for k in range(len(self)):
                yield self[k]
        return gen()
        
    @property
    def titles(self):
        return [d.title for d in self]
    
    @property
    def filenames(self):
        """
        List of filenames for previously saved data
        """
        return [d.source.saved_filename for d in self]
    
    @property
    def short_files(self):
        return [d.source.short_file for d in self]
    
    @property
    def save_name(self):
        """
        List of filenames used for saving data
        """
        names=list()
        for k,d in enumerate(self.data_objs):
            if d is None:
                name=self.saved_files[k]    #saved files can be shorter than d,
                                            #but I think d cannot be None for the latter values
            else:
                name=os.path.split(d.source.default_save_location)[1]
            if name in names:
                name0=name[:-5]+'{}'+name[-5:]
                k=1
                while name in names:
                    name=name0.format(k)
                    k+=1
            names.append(name)
        names=[os.path.join(self.directory,name) for name in names]
        return names
        
    @property
    def sens_list(self):
        return [d.sens for d in self]
    
    @property
    def detect_list(self):
        return [d.detect for d in self]
    
    @property
    def saved(self):
        "Logical index "
        return [True if d is None else h==d._hash for h,d in zip(self._hashes,self.data_objs)]
    
    def save(self,i='all',include_rawMD=False):
        """
        Save data object stored in the project by index, or set i to 'all' to
        save all data objects. Default is to derive the filename from the title.
        To save to a specific file, use data.save(filename='desired_name') instead
        of saving from the project.
        """
        ME=Defaults['max_elements']
        if i=='all':
            for i in range(len(self)):
                if not(self.saved[i]):
                    # if self[i].R.size>ME:
                    #     print('Skipping data object {0}. Size of data.R ({1}) exceeds default max elements ({2})'.format(i,self[i].R.size,ME))
                    #     continue
                    if self[i].source.status=='raw' and str(self[i].sens.__class__).split('.')[-1][:-2]=='MD'\
                        and not(include_rawMD):
                        print('Skipping data object "{0}".\n Set include_rawMD to True to include raw MD data'.format(self[i].title))
                        continue
                            
                    self.save(i)
        else:
            assert i<len(self),"Index {0} to large for project with {1} data objects".format(i,len(self))
            if not(self.saved[i]):
                
                src_data=self[i].source._src_data
                if src_data is None:
                    src_fname=None
                elif src_data is not None and isinstance(src_data,str):
                    # Is the data already stored in the project?
                    if os.path.join(self.directory,src_data) in [os.path.join(self.directory,file) for file in self.saved_files]:
                        src_fname=os.path.join(self.directory,src_data)
                    else:
                        src_fname=os.path.abspath(src_data)
                else:
                    if src_data in self.data_objs:
                        file=self.saved_files[self.data_objs.index(src_data)]
                        if file is None:
                            src_fname=None
                        else:
                            src_fname=os.path.join(self.directory,file)
                    else:
                        src_fname=None
                        
                        
                    
                    
                # src_fname=None
                # if self[i].source._src_data is not None and not(isinstance(self[i].source._src_data,str)):
                #     # k=np.argwhere([self[i].src_data==d for d in self.data_objs])[0,0]
                #     k=self.data_objs.index(self[i].src_data) if self[i].src_data in self.data_objs else None
                    
                #     # if self[k].R.size<=ME:
                #     #     self.save(k)
                #     if k is not None:
                #         #Next 6 lines commented on 8 December 2023 (why did we do this?)
                #         # if not(self[k].source.status=='raw' and str(self[k].sens.__class__).split('.')[-1][:-2]=='MD')\
                #         #     or include_rawMD:
                #         #     self.save(k)
                #         # else:
                #         #     # print('Skipping source data of object {0} (project index {1}). Size of source data exceeds default max elements ({2})'.format(i,k,ME))
                #         #     print('Skipping source data of object "{0}".\n Set include_rawMD to True to include raw MD data'.format(self[i].title))
                #         src_fname=self.save_name[k]
                #     elif self[i].source._src_data is not None and self[i].source._src_data in self.saved_files:
                #         src_fname=self[i].source._src_data
                self[i].save(self.save_name[i],overwrite=True,save_src=False,src_fname=src_fname)
                self[i].source.saved_filename=self.save_name[i]
                self._hashes[i]=self[i]._hash #Update the hash so we know this state of the data is saved
                self._saved_files[i]=self.save_name[i]
                self.project.pinfo['filename',np.argwhere(self.project._index==i)[0,0]]=os.path.split(self.save_name[i])[1]
          

class DataSub(DataMngr):
    """
    Data storage object used for creating a sub-project of the given project.
    Removes various functionality of the main class.
    """
    def __init__(self, project, *data):
        assert not project.subproject,'Sub-project data objects should reference the parent project'
        self.project = project #This should be the parent project, not the subproject
        self.data_objs = data
        self._hashes = None
    "De-activated functions/properties below"
    @property
    def saved_files(self):
        pass
    def load_data(self, *args, **kwargs):
        pass
    def append_data(self, *args, **kwargs):
        pass
    def remove_data(self, *args, **kwargs):
        pass
    @property
    def saved(self):
        pass
    def save(self,*args,**kwargs):
        pass
    
    
    

            

#%% Detector Manager
class DetectMngr():
    def __init__(self,project):
        self.project=project

    def __iter__(self):
        r=self.detectors
        def gen():
            for r0 in r:
                yield r0
        return gen()
    
    def plot_rhoz(self,index=None,ax=None,norm=False,**kwargs):
        """
        If only one unique detector in the corresponding project, then this
        runs plot_rhoz from that detector

        Parameters
        ----------
        index : TYPE, optional
            DESCRIPTION. The default is None.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        norm : TYPE, optional
            DESCRIPTION. The default is False.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        self.unify_detect()
        ax=plt.subplots()[1]
        for d in self.detectors:
            d.plot_rhoz(index=index,ax=ax,norm=norm,**kwargs)
        return ax

    @property
    def detectors(self):
        r=list()
        for d in self.project:
            if not(any([r0 is d.detect for r0 in r])):
                r.append(d.detect)
        return r
        
    def unify_detect(self, chk_sens_only: bool = False) ->None:
        """
        Checks for equality among the detector objects and assigns all equal
        detectors to the same object. This allows one to only optimize one of
        the data object's detectors, and all other objects with equal detectors
        will automatically match (also saves memory)
        
        Note that by default, if two detector objects based on matching 
        sensitivity 

        Parameters
        ----------
        chk_sens_only : bool, optional
            If two detector objects have matching input sensitivities, but have
            been optimized differently, they will not be considered equal, unless
            this flag is set to True. The default is False.

        Returns
        -------
        None

        """
        proj=self.project
        r=[d.detect for d in proj]
        s=[r0.sens for r0 in r]
        
        for k,(s0,r0) in enumerate(zip(s,r)):
            i=None
            if chk_sens_only:
                if s0 in s[:k]:
                    i=s[:k].index(s0)
            else:
                if r0 in r[:k]:
                    i=r[:k].index(r0)
            if i is not None:
                proj[k].sens=s[i]
                proj[k].detect=r[i]
                
    def unique_detect(self, index: int = None) -> None:
        for k,i in enumerate(self.project._index):
            if self.project.data.data_objs[i] is not None:
                d=self.project[k]
                sens=d.sens
                d.detect=d.detect.copy()
                d.detect.sens=sens
        
        
        # project=self.project
        # if index is None:
        #     for i in range(len(project.data)):
        #         self.unique_detect(i)
        # else:
        #     d = project.data[index]
        #     sens = d.detect.sens
        #     d.detect = d.detect.copy()
        #     d.detect.sens = sens
    
    def r_auto(self,n:int,Normalization:str='MP',NegAllow:bool=False) -> None:
        """
        Finds the n optimal detector sensitivites for all data objects in the
        current subproject.

        Parameters
        ----------
        n : int
            Number of detector sensitivities to optimize.
        Normalization : str, optional
            Type of normalization used for optimization.
            'MP': Equal maxima, where if S2 is included, rho0 is positive
            'M': Equal maxima, where the sum of detector sensitivities is 1
                 -'M' and 'MP' are the same if S2 not used
            'I': Equal integrals
            The default is 'MP'.
        NegAllow : bool, optional
            Allow detector sensitivities to oscillate below zero. 
            The default is False.

        Returns
        -------
        None

        """
        self.unify_detect(chk_sens_only=True)
        for r in self:r.r_auto(n,Normalization=Normalization,NegAllow=NegAllow)
        
        return self
        
    def r_target(self,target:np.ndarray,n:int=None) -> None:
        """
        Optimizes detector sensitivities to some target function for all data
        objects in the current subproject.

        Parameters
        ----------
        target : np.ndarray
            numpy array with NxM elements. N is the number of detector sensitivities
            to match, and M is the number of correlation times stored in the
            sensitivity objects (default is 200)
        n : int, optional
            Number of singular values to use to match to the target function.
            By default, n gets set to N (see above), but often the reproduction
            of experimental sensitivities requires a larger value
            The default is None.

        Returns
        -------
        None

        """
        self.unify_detect(chk_sens_only=True)
        for r in self:r.r_target(target,n=n)
        
        return self
    
    def inclS2(self):
        """
        Include S2 in analysis (NMR experiments only!)

        Returns
        -------
        None.

        """
        for r in self:r.inclS2()
        
        return self
    
    def r_no_opt(self,n:int) -> None:
        """
        Optimize n detectors for pre-processing, i.e. with the no_opt setting

        Parameters
        ----------
        n : int
            Number of detectors to use.

        Returns
        -------
        None

        """
        self.unify_detect(chk_sens_only=True)
        for r in self:r.r_no_opt(n)
        
        return self
    
    def r_zmax(self,zmax,Normalization:str='MP',NegAllow:bool=False) -> None:
        """
        Optimize n detectors with maxima located at the positions given in zmax.

        Parameters
        ----------
        zmax : list,np.ndarray
            List of the target maxima for n detectors (n is the length of zmax).
        Normalization : str, optional
            Type of normalization used for optimization.
            'MP': Equal maxima, where if S2 is included, rho0 is positive
            'M': Equal maxima, where the sum of detector sensitivities is 1
                 -'M' and 'MP' are the same if S2 not used
            'I': Equal integrals
            The default is 'MP'.
        NegAllow : bool, optional
            Allow detector sensitivities to oscillate below zero. 
            The default is False.

        Returns
        -------
        None

        """
        self.unify_detect(chk_sens_only=True)
        for r in self:r.r_zmax(zmax,Normalization=Normalization,NegAllow=NegAllow)
        
        return self


#%% ChimeraX Manager
class Chimera():
    def __init__(self,project):
        self.project=project
        self._current=[-1]
        self._CMXids=[]
        self.CMX=clsDict['CMXRemote']
        self.saved_commands=[] #Automatically run these commands when chimera is initialized.
    
    @property
    def CMXid(self):
        if self.current is not None:return self._CMXids[self.current]
    
    @property
    def current(self):
        if self._current[0]==-1:return None
        if not(len(self._CMXids)):return None
        if not(self.CMX.isConnected(self._CMXids[self._current[0]])):
            while len(self._CMXids):
                if self.CMX.isConnected(self._CMXids[-1]):
                    print('Session {0} was not connected. Reseting to session {1}'\
                          .format(self._current[0],len(self._CMXids)-1))
                    self._current[0]=len(self._CMXids)-1
                    return self._current[0]
                else:
                    self._CMXids.pop()
            print('No sessions were connected')
            self._current[0]=-1
            return None
        return self._current[0]
    
    def __setattr__(self,name,value):
        if name=='current': #Set the current chimeraX connection, and assert that there is an active connection
            assert isinstance(value,(int,np.integer)) and value>=0,"ID must be a non-negative integer"
            while len(self._CMXids)<=value:
                self._CMXids.append(None)
            if self._CMXids[value] is None or not(self.CMX.isConnected(self._CMXids[value])):
                self._CMXids[value]=self.CMX.launch()
                if self.CMX.isConnected(self._CMXids[value]):
                    self._current[0]=value
            self._current[0]=value
            return
        super().__setattr__(name,value)
    
    def __call__(self,index=None,rho_index=None,scaling=None,offset=None):
        if scaling is None:
            m=0
            for d in self.project:
                i=np.arange(d.R.shape[0]) if index is None else index
                r=np.arange(d.R.shape[1]) if rho_index is None else rho_index
                m=max((d.R[i][:,r].max(),m))
            scaling=1/m
        if self.current is None:self.current=0
        
        
        #A bunch of stuff to try to guess which atoms to align
        res0=[np.min(d.select.uni.residues.resids) for d in self.project]
        ress=[np.max([res0[0],r]) for r in res0]
        i0=[np.argwhere(r==d.select.uni.residues.resids)[0,0] for d,r in zip(self.project,ress)]
        resl=[len(d.select.uni.residues[i:]) for i,d in zip(i0,self.project)]
        resl=[np.min([resl[0],r]) for r in resl]
        resf=[(self.project[0].select.uni.residues.resids[i0[0]+l-1],
               d.select.uni.residues.resids[i+l-1]) for d,i,l in zip(self.project,i0,resl)]

        for k,d in enumerate(self.project):
            if offset is None:
                offset=np.std(d.select.pos,0)*6
                # offset[offset!=offset.min()]=0
                ax=['x','y','z'].pop(np.argmin(offset))
                offset=offset.min()
            d.chimera(index=index,rho_index=rho_index,scaling=scaling)
            
            if not(k):
                mn=self.CMX.valid_models(self.CMXid)[-1]
            else:
                mdl_num=self.CMX.valid_models(self.CMXid)[-1]
                cmds='align #{3}:{4}-{5}@CA toAtoms #{0}:{1}-{2}@CA cutoffDistance 5'.format(\
                                    mn,ress[k],resf[k][0],mdl_num,ress[k],resf[k][1])
                self.command_line(cmds)
            
                
            # self.CMX.conn[self.CMXid].send(('shift_position',-1,offset*k))
                self.command_line('move {0} {1} models #{2} coordinateSystem #{3}'.format(\
                                ax,offset*k,mdl_num,mn))
        self.command_line('view')
        
    def CCchimera(self,index=None,rho_index=None,indexCC:int=None,
                  scaling:float=1,norm:bool=True,offset=None):
        
        """
        Plots the cross correlation of motion for a given detector window in 
        chimera. 

        Parameters
        ----------
        index : list-like, optional
            Select which residues to plot. The default is None.
        rho_index : int, optional
            Select which detector to initially show. The default is None.
        indexCC : int,optional
            Select which row of the CC matrix to show. Must be used in combination
            with rho_index. Note that if index is also used, indexCC is applied
            AFTER index.
        scaling : float, optional
            Scale the display size of the detectors. If not provided, a scaling
            will be automatically selected based on the size of the detectors.
            The default is None.
        norm : bool, optional
            Normalizes the data to the amplitude of the corresponding detector
            responses (makes diagonal of CC matrix equal to 1).
            The default is True

        Returns
        -------
        None

        """
        
        if self.current is None:self.current=0
        
        
        #A bunch of stuff to try to guess which atoms to align
        res0=[np.min(d.select.uni.residues.resids) for d in self.project]
        ress=[np.max([res0[0],r]) for r in res0]
        i0=[np.argwhere(r==d.select.uni.residues.resids)[0,0] for d,r in zip(self.project,ress)]
        resl=[len(d.select.uni.residues[i:]) for i,d in zip(i0,self.project)]
        resl=[np.min([resl[0],r]) for r in resl]
        resf=[(self.project[0].select.uni.residues.resids[i0[0]+l-1],
               d.select.uni.residues.resids[i+l-1]) for d,i,l in zip(self.project,i0,resl)]

        for k,d in enumerate(self.project):
            if not(hasattr(d,'CC')) or d.CC is None:continue
            if offset is None:
                offset=np.std(d.select.pos,0)*6
                # offset[offset!=offset.min()]=0
                ax=['x','y','z'].pop(np.argmin(offset))
                offset=offset.min()
            d.CCchimera(index=index,rho_index=rho_index,indexCC=indexCC,scaling=scaling,
                        norm=norm)
            
            if not(k):
                mn=self.CMX.valid_models(self.CMXid)[-1]
            else:
                mdl_num=self.CMX.valid_models(self.CMXid)[-1]
                cmds='align #{3}:{4}-{5}@CA toAtoms #{0}:{1}-{2}@CA cutoffDistance 5'.format(\
                                    mn,ress[k],resf[k][0],mdl_num,ress[k],resf[k][1])
                self.command_line(cmds)
            
                
            # self.CMX.conn[self.CMXid].send(('shift_position',-1,offset*k))
                self.command_line('move {0} {1} models #{2} coordinateSystem #{3}'.format(\
                                ax,offset*k,mdl_num,mn))
        self.command_line('view')
    
    
    
    def command_line(self,cmds:list=None,ID:int=None) -> None:
        """
        Send commands to chimeraX as a single string, as a list, or interactively
        (provide cmds as a string, as a list of strings, or None, respectively)

        Parameters
        ----------
        cmds : list, optional
            List of commands to send to the specified chimera instance. 
            The default is None, in which case an interactive command line will
            be opened. May optionally be provided a single string
        ID : int, optional
            Specify which chimera session to interact with. The default is None,
            which sends to the active session.

        Returns
        -------
        None.

        """
        CMXid=self._CMXids[ID] if ID is not None else self.CMXid
        if CMXid is None:
            print("No active sessions, exiting command_line")
            return
        
        if cmds is not None:
            if isinstance(cmds,str):cmds=[cmds]
            for cmd in cmds:
                self.CMX.send_command(CMXid,cmd)
        else:
            print('Type "exit" to return to python')
            cmd=''
            while True:  
                cmd=input('ChimeraX>')
                if cmd.strip()=='exit':break
                self.CMX.send_command(CMXid,cmd)
                
    def close(self,ID:int=None) -> None:
        """
        Closes all models in the current or specified chimeraX session.

        Parameters
        ----------
        ID : int, optional
            Specifies which session in which to close the models.
            The default is None, which closes in the current chimeraX session.

        Returns
        -------
        None

        """
        self.command_line(cmds='close',ID=ID)
                
    def savefig(self,filename:str,options:str='',ID:int=None,overwrite:bool=False)-> None:
        """
        Saves the current chimera window to a figure. If a relative path is
        provided, this will save within the project figure folder. One may
        pass options to the save in chimera via a string. 

        Parameters
        ----------
        filename : str
            Save location. Relative locations will be stored in the project 
            figure folder
        options : str, optional
            String contain options for saving in chimeraX. Multiple options can
            be included in the string, as if they were being entered withing
            chimera. The default is None.
        ID : int, optional
            Specify which chimeraX session to save. The default is None, which 
            saves the active session.
        overwrite : bool, optional
            Overwrite existing figures. The default is False.

        Returns
        -------
        None
            DESCRIPTION.

        """
        CMXid=self._CMXid[ID] if ID is not None else self.CMXid
        assert CMXid is not None,"No active sessions, save not completed"

        if self.project is not None:   
            if self.project.directory is not None:
                if not(os.path.exists(os.path.join(self.project.directory,'figures'))):
                    os.mkdir(os.path.join(self.project.directory,'figures'))
                
                filename=os.path.join(os.path.join(self.project.directory,'figures'),filename)
        if len(filename.split('.'))<2:filename+='.png'
        
        if not(overwrite):
            assert not(os.path.exists(filename)),'File already exists. Set overwrite=True'
            
        
        self.command_line('save "{0}" {1}'.format(filename,options))
        
    def draw_tensors(self,A,pos=None,colors=((1,.39,.39,1),(.39,.39,1,1)),Aiso=None,comp='Azz'):
        if self.current is None:self.current=0
        self.CMX.run_function(self.CMXid,'draw_tensors',A,Aiso,pos,colors,comp)
        
    @property
    def valid_models(self):
        return self.CMX.valid_models(self.CMXid)


#%% Numpy nice display

        
class nparray_nice(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def _ipython_display_(self):
        for x in self:print(x)
        
    def __repr__(self):
        out=''
        for x in self:out+=x+'\n'
        return out 
    
def mk_nparray_nice(x):
    if not(hasattr(x,'__len__')):x=[x]
    out=nparray_nice(shape=[len(x)],dtype=object)
    for k,x0 in enumerate(x):
        out[k]=x0
    return out

#%% Project class
class Project():
    """
    Project class of pyDIFRATE, used for organizing data and plots
    """
    def __init__(self, directory:str=None, create:bool = False, subproject:bool = False) -> None:
        """
        Initialize a pyDIFRATE project, for organizing data and plots

        Parameters
        ----------
        directory : str
            Path to an existing or new project.
        create : bool, optional
            Create a new directory on the specified path if one does not exist.
            The default is False.
        subproject : bool, optional
            Set to True if this project is a subproject of another project. 
            Only intended for internal use. The default is False.

        Returns
        -------
        None

        """
        super().__setattr__('_index',np.zeros(0,dtype=int))
        self._index=np.zeros(0,dtype=int)
        self._index0=None
        self.name = directory   #todo maybe create the name otherwise?
        self._directory = os.path.abspath(directory) if directory is not None else None
        if self.directory and not(os.path.exists(self.directory)) and create:
            os.mkdir(self.directory)
            os.mkdir(os.path.join(self.directory,'scratch'))
        # assert os.path.exists(self.directory),'Project directory does not exist. Select an existing directory or set create=True'
               
        self.data=DataMngr(self)
        self._subproject = subproject  #Subprojects cannot be edited/saved
        self.plots = [None]
        self._current_plot = [0]
        
        self.read_proj()
        self.chimera=Chimera(self)
    
    @property
    def directory(self):
        return self._directory
    
    @property
    def parent(self):
        if self._subproject:return self._parent
        return self
    #%% setattr        
    def __setattr__(self,name,value):
        # if name=='_index':  #Why does this need special treatement?
        #     super().__setattr__(name,value)
            
        "Could we do something for setting the project directory?"
        if name=='directory':
            assert self.directory is None,'Project directory cannot be changed'
            assert isinstance(value,str),'Project directory must be a string'
            assert not(os.path.exists(value)),'Project directory already exists'
            os.mkdir(value)
            os.mkdir(os.path.join(value,'data'))
            super().__setattr__('_directory',value)
            return

        if len(self)==1 and name!='chimera' and not(hasattr(self.__class__,name)) and hasattr(self[0],name):
            #Only one data object in project, and the data object has attribute name, but the project does not
            setattr(self[0],name,value) #Set the attribute for the data object, not the project
            return
        
        "Special behavior for current plot"
        if name=='current_plot':  
            self._current_plot[0]=value
            if value:
                while len(self.plots)<value:
                    self.plots.append(None)
                if self.plots[value-1] is None:
                    self.plots[value-1]=clsDict['DataPlots']()
                    self.plots[value-1].project=self
            return
        super().__setattr__(name,value)
        
    def __getattr__(self,name):
        """
        This allows us to access properties of the data in case a project selection
        yields just a single data object.

        Parameters
        ----------
        name : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        
        # Some consoles sometimes query objects for various attributes. This
        # triggers __getattr__ if the attribute doesn't exist in project, and
        # therefore can furthermore cause an unnecessary data load which slows
        # things down considerably. We collect these attributes here and
        # thereby prevent the unintended data load
        blacklist=['shape','_repr_mimebundle_','_ipython_canary_method_should_not_exist_']
        if len(self)==1 and name not in blacklist and hasattr(self[0],name):
            return getattr(self[0],name)
        raise AttributeError(f"'Proj' object has no attribute '{name}'")
        
    #%% Detector manager
    @property
    def detect(self):
        return DetectMngr(self)
    
    #%% Chimera Cross-correlation
    def CCchimera(self,index=None,rho_index=None,indexCC:int=None,
                  scaling:float=1,norm:bool=True,offset=None):
        self.chimera.CCchimera(index=index,rho_index=rho_index,indexCC=indexCC,
                               scaling=scaling,norm=norm,offset=offset)
    
    #%% Clear memory
    def clear_memory(self,include_rawMD=False):
        """
        Moves the project from memory onto the drive, thus freeing up memory
        that is taken up by data objects currently stored in memory. Note that 
        this operation will remove any raw MD data permanently from the project.
        One may set include_rawMD to True to save this data to file

        Parameters
        ----------
        include_rawMD : bool, optional
            Saves raw MD data in the project. The default is False.

        Returns
        -------
        None.

        """
        assert self.directory is not None,"clear_memory can only run if the project has a directory for saving"
        self.save(include_rawMD)
        # self=Project(self.directory)
        del_ind=list()
        del_file=list()
        for k,i in enumerate(self._index):
            if self.pinfo[k]['status']=='raw' \
                and str(self.data[i].sens.__class__).split('.')[-1][:-2]=='MD' \
                and self.data.filenames[i] is None: #Raw and loaded and MD data and not saved
                del_ind.append(k)  #These get deleted during clear_memory
        
        self.remove_data(del_ind,delete=True)
        
        # for i in self._index:
        #     if self.data.data_objs[i] is not None \
        #         and not(os.path.exists(self.data[i].source._src_data)):
        #             self.data[i].src_data=None
        
        self.data=DataMngr(self)
        gc.collect() #Garbage collect (get rid of the old data object)
        
        
        
    #%% Read/write project file
    def read_proj(self):
        info=clsDict['Info']()
        flds=['Type','status','short_file','title','additional_info','filename']
        for f in flds:info.new_parameter(f)
        
        if self.directory and os.path.exists(os.path.join(self.directory,'project.txt')):
            dct={}
            with open(os.path.join(self.directory,'project.txt'),'r') as f:
                for line in f:
                    if line.strip()=='DATA':    #Start reading a new data entry
                        if len(dct):
                            info.new_exper(**dct)    #Store the previous entry if exists
                        dct={}  #Reset entry
                        for l in f:     #Sweep until the end of this entry
                            if l.strip()=='END:DATA':break   #End of entry reached
                            if len(l.split(':\t'))==2:
                                k,v=[l0.strip() for l0 in l.strip().split(':\t')]
                                if k in flds:dct[k]=v #Add this field to dct
                if len(dct):info.new_exper(**dct)    #Store the current entry
                
        for k,file in enumerate(self.data.saved_files):
            if file not in info['filename']:   #Also include data that might be missing from the project file
                # print(file)    
                src=self.data[k].source
                dct={f:getattr(src,f) for f in flds}
                dct['filename']=file
                info.new_exper(**dct)
        
        _index=list()
        for file in info['filename']:
            if file in self.data.saved_files:
                _index.append(self.data.saved_files.index(file))
            else:
                _index.append(None)
                print('File:\n{0}\n was missing from project'.format(file))
        
        while None in _index: #Delete all missing data
            i=_index.index(None)
            _index.pop(i)

        self.pinfo=clsDict['Info']()
        for f in flds:self.pinfo.new_parameter(f)
        
        for k in range(len(_index)):
            self.pinfo.new_exper(**info[_index.index(k)])
        self._index=np.array(_index,dtype=int)
                
    def write_proj(self):
        self.update_info()
        with open(os.path.join(self.directory,'project.txt'),'w') as f:
            for i in self._index:
                if self.data.saved[i]:
                    f.write('DATA\n')
                    for k,v in self.pinfo[i].items():f.write('{0}:\t{1}\n'.format(k,v))
                    f.write('END:DATA\n')
                
    def update_info(self)->None:
        """
        Changes to identifying information in source (Type, status, short_file, 
        etc.) will not be automatically reflected in a project's indexing. 
        Run update_info on the project (or subproject) where changes have been
        made to obtain the updated parameters

        Returns
        -------
        None.

        """
        for i in self._index:
            if self.data.data_objs[i] is not None:
                for k in self.pinfo.keys:
                    if k=='filename':
                        if self.data.directory is not None:
                            self.pinfo[k,i]=os.path.split(self.data.save_name[i])[1]
                        else:
                            self.pinfo[k,i]=None
                    else:
                        self.pinfo[k,i]=getattr(self.data.data_objs[i].source,k)
            else:
                if self.data.directory is not None:
                    self.pinfo['filename',i]=os.path.split(self.data.save_name[i])[1]
            

    #%%Indexing/subprojects

            
    @property
    def subproject(self):
        return self._subproject
    
    def append_data(self,data,incl_source=True):
        assert not(self._subproject),"Data cannot be appended to subprojects"
        self.data.append_data(data,incl_source=incl_source)
        
    def remove_data(self,index,delete=False):
        #TODO Some problems may arise if data is removed but not deleted, and the project is subsquently saved
        # assert not(self._subproject),"Data cannot be removed from subprojects"
        
        if not(hasattr(index,'__len__')):index=[index]
        index=np.array(index,dtype=int)
        assert np.all(index<len(self)),'index must be less than len(proj)'
        
        index=np.mod(index,len(self))
        if self._subproject:
            i=[self.parent_index[i] for i in index]
            self._parent.remove_data(i,delete=delete)
            index=np.sort([self._index[i] for i in index])[::-1] #Convert to data index
            for i in index:
                self._index=self._index[self._index!=i]
                self._index[self._index>i]-=1
        else:
            index=np.sort([self._index[i] for i in index])[::-1] #Convert to data index
            
            # proj=self[index]
            
            # if hasattr(proj,'R'):#Single index given, thus self[index] returns a data object
            #     index=[self.data.data_objs.index(proj)]
            # else:
            #     index=np.sort(proj._index)[::-1]
            
            # if delete and len(index)>1:
            #     print('Delete data sets permanently by full title or index (no multi-delete of saved data allowed)')
            #     return
            
            for i in index:
                self._index=self._index[self._index!=i]
                self._index[self._index>i]-=1
                self.data.remove_data(index=i,delete=delete)

            if delete: #We need to save– otherwise the project file will be corrupted if the user doesn't do this
                self.save()

    def __iter__(self):
        def gen():
            for k in self._index:
                yield self.data[k]
        return gen()
    
    def __len__(self) -> int:
        return self._index.size
        # return self._index.size if hasattr(self,'_index') else 0

    @property
    def size(self) -> int:
        return self.__len__()
    
    def __copy__(self):
        cls = self.__class__
        result = cls.__new__(cls)
        result.__dict__.update(self.__dict__)
        return result
    
    def __getitem__(self, index: int):
        """
        Extract a data object or objects by index or title (returns one item) or
        by Type or status (returns a list).
        """
        if isinstance(index, int) or (hasattr(index,'ndim') and index.ndim==0): #Just return the data object
            assert index < self.__len__(), "index too large for project of length {}".format(self.__len__())
            return self.data[self._index[index]]
        
        
        if len(self)==0:
            proj=copy(self)
            proj._parent=self.parent
            proj._subproject=True
            return proj
        proj=copy(self)
        proj._subproject=True
        proj._parent=self.parent
        proj._current_plot=self._current_plot #This line and the next should let us control plots from subproject
        proj.plots=self.plots
        proj.chimera=copy(self.chimera)
        proj.chimera.project=proj
        if isinstance(index,str):
            flds=['Types','statuses','additional_info','titles','short_files']
            for f in flds:
                if index in getattr(self,f):
                    proj._index=self._index[getattr(self,f)==index]
                    return proj
                if f=='short_files':
                    "Also check for match without file ending"
                    values=np.array([v.rsplit('.',1)[0] for v in getattr(self,f)],dtype=object)
                    if index in values:
                        proj._index=self._index[values==index]
                        return proj
               
            r = re.compile(index)
            i=list()
            for t in self.titles:
                i.append(True if r.match(t) else False)
                
            proj._index=self._index[np.array(i)]
        elif hasattr(index,'__len__'):
            proj._index=self._index[index]
        elif isinstance(index,slice):
            start = 0 if index.start is None else index.start
            step = 1 if index.step is None else index.step
            stop = self.size if index.stop is None else min(index.stop, self.size)
            start %= self.size
            stop = (stop-1) % self.size+1
            if step<0:start,stop=stop-1,start-1
            proj._index = self._index[np.arange(start,stop,step)]
            if len(proj._index):
                return proj
        else:
            print('index was not understood')
            return
        return proj
        
    @property
    def Types(self):
        return mk_nparray_nice(self.pinfo['Type'][self._index])
    
    @property
    def statuses(self):
        return mk_nparray_nice(self.pinfo['status'][self._index])
    
    @property
    def titles(self): 
        return mk_nparray_nice(self.pinfo['title'][self._index])
    
    @property
    def short_files(self):
        return mk_nparray_nice(self.pinfo['short_file'][self._index])
    
    @property
    def additional_info(self):
        return mk_nparray_nice(self.pinfo['additional_info'][self._index])
    
    @property
    def filenames(self):
        return mk_nparray_nice(self.pinfo['filename'][self._index])
    
    @property
    def parent_index(self):
        """
        Returns the location of data objects of a subproject in the parent project

        Returns
        -------
        np.array

        """
        if not(self._subproject):return None
        return np.array([np.argwhere(i==self._parent._index)[0,0] for i in self._index],dtype=int)
        
        
    def save(self,include_rawMD:bool=False,include_pdbs:bool=True):
        """
        Saves the project. By default, raw MD correlation functions will not be
        saved, and pdbs will be created and saved for each selection object in
        the project (pdbs for data not in memory will not be saved!)

        Parameters
        ----------
        include_rawMD : bool, optional
            Determines whether or not to save raw MD data, i.e. the full 
            correlation functions. The default is False.
        include_pdbs : bool, optional
            Determines whether or not to create a pdb for each selection object.
            Creating a pdb will allow one to load the project on a computer
            where the original structural data is not stored. 
            The default is True.

        Returns
        -------
        None.

        """
        assert not(self._subproject),"Sub-projects cannot be saved"
        self.data.save(include_rawMD=include_rawMD)
        self.write_proj()
        if include_pdbs:self.save_pdbs()
        
    def save_pdbs(self,load_all:bool=False):
        """
        Saves a pdb for every selection object that is currently loaded and 
        saved in the project. Note, this operation acts on the full project,
        not the subproject.

        Parameters
        ----------
        load_all : bool, optional
            Force load all data in the project to ensure a pdb from each. 
            Can be time/memory consuming. The default is False.

        Returns
        -------
        None.

        """
        pdb_dir=os.path.join(self.directory,'pdbs')  #Directory for storing pdbs
        if not(os.path.exists(pdb_dir)):
            os.mkdir(pdb_dir)
        
        
        full_pdb=True #TODO Future option?
        """Probably not a future option. The issue is that saving a partial
        pdb changes the atom ids, which are used to define the selection
        objects. We would require a mechanism for re-indexing the selections,
        while still accounting for the possibility of reloading from the original
        topology which then does not have atom renumbering...
        """
        
        #Load list of existing pdbs
        data_loc=[]
        saved_pdb=[]
        origin=[]
        if os.path.exists(os.path.join(pdb_dir,'pdb_list.txt')):
            with open(os.path.join(pdb_dir,'pdb_list.txt'),'r') as f:
                for line in f:
                    data_file,pdb_file,pdb_orig=line.strip().split(':')
                    data_loc.append(data_file)
                    saved_pdb.append(pdb_file)
                    origin.append(pdb_orig)
        
        if load_all:
            for d in self.data:pass  #Loads all data
          
        with open(os.path.join(pdb_dir,'pdb_list.txt'),'w') as f:
            for q,(d,filename) in enumerate(zip(self.data.data_objs,self.data.saved_files)):
                
                if filename is not None:
                    filename=os.path.split(filename)[1]  #Make sure just the file
                    if d is None:  #Unloaded data 
                        if filename in data_loc:  #And that data has a previously saved pdb
                            i=data_loc.index(filename)
                            f.write(f'{filename}:{saved_pdb[i]}:{origin[i]}\n')
                    elif d is not None and filename is not None:  #Loaded data
                        sel=d.source.select
                        if sel is None or sel.uni is None:  #no selection loaded
                            pass
                        elif os.path.abspath(sel.uni.filename) in origin:  #pdb already saved
                            i=origin.index(os.path.abspath(sel.uni.filename))
                            f.write(f'{filename}:{saved_pdb[i]}:{origin[i]}\n')
                        elif os.path.split(os.path.abspath(sel.uni.filename))[0]==os.path.join(self.directory,'pdbs')\
                            and os.path.split(sel.uni.filename)[1] in saved_pdb: #pdb created on previous save
                            i=saved_pdb.index(os.path.split(sel.uni.filename)[1])
                            f.write(f'{filename}:{saved_pdb[i]}:{origin[1]}\n')
                        else: #We need to save the pdb
                            fileout=os.path.split(sel.uni.filename)[1].rsplit('.',maxsplit=-1)[0]
                            count=0
                            file0=fileout
                            while fileout in saved_pdb: #Ensure unique save location
                                count+=1
                                fileout=file0+str(count)
                             
                            sel.traj[0]  #Rewind the trajectory before writing
                            
                            #Maybe just store a subset of the pdb?
                            
                            if full_pdb:
                                copyfile(sel.molsys.topo,os.path.join(pdb_dir,fileout+'.pdb'))
                                # if len(sel.uni.atoms)<100000:  #Built in writer doesn'twork for 100000 or more atoms
                                #     sel.uni.atoms.write(os.path.join(pdb_dir,fileout+'.pdb')) #write the pdb
                                # else:
                                #     write_PDB(sel.uni.atoms,os.path.join(pdb_dir,fileout+'.pdb'),overwrite=True)
                            else:
                                fileout+=str(q+1)
                                sel0=np.sum(sel.sel1+sel.sel2)
                                sel0=np.sum([res.atoms for res in sel0.residues]) #Collect all residues
                                
                                if len(sel0)<100000:  #Built in writer doesn'twork for 100000 or more atoms
                                    sel0.write(os.path.join(pdb_dir,fileout+'.pdb')) #write the pdb
                                else:
                                    write_PDB(sel0,os.path.join(pdb_dir,fileout+'.pdb'),overwrite=True)
                                                      
                            data_loc.append(filename)
                            saved_pdb.append(fileout)
                            origin.append(os.path.abspath(sel.uni.filename)+('' if full_pdb else f'({q+1})') )
                            f.write(f'{data_loc[-1]}:{saved_pdb[-1]}:{origin[-1]}\n')
                        
                    

    #%% Project operations (|,&,-, i.e. Union, Intersection, and Difference)
    def __add__(self,obj):
        "Can't make up my mind about this...the or operation is sort of like adding two sets"
        return self.__or__(obj)
    
    def __radd__(self,obj): #This allows the built-in "sum" function to work
        if obj==0:  
            return self
        return self.__or__(obj)
    
    def __or__(self,obj):
        if hasattr(obj,'parent'):
            assert self.parent is obj.parent,"Project operations (+,|,-,&) are only defined within the same parent project"
        
        proj=copy(self)
        proj._subproject=True
        proj._parent=self._parent if self._subproject else self
        proj.chimera=copy(self.chimera)
        proj.chimera.project=proj
        
        if str(clsDict['Data'])==str(obj.__class__):
            if obj in self.data.data_objs:
                i=self.data.data_objs.index(obj)
                if i not in proj._index:
                    proj._index=np.concatenate((proj._index,[i]))
                else:
                    print('Warning: Data object {} already in subproject'.format(obj.title))
                return proj
            else:
                print('Warning: Union only defined withing subprojects of the same main project')
                return
            
        
        assert str(self.__class__)==str(obj.__class__),"Operation not defined"
        
        for i in obj._index:
            if i not in proj._index:
                proj._index=np.concatenate((proj._index,[i]))
        return proj
        
    def __sub__(self,obj):
        assert self.parent is obj.parent,"Project operations (+,|,-,&) are only defined within the same parent project"
        proj=copy(self)
        proj._subproject=True
        proj._parent=self._parent if self._subproject else self
        proj.chimera=copy(self.chimera)
        proj.chimera.project=proj
        
        if str(clsDict['Data'])==str(obj.__class__):
            if obj in self.data.data_objs:
                i=self.data.data_objs.index(obj)
                if i in proj._index:
                    proj._index=proj._index[proj._index!=i]
                else:
                    print('Warning: Data object {} not in subproject'.format(obj.title))
                return proj
            else:
                print('Warning: Exclusion only defined withing subprojects of the same main project')
                return
        assert str(self.__class__)==str(obj.__class__),"Operation not defined"
        
        for i in obj._index:
            if i in proj._index:
                proj._index=proj._index[proj._index!=i]
        return proj
    
    def __and__(self,obj):
        assert self.parent is obj.parent,"Project operations (+,|,-,&) are only defined within the same parent project"
        proj=copy(self)
        proj._subproject=True
        proj._parent=self._parent if self._subproject else self
        proj.chimera=copy(self.chimera)
        proj.chimera.project=proj
        
        if str(clsDict['Data'])==str(obj.__class__):
            if obj in self.data.data_objs:
                i=self.data.data_objs.index(obj)
                proj._index=np.array([i] if i in self._index else [],dtype=int)
                return proj
            else:
                print('Warning: Intersection only defined withing subprojects of the same main project')
                return
        else:
            proj._index=np.intersect1d(self._index,obj._index)
            return proj
        
        
        
    
    #%% Plotting functions
    @property
    def plot_obj(self):
        """
        Returns the current DataPlots object.

        Returns
        -------
        DataPlots
            DESCRIPTION.

        """
        if self.current_plot:
            return self.plots[self.current_plot-1]
    
    @property
    def fig(self) -> Figure:
        """
        Returns the current matplotlib Figure

        Returns
        -------
        Figure

        """
        if self.current_plot:
            return self.plot_obj.fig
    
    def savefig(self,filename:str=None,fignum:int=None,filetype:str='png',overwrite:bool=False) -> None:
        """
        Saves a figure from the project into the project's figure folder.
        
        Parameters
        ----------
        fignum : int, optional
            Index of the figure in the project (1 or higher). Defaults to the
            current figure if not provided. The default is None.
        filename : str, optional
            Filename for the figure. Defaults to the window title if not 
            provided. Provide an absolute path to save outside of the project 
            folder.
            The default is None.
        overwrite : bool, optional
            Overwrite existing figures. The default is False.

        Returns
        -------
        None

        """
        if not(os.path.exists(os.path.join(self.directory,'figures'))):
            os.mkdir(os.path.join(self.directory,'figures'))
            
        if fignum is None:fignum=self.current_plot
        assert self.plots[fignum-1] is not None,"Selected figure ({}) does not exist".format(fignum)
        if filename is None:
            filename=self.plots[fignum-1].fig.canvas.get_window_title()
            for s in ['/','.']:filename=filename.replace(s,'_')
        for s in [' ','%','&','{','}','<','>','*','?','$','!',"'",'"',':','@','+','`','|','=']:
            # filename=filename.replace(s,'' if s in [',','.','"',"'"] else '_')
            filename=filename.replace(s,'_')
            while '__' in filename:filename=filename.replace('__','_')
        if filetype[0]=='.':filetype=filetype[1:]
        if filename[-4]!='.':filename+='.'+filetype
        
        filename=os.path.join(os.path.join(self.directory,'figures'),filename)
        
        if not(overwrite):
            assert not(os.path.exists(filename)),'File already exists, set overwrite=True'
        
        self.plots[fignum-1].fig.savefig(filename)
        
    @property
    def current_plot(self):
        if self._current_plot[0]>len(self.plots):self._current_plot[0]=len(self.plots)
        return self._current_plot[0]
    
    def close_fig(self, fig):
        """
        Closes a figure of the project

        Parameters
        ----------
        plt_num : int
            Clears and closes a figure in the project. Provide the index 
            (ignores indices corresponding to non-existant figures)

        Returns
        -------
        None.

        """
        if isinstance(fig,str) and fig.lower()=='all':
            for i in range(len(self.plots)):self.close_fig(i)
            self.plots.clear()
            self.plots.append(None)
            return
        fig-=1
        if len(self.plots) > fig and self.plots[fig] is not None:
            hdl=self.plots[fig]
            # self.plots[fig].close()
            plt.close(self.plots[fig].fig)
            self.plots[fig] = None
        
    def plot(self, data=None,fig=None, style='plot',errorbars=False, index=None, rho_index=None, split=True, plot_sens=True, title=None, **kwargs):
        """
        

        Parameters
        ----------
        data : pyDR.Data, optional
            data object to be plotted. The default is None.
        data_index : int, optional
            index to determine which data object in the project to plot. The default is None.
        fig : int, optional
            index to determine which plot to use. The default is None (goes to current plot).
        style : str, optional
            'p', 's', or 'b' specifies a line plot, scatter plot, or bar plot. The default is 'plot'.
        errorbars : bool, optional
            Show error bars (True/False). The default is True.
        index : int/bool array, optional
            Index to determine which residues to plot. The default is None (all residues).
        rho_index : int/bool array, optional
            index to determine which detectors to plot. The default is None (all detectors).
        split : bool, optional
            Split line plots with discontinuous x-data. The default is True.
        plot_sens : bool, optional
            Show the sensitivity of the detectors in the top plot (True/False). The default is True.
        **kwargs : TYPE
            Various arguments that are passed to matplotlib.pyplot for plotting (color, linestyle, etc.).

        Returns
        -------
        DataPlots object

        """
        if fig is None:fig=self.current_plot if self.current_plot else 1
        self.current_plot=fig
        
        
        
        # if title:self.fig.canvas.set_window_title(title)
        
        # if data is None and data_index is None: #Plot everything in project
        #     if self.size:
        #         for i in range(self.size):
        #             out=self.plot(data_index=i,style=style,errorbars=errorbars,index=index,
        #                      rho_index=rho_index,plot_sens=plot_sens,split=split,fig=fig,**kwargs)
        #         return out
        #     return
        
        # if data is None:
        #     data = self[data_index]
        # if self.plots[fig-1] is None:
        #     self.plots[fig-1] = clsDict['DataPlots'](data=data, style=style, errorbars=errorbars, index=index,
        #                  rho_index=rho_index, plot_sens=plot_sens, split=split, **kwargs)
        #     self.plots[fig].project=self
        # else:
        
        if data is None:            
            for data in self:
                self.plots[fig-1].append_data(data=data,style=style,errorbars=errorbars,index=index,
                             rho_index=rho_index,plot_sens=plot_sens,split=split,**kwargs)
        else:
            self.plots[fig-1].append_data(data=data,style=style,errorbars=errorbars,index=index,
                         rho_index=rho_index,plot_sens=plot_sens,split=split,**kwargs)
        return self.plots[fig-1]
    
    def add_fig(self,fig):
        if len(self.plots) and self.plots[-1] is None:
            self.plots[-1]=clsDict['DataPlots'](fig=fig)
        else:
            self.plots.append(clsDict['DataPlots'](fig=fig))
        self.plots[-1].project=self
        self.current_plot=len(self.plots)
        

    def comparable(self, i: int, threshold: float = 0.9, mode: str = 'auto', min_match: int = 2) -> tuple:
        """
        Find objects that are recommended for comparison to a given object. 
        Provide either an index (i) referencing the data object's position in 
        the project or by providing the object itself. An index will be
        returned, corresponding to the data objects that should be compared to
        the given object
        
        Comparability is defined as:
            1) Having some selections in common. This is determined by running 
                self[k].source.select.compare(i,mode=mode)
            2) Having 
                a) Equal sensitivities (faster calculation, performed first)
                b) overlap of the object sensitivities above some threshold.
                This is defined by sens.overlap_index, where we require 
                at least min_match elements in the overlap_index (default=2)
        """

        #todo argument mode is never used?

        if isinstance(i, int):
            i = self[i] #Get the corresponding data object
        out = list()
        for s in self:
            if s.select.compare(i.select)[0].__len__() == 0:
                out.append(False)
                continue
            out.append(s.sens.overlap_index(i.sens, threshold=threshold)[0].__len__() >= min_match)
        return np.argwhere(out)[:, 0]
                    
    


    #%% Fitting functions
    def opt2dist(self, rhoz=None,rhoz_cleanup:bool=False, parallel:bool=False):
        """
        Forces a set of detector responses to be consistent with some given distribution
        of motion. Achieved by performing a linear-least squares fit of the set
        of detector responses to a distribution of motion, and then back-calculating
        the detectors from that fit. Set rhoz_cleanup to True to obtain monotonic
        detector sensitivities: this option eliminates unusual detector due to 
        oscilation and negative values in the detector sensitivities. However, the
        detectors are no longer considered "DIstortion Free".
                                
    
        Parameters
        ----------
        rhoz : np.array, optional
            Provide a set of functions to replace the detector sensitivities.
            These should ideally be similar to the original set of detectors,
            but may differ somewhat. For example, if r_target is used for
            detector optimization, rhoz may be set to removed residual differences.
            Default is None (keep original detectors)
        
        rhoz_cleanup : bool, optional
            Modifies the detector sensitivities to eliminate oscillations in the
            data. Oscillations larger than the a threshold value (default 0.1)
            are not cleaned up. The threshold can be set by assigning the 
            desired value to rhoz_cleanup. Note that rhoz_cleanup is not run
            if rhoz is defined.
            Default is False (keep original detectors)
    
        parallel : bool, optional
            Use parallel processing to perform optimization. Default is False.
    
        Returns
        -------
        Subproject (Project)
    
        """
        
        # index0=copy(self.parent._index)
        self._projDelta(initialize=True)
        
        sens = list()
        detect = list()
        count = 0
        for d in self:
            if hasattr(d.sens,'opt_pars') and 'n' in d.sens.opt_pars:
                fit = d.opt2dist(rhoz=rhoz,rhoz_cleanup=rhoz_cleanup, parallel=parallel)
                if fit is not None:
                    count += 1
                    if fit.sens in sens:
                        i = sens.index(fit.sens)
                        fit.sens = sens[i]
                        fit.detect = detect[i]
                    else:
                        sens.append(fit.sens)
                        detect.append(clsDict['Detector'](fit.sens))
        print('Optimized {0} data objects'.format(count))
        
        # out=self[:0]
        # out._index=np.setdiff1d(self.parent._index,index0)
        out=self._projDelta()
        
        return out
    
    def fit(self, bounds: bool = 'auto', parallel: bool = False) -> None:
        """
        Fit all data in the project that has optimized detectors.
        """
        sens = list()
        detect = list()
        count = 0
        to_delete=list()
        
        # index0=copy(self.parent._index)
        
        self._projDelta(initialize=True)
        
        for d in self:
            if 'n' in d.detect.opt_pars:
                count += 1
                fit = d.fit(bounds=bounds, parallel=parallel)
                if fit.sens in sens:
                    i = sens.index(fit.sens)    #We're making sure not to have copies of identical sensitivities and detectors
                    fit.sens = sens[i]
                    fit.detect = detect[i]
                else:
                    sens.append(fit.sens)
                    detect.append(fit.detect)
                if Defaults['reduced_mem']:
                    if d.source.status=='raw':
                        i0=self.data.data_objs.index(d)
                        self.data.data_objs[i0]=None
                        to_delete.append(self._index.tolist().index(i0))
        self.remove_data(to_delete)
        
        # out=self[:0]
        # out._index=np.setdiff1d(self.parent._index,index0)

        out=self._projDelta()
        print('Fitted {0} data objects'.format(count))
        return out
    
    def _projDelta(self,initialize=False):
        """
        Determines how a project changes after an operation, and returns the
        project difference

        Parameters
        ----------
        initialize : TYPE, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        None.

        """
        if initialize:
            self._index0=copy(self.parent._index)
            return
        else:
            assert self._index0 is not None,"Finding the project Delta requires first initializing"
            out=self[:0]
            out._index=np.setdiff1d(self.parent._index,self._index0)
            self._index0=None
            return out
            
            
    
    def modes2bonds(self,inclOverall:bool=False,calcCC='auto'):
        """
        
        Converts iRED mode detector responses into bond-specific detector 
        responses, including calculation of cross-correlation matrices for each 
        detector. These are stored in CC and CCnorm, where CC is the unnormalized
        correlation and CCnorm is the correlation coefficient, i.e. Pearson's r
        
        Parameters
        ----------
        inclOverall : bool, optional
            Determines whether to include the 3 or 5 overall modes of motion
            (depends on rank). The default is False
        calcCC : bool, optional
            Usually modes2bonds is run after fitting the modes with detectors.
            Then, cross correlation is only calculated for a few detectors. 
            However, if run before fitting, then a large number of cross
            correlation terms would be calculated. Therefore, by default, we only
            calculate CC if there are less than 40 different detectors/time points.
            Override this behavior by setting to True or False
            The default is 'auto'.
        
        Returns
        -------
        None (appends data to project)
        """
        
        # index0=copy(self.parent._index)
        self._projDelta(initialize=True)
        
        count = 0
        for d in self:
            if hasattr(d,'iRED') and 'Lambda' in d.iRED:
                count+=1
                d.modes2bonds(inclOverall=inclOverall)
                
        # out=self[:0]
        # out._index=np.setdiff1d(self.parent._index,index0)
        out=self._projDelta()
        
        print('Converted {0} iRED data objects from modes to bonds'.format(count))
        return out
                

    #%% iPython stuff   
    def _ipython_display_(self):
        print("pyDIFRATE project with {0} data sets\n{1}\n".format(self.size,super().__repr__()))
        print('Titles:')
        for t in self.titles:print(t)
        
    def __repr__(self):
        out='pyDIFRATE project with {0} data sets\n\ntitles:\n'.format(self.size)
        for t in self.titles:out+=t+'\n'
        return out
        
    def _ipython_key_completions_(self):
        out = list()
        for k in ['Types','statuses','additional_info','titles','short_files']:
            for v in getattr(self, k):
                if v not in out:
                    out.append(v)
        for v0 in self.short_files:
            if v0 is not None:
                v=v0.rsplit('.',1)[0]
                if v not in out and v is not None:
                    out.append(v)
        return out
 



       








