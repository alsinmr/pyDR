#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 09:38:19 2023

@author: albertsmith
"""

from pyDR import clsDict,Defaults
import os
from pyDR.misc.Averaging import avgDataObjs

class Chunk():
    def __init__(self,fr_obj,chunks:int=5,mode:str='slow',n:int=15,\
                 temp_proj:str='chunked',remove_temp:bool=True,**kwargs):
        """
        Usage:
            with Chunk(fr_obj,chunks=5,mode='slow',...) as chunk:
                data=chunk()
                
            Execution in the with construct is not required, but ensures that if
            processing is interupted, the original fr_obj is returned to its 
            initial state. This is the recommended usage.
        
        
        The frame chunker is used for processing large correlation functions in steps
        by either skipping time points where the chunks are (5 steps): t[0::5],
        t[1::5],t[2::5],t[3::5],t[4::5]) or taking the first n points, the second 
        n points, etc., where n=len(t)//5, and the chunks are t[0:n],t[n:2*n],
        t[2*n:3*n], t[3*n:4*n],t[4*n:5*n].
        
        The first mode is 'fast', where chunking will cause one to lose information
        about slow motion, and the second mode is 'slow', where chunking
        will cause one to lose information about slow motion.
        
        After producing a chunk, the chunker will use n unoptimized detectors to
        process the resulting correlation functions. The original correlation 
        functions will be discarded immediately.
        
        Chunks are stored in a temporary project (temp_proj). If the chunker is
        started and the temporary project already exists, then it will assume that
        existing data in the project is from a partially-completed chunking job
        (thus allowing one to re-start an incomplete job). Then, care should be
        taken to ensure that there is not a temporary project remaining with a 
        different chunking job, since the reconstruction of the chunks at the end
        will likely fail.
        
        The chunker will set pyDR into reduced_mem mode. 
        
        If fr_obj.traj.step is not 1, then the chunker will return processing that
        uses that step setting on top of the chunking. For example, for mode='slow',
        if we have fr_obj.traj.step=5 and chunks=5, then the step will be set to
        25 for each chunk.
        
        When chunking is completed, a list of data objects will be returned
        resulting from the chunking operation. If the original fr_obj has an 
        attached project, those data objects will also be appended to the data object.
        
        If no frames are stored in the fr_obj, then md2data is run instead
        of frames2data.
        
        kwargs are passed to frames2data/md2data. Note that return_index is not
        allowed as a kwarg
    
        Parameters
        ----------
        fr_obj : pyDR frame object
            Frame object to be processed via chunking.
        chunks : int, optional
            Number of chunks to use. The default is 5.
        mode : str, optional
            Whether to chunk to investigate fast or slow motion. 
            The default is 'slow'.
        n : int, optional
            Number of unoptimized detectors to fit the data to. 
            The default is 15.
        temp_folder : str, optional
            Location of the temporary project for chunking. 
            The default is 'gechunked'.
        remove_temp : bool, optional
            Boolean to determine if the temporary project should be deleted after
            the full chunking process is complete. The default is True.
        kwargs : type
            Keyword arguments to be passed to frames2data or md2data.
            (mode,return_index,include)
    
        Returns
        -------
        None.
    
        """
        
        self.fr_obj=fr_obj
        self.project=fr_obj.project
        traj=self.traj=fr_obj.traj
        self.settings={'t0':traj.t0,'tf':traj.tf,'step':traj.step}
        self.chunks=chunks
        self.mode=mode
        self.n=n
        self.temp_proj=clsDict['Project'](temp_proj,create=True)
        self.fr_obj.project=self.temp_proj
        self.remove_temp=True
        self._reduced_mem=Defaults['reduced_mem']
        self.kwargs=kwargs
        self._post_process=None
        self.detect=None
        self.verbose=True
        Defaults['reduced_mem']=True  #Run chunking in reduced memory mode
        
    def __enter__(self):
        return self
    
    def __exit__(self,type,value,traceback):
        """
        Runs self.cleanup and passes errors

        Parameters
        ----------
        type : TYPE
            DESCRIPTION.
        value : TYPE
            DESCRIPTION.
        traceback : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.cleanup()

    
    def cleanup(self):
        """
        Resets some settings in the frame_object and the Default parameter
        reduced_mem

        Returns
        -------
        None.

        """
        Defaults['reduced_mem']=self._reduced_mem
        self.traj.t0,self.traj.tf,self.traj.step=self.settings.values()
        self.fr_obj.project=self.project
        if len(self.temp_proj)==0:
            self.delete_temp()
    
    def __call__(self):
        for k in range(self.complete_chunks,self.chunks):
            self.run_chunk(k)
        out=self.average_results()
        if self.project is not None:
            for d in out:self.project.append_data(d)
            print('Data transfered to original project')
        if self.remove_temp:self.delete_temp()
        return out
        
        
    def delete_temp(self):
        """
        Deletes the temporary project and its full contents

        Returns
        -------
        None.

        """
        
        def del_dir(directory):
            for file in os.listdir(directory):
                file=os.path.join(directory,file)
                if os.path.isdir(file):
                    del_dir(file)
                else:
                    os.remove(file)
            os.rmdir(directory)
        
        del_dir(self.temp_proj.directory)
    
    @property
    def objs_per_chunk(self):
        """
        Determines how many data objects a chunk should return

        Returns
        -------
        int

        """
        
        if self.fr_obj.nf==0:
            return 1
        
        if 'return_index' in self.kwargs:
            return_index=self.kwargs['return_index']
        else:
            return_index=self.fr_obj.return_index

        if 'include' in self.kwargs:
            include=self.kwargs['include']
        else:
            include=self.fr_obj.include
                    
        nf=sum(include) if (include is not None and len(include)==self.fr_obj.nf)\
            else self.fr_obj.nf

        count=0
        i=[i0 for i0 in return_index]
        if i[0]:count+=nf+1
        if i[7]:count+=1
        if i[8]:count+=1
        return count
        
    @property
    def complete_chunks(self):
        """
        Returns the number of chunks that have already been completed

        Returns
        -------
        int

        """
        
        return len(self.temp_proj)//self.objs_per_chunk
    
    @property
    def partial_chunk(self):
        """
        If a chunk was partially completed, this determines how many data
        objects of that chunk were finished

        Returns
        -------
        int

        """
        return len(self.temp_proj)-self.complete_chunks*self.objs_per_chunk
    
    def chunk_pars(self,k:int):
        """
        Returns t0, tf, and step for the kth chunk

        Parameters
        ----------
        k : int
            Index of the chunk to get parameters.

        Returns
        -------
        tuple (t0,tf,step)

        """
        
        if self.mode=='slow':
            step=self.settings['step']*self.chunks
            t0=self.settings['t0']+k
            tf0=self.settings['tf']
            tf=((tf0-self.settings['t0']-self.chunks+1)//step)*step+t0
        else:
            step=self.settings['step']
            length=(self.settings['tf']-self.settings['t0'])//self.chunks
            t0=k*length
            tf=(k+1)*length
        return t0,tf,step
    
    def run_chunk(self,k:int):
        """
        Runs the kth chunk

        Parameters
        ----------
        k : int
            Index of the chunk to run.

        Returns
        -------
        None.

        """
        
        if self.verbose:print(f'Running chunk {k+1} of {self.chunks}')
        self.traj.t0,self.traj.tf,self.traj.step=self.chunk_pars(k)
        self.fr_obj.load_frames()
        if self.verbose:print('Data loading completed')
        pc=self.partial_chunk
        
        if self._post_process is not None:
            Type,args,kwargs=self._post_process
            self.fr_obj.post_process(Type=Type,*args,**kwargs)
        if self.verbose and self._post_process:print('Post-processing completed')
        
        if self.fr_obj.nf==0:
            self.fr_obj.md2data(**self.kwargs)
        else:
            self.fr_obj.frames2data(**self.kwargs)
        if self.verbose:print('Correlation functions calculated')
            
            
        if pc: #Delete raw data where results are already stored
            self.temp_proj.remove_data(range(-self.objs_per_chunk,-self.objs_per_chunk+pc))
        
        if self.detect is not None:  #Recycle the detector object
            for d in self.temp_proj['raw']:d.detect=self.detect
        else:
            self.temp_proj['raw'].detect.r_no_opt(self.n)
            self.detect=self.temp_proj['raw'][0].detect
            
        self.temp_proj['raw'].fit()
        if self.verbose:print('Data fitted')
        self.temp_proj.save()
        if self.verbose:print('Data saved')
        
    def average_results(self):
        """
        Extracts data from the temporary project, averages it together, and 
        appends the result to the original project

        Returns
        -------
        None.

        """
        out=list()
        for k in range(self.objs_per_chunk):
            sub=self.temp_proj[k::self.objs_per_chunk]
            for d in sub:d.project=None
            out.append(avgDataObjs(sub))
            for d in sub:d.project=self.temp_proj
        return out
    
    def post_process(self,Type:str=None,*args,**kwargs):
        """
        Tells the chunker to include post processing. Args and kwargs are
        passed to fr_obj.post_process

        Parameters
        ----------
        Type : str
            Name of post-processing function to use.
        *args : TYPE
            Positional arguments for fr_obj.post_process.
        **kwargs : TYPE
            Keyword arguments for fr_obj.post_process.

        Returns
        -------
        None.

        """
        self._post_process=Type,args,kwargs
        
        
        
            
    
        
    
        