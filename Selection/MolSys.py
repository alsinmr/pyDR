#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 13:34:36 2021

@author: albertsmith
"""
import numpy as np
from MDAnalysis import Universe, AtomGroup
from MDAnalysis.topology.guessers import guess_atom_type
from pyDR.misc.ProgressBar import ProgressBar
from pyDR.Selection import select_tools as selt
from pyDR.MDtools.vft import pbc_corr
from pyDR.IO import getPDB,readPDB,write_PDB
from pyDR import Defaults,clsDict
from copy import copy
import os
from pyDR.Selection.MovieSys import MovieSys
from time import time
dtype=Defaults['dtype']

class MolSys():
    """
    Object for storage of the molecule or MD trajectory
    """
    def __init__(self,topo:str,traj_files:list=None,t0:int=0,tf:int=None,step:int=1,dt:float=None,project=None):
        """
        Generates a MolSys object, potentially with a trajectory attached, in
        case traj_files is specified. Note that if the topology provided does
        not include positions (.psf, for example), then you must provide
        traj_files for viewing purposes.

        Parameters
        ----------
        topo : str
            Location of the topology file (pdb,psf, etc). The default is None.
        traj_files : str or list of strs, optional
            Location(s) of trajectory. The default is None.
        t0 : int, optional
            First time point to use in the trajectory. The default is 0.
        tf : int, optional
            Last time point to use in the trajectory. Set to None to go to the 
            end of the trajectory. The default is None.
        step : int, optional
            Step size to skip frames in trajectory. Set to 1 for all frames.
            The default is 1.
        dt : float, optional
            Overrides the saved timestep in the MD trajectory (ps). Set to None
            to use the saved timestep. The default is None.
        project : pyDR.Project, optional
            Attach a Project object to this MolSys, such that data generated from
            this MolSys will be attached to the project. The default is None.

        Returns
        -------
        MolSys.

        """
        self._directory=None
        self._uni=None
        
        
        if topo is not None and len(topo)==4 and not(os.path.exists(topo)):
            topo=getPDB(topo)
        
        if topo is None:
            self._uni=None
            self._traj=None
            return
        elif traj_files is not None and not(isinstance(traj_files,list) and len(traj_files)==0):
            if isinstance(traj_files,list):
                traj_files=[os.path.abspath(tf) for tf in traj_files]
            else:
                traj_files=os.path.abspath(traj_files)
            self._uni=Universe(os.path.abspath(topo))
        else:
            try:
                self._uni=Universe(topo)
            except:
                self._uni=readPDB(topo)

        self._traj_info={'traj_files':traj_files,'t0':t0,'tf':tf,'step':step,'dt':dt}
        self._orig_topo=topo
        
        i=self.uni.atoms.types==''
        self.uni.atoms[i].types=[guess_atom_type(s.name) for s in self.uni.atoms[i]] #Fill in types where missing (ok??) 
        self._traj=Trajectory(self.uni.trajectory,t0=t0,tf=tf,step=step,dt=dt) \
            if hasattr(self.uni,'trajectory') else None
         
        self.project=project
        self.make_pdb()
        
        
        
        self._movie=None
        
        
    
    @property
    def uni(self):
        "Return the universe object corresponding to this element of the MolSys"
        return self._uni
    
    @property
    def traj(self):
        if self._traj_info['traj_files'] is not None: #Skip loading the trajectory until required
            self.new_trajectory(**self._traj_info)
            self._traj_info['traj_files']=None
        return self._traj
    @property
    def topo(self):
        if self.uni is not None:
            return self.uni.filename

    @property
    def details(self):
        out=['Topology: {}'.format(self._orig_topo)]
        out.extend(self.traj.details)
        return out

    @property
    def directory(self):
        """
        Returns a working directory for temporary files

        Returns
        -------
        str

        """
        if self._directory is None:
            # Set up temp directory, filename
            if self.project is None or self.project.directory is None:
                directory=os.path.abspath('temp_pyDR')
            else:
                directory=os.path.join(self.project.directory,'temp_pyDR')
            if not(os.path.exists(directory)):os.mkdir(directory)
            self._directory=directory
        return self._directory
            

    @property
    def _hash(self):
        return hash(self.topo)

    
    def select_atoms(self,select_str:str):
        """
        Runs the MDAnalysis universe select_atoms function.

        Parameters
        ----------
        select_str : str
            Selection string to be called on the mdanalyis universe select_atoms
            function.

        Returns
        -------
        AtomGroup
            MDAnalysis atom group.

        """
        
        return self.uni.atoms.select_atoms(select_str)
    
    def select_filter(self,resids=None,segids=None,filter_str=None) -> AtomGroup:
        """
        Create a selection from the MolSys universe based on filtering by
        resid, segid, and a filter string.        

        Parameters
        ----------
        resids : list/array/single element, optional
            Restrict selected residues. The default is None.
        segids : list/array/single element, optional
            Restrict selected segments. The default is None.
        filter_str : str, optional
            Restricts selection to atoms selected by the provided string. String
            is applied to the MDAnalysis select_atoms function. The default is None.

        Returns
        -------
        AtomGroup
            DESCRIPTION.

        """
        return selt.sel_simple(self.uni.atoms,resids=resids,segids=segids,filter_str=filter_str)

    def __hash__(self):
        return hash(self.topo) + hash(self.traj)
    
    def chimera(self):
        """
        Opens the molecule in Chimera for viewing

        Returns
        -------
        None.

        """
        
        CMXRemote=clsDict['CMXRemote']

        if self.project is not None:
            ID=self.project.chimera.CMXid
            if ID is None:
                self.project.chimera.current=0
                ID=self.project.chimera.CMXid
        else: #Hmm....how should this work?
            ID=CMXRemote.launch()


        CMXRemote.send_command(ID,'open "{0}" maxModels 1'.format(self.topo))
        
        vm=[]
        t0=time()
        while not(vm):
            vm=CMXRemote.valid_models(ID)
            assert time()-t0<10,'Timeout occured while opening model in ChimeraX'
        mn=vm[-1]
        CMXRemote.command_line(ID,f'sel #{mn}')

        CMXRemote.send_command(ID,'style sel ball')
        CMXRemote.send_command(ID,'size sel stickRadius 0.2')
        CMXRemote.send_command(ID,'size sel atomRadius 0.8')
        # CMXRemote.send_command(ID,'~ribbon')
        # CMXRemote.send_command(ID,'show sel')
        # CMXRemote.send_command(ID,'color sel tan')
        CMXRemote.send_command(ID,'~sel')
        
        if self.project is not None and self.project.chimera.saved_commands is not None:
            for cmd in self.project.chimera.saved_commands:
                CMXRemote.send_command(ID,cmd)
                
    def make_pdb(self,ti:int=None,replace:bool=True)->str:
        """
        Creates a pdb from the topology and trajectory files at the specified
        time point (default ti=0). By default, the pdb will be used as the
        topology file for this molsys object and will replace the existing 
        topology (i.e. we'll reload the MDAnalysis universe)

        Parameters
        ----------
        ti : int, optional
            Determines which time point of the trajectory to use for the pdb. If
            ti is not provided and the topology is already a pdb, then make_pdb
            will exit, and return the current topology. If ti is not provided,
            but the topology is not a pdb, then ti will be set to 0.
            The default is None.
        replace : bool, optional
            Determines whether to replace the MDAnalysis universe. The default 
            is True.

        Returns
        -------
        str
            Location of the new pdb. Located in same folder as original topology,
            unless write access is restricted. Then writes to current directory
                                     
        """
    
        
        if self.topo.rsplit('.')[-1]=='pdb' and ti is None:return self.topo #Already a pdb
        if ti is None:ti=0
        
        if self.traj is not None and len(self.traj):self.traj[ti]
        
        
        if self.project is not None and self.project.directory is not None \
            and os.path.exists(os.path.join(self.project.directory,'scratch')):
                folder=os.path.join(self.project.directory,'scratch')
        else:
            folder=os.path.split(self.topo)[0]
        
        filename=os.path.split(self.topo)[1].rsplit('.',maxsplit=1)[0]+'_ts{0}.pdb'.format(self.traj.mda_traj.ts.frame)
        filename=os.path.join(folder,filename) if os.access(folder, os.W_OK) else os.path.abspath(filename)
        if len(self.uni.atoms)<100000:
            self.uni.atoms.write(filename)
        else:
            write_PDB(self.uni.atoms,filename)
        
        if replace:
            self._uni=Universe(filename)
            self._uni.trajectory=self.traj.mda_traj
            
        return filename
    
    def new_molsys_from_sel(self,sel):
        """
        Creates a new MolSys object using a subset of the current topology.
        
        Note that we do not immediately associate this with a trajectory, 
        although that may be done later, via replace_traj
        
        This will also create a temporary folder either in the current directory
        or the project directory, if a project directory exists, to store
        the new topology.

        Parameters
        ----------
        sel : Atom Group
            Atom group to create new molsys from.

        Returns
        -------
        MolSys

        """
        # Check that the universe is the same
        assert self.uni==sel.universe,'Atomgroup must have the same universe as MolSelect object'
        
        directory=self.directory
        
        filename=os.path.split(self.topo)[1].split('.')[0]
        filename0=filename
        k=0
        while os.path.exists(os.path.join(directory,filename0+'.pdb')):
            k+=1
            filename0=filename+str(k)
        else:
            if k!=0:filename=filename0
        filename=os.path.join(directory,filename)+'.pdb'
        sel.write(filename)
        
        return MolSys(topo=filename,project=self.project)
    
    def new_trajectory(self,traj_files,t0:int=0,tf:int=None,step:int=1,dt:float=None):
        """
        Replace the current trajectory with a new trajectory

        Parameters
        ----------
        traj_files : str or list of strs, optional
            Location(s) of trajectory. The default is None.
        t0 : int, optional
            First time point to use in the trajectory. The default is 0.
        tf : int, optional
            Last time point to use in the trajectory. Set to None to go to the 
            end of the trajectory. The default is None.
        step : int, optional
            Step size to skip frames in trajectory. Set to 1 for all frames.
            The default is 1.
        dt : float, optional
            Overrides the saved timestep in the MD trajectory (ps). Set to None
            to use the saved timestep. The default is None.
        
        Returns
        -------
        self
        
        """
        
        if isinstance(traj_files,list):
            traj_files=[os.path.abspath(tf) for tf in traj_files]
        else:
            traj_files=os.path.abspath(traj_files)
        new=Universe(self.topo,traj_files)
        
        self.uni.trajectory=new.trajectory
        self._traj=Trajectory(self.uni.trajectory,t0=t0,tf=tf,step=step,dt=dt)
        
        return self
        
    
    def __del__(self):
        """
        Deletes topology and working directory if temporary 

        Returns
        -------
        None.

        """
        if os.path.split(os.path.split(self.topo)[0])[1]=='temp_pyDR':
            os.remove(self.topo)
            
        if self._directory is not None:
            if len(os.listdir(self.directory))==0:
                os.rmdir(self.directory)
                
    @property
    def movie(self):
        """
        Returns a simple movie launcher/updater if a trajectory is present

        Returns
        -------
        None.

        """
        if len(self.traj.mda_traj)>5000:
            print('Warning: Trajectory has more than 5000 frames. Loading into chimera may be slow')
        
        if self._movie is None:self._movie=MovieSys(self)
        return self._movie


        
        

class Trajectory():
    def __init__(self,mda_traj,t0=0,tf=None,step=1,dt=None):
        self.__tf=len(mda_traj)
        self.t0=t0
        self.tf=tf if tf is not None else self.__tf
        self.step=step
        self.__dt=dt if dt else mda_traj.dt
        self.mda_traj=mda_traj
        self.ProgressBar=False

    def __hash__(self):
        return hash(self.t0) + hash(self.tf) + hash(self.step) + hash(self.dt)

    @property
    def dt(self):
        return self.__dt*self.step
    
    @property
    def time(self):
        return self.mda_traj.time
    
    @property
    def frame(self):
        return (self.mda_traj.frame-self.t0)//self.step
    
    @property
    def lengths(self):
        l=(self.mda_traj.total_times/self.mda_traj.dt).astype(int)
        l[0]-=self.t0
        l[-1]-=self.__tf-self.tf
        return (l/self.step).astype(int)
        
    
    @property
    def details(self):
        out=['Trajectory:'+', '.join(self.files)]
        out.append('t0={0}, tf={1}, step={2}, dt={3} ps, original length={4}'.format(self.t0,self.tf,self.step,self.dt,self.__tf))
        return out

    def __setattr__(self,name,value):
        "Make sure t0, tf, step are integers"
        if name=='project': #Ensure project is really a project
            if value is None or str(value.__class__).split('.')[-1][:-2]=='Project':
                super().__setattr__(name, value)
            return
        
        if name in ['t0','tf','step'] and value is not None:
            value=int(value)
        if name=='tf':
            if value is None:value=self.__tf
            if value>self.__tf:
                # print(f'Warning: tf={value} is greater than the original trajectory length, setting to {self.__tf}')
                value=self.__tf
            value=(value-1)%self.__tf+1 #Take care of negative indices
            
        if name=='t0':
            value=(value)%self.__tf #Take care of negative indices
        super().__setattr__(name,value)
        
    def __getitem__(self,index):
        if np.array(index).dtype=='bool':
            i=np.zeros(self.tf-self.t0,dtype=bool)
            i[::self.step]=True
            index=np.concatenate((np.zeros(self.t0,dtype=bool),i,np.zeros(self.__tf-self.tf,dtype=bool)))
            return self.mda_traj[index]
        elif isinstance(index,slice):
            stop=index.stop if index.stop else len(self)
            assert stop<=self.__len__(),'stop index must be less than or equal to the truncated trajectory length'
            start=index.start if index.start else 0
            
            stop=stop if stop==len(self) else stop%len(self)
            step=index.step if index.step else 1
            
            def iterate():
                for k in range(start,stop,step):
                    if self.ProgressBar:ProgressBar((k+1-start),stop-start,'Loading:','',0,40)
                    yield self[k]
            return iterate()
        elif hasattr(index,'__iter__'):
            def iterate():
                for m,k in enumerate(index):
                    if self.ProgressBar and hasattr(index,'__len__'):ProgressBar(m+1,len(index),'Loading:','',0,40)
                    yield self[k]
            return iterate()
        else:
            assert index<self.__len__(),"index must be less than the truncated trajectory length"
            index%=self.__len__() #Take care of negative indices
            return self.mda_traj[self.t0+index*self.step]
    
    def __len__(self):
        return int((self.tf-self.t0)/self.step)
    
    def __iter__(self):
        return self[:]
    
    @property
    def files(self):
        if hasattr(self.mda_traj,'filenames'):
            return [f for f in self.mda_traj.filenames] #Always return a list
        else:
            return [self.mda_traj.filename] #Always return a list
    
class MolSelect():
    def __init__(self,topo:str=None,traj_files:list=None,molsys:MolSys=None,t0:int=0,tf:int=None,step:int=1,dt:float=None,project=None):
        """
        Provide either a MolSys object, or provide a topology file (usually pdb)
        to generate a selection (MolSelect) object. 

        Parameters
        ----------
        molsys : MolSys, optional
            Provide a MolSys directly, or provide location of a topology, such
            that MolSys is generated upon initialization.
        topo : str, optional
            Location of the topology file (pdb,psf, etc). The default is None.
        traj_files : str or list of strs, optional
            Location(s) of trajectory. The default is None.
        t0 : int, optional
            First time point to use in the trajectory. The default is 0.
        tf : int, optional
            Last time point to use in the trajectory. Set to None to go to the 
            end of the trajectory. The default is None.
        step : int, optional
            Step size to skip frames in trajectory. Set to 1 for all frames.
            The default is 1.
        dt : float, optional
            Overrides the saved timestep in the MD trajectory (ps). Set to None
            to use the saved timestep. The default is None.
        project : pyDR.Project, optional
            Attach a Project object to this MolSys, such that data generated from
            this MolSys will be attached to the project. The default is None.

        Returns
        -------
        None.

        """
        assert not(molsys is None and topo is None),'Either molsys or topo must be provided'
        
        
        if isinstance(topo,clsDict['MolSys']):
            molsys=topo
            topo=None
        
        if molsys is None:
            molsys=MolSys(topo=topo,traj_files=traj_files,t0=t0,tf=tf,step=step,dt=dt,project=project)
        
        super().__setattr__('molsys',molsys)
        self._sel1=None
        self._sel2=None
        self._repr_sel=None
        self._label=None
        self._project=None
        super().__setattr__('_mdmode',False)

    def __setattr__(self,name,value):
        """
        Use to set special behaviors for certain attributes
        """
        if name in ['MolSys']:
            print('Warning: Changing {} is not allowed'.format(name))
            return
        
        if name=='project':
            if value is None or str(value.__class__).split('.')[-1][:-2]=='Project':
                self._project=value
            return
        
        if name=='label':
            super().__setattr__('_label',value)
            return
        
        if name=='_mdmode':
            if value==True and not(self._mdmode):
                if (self._sel1 is not None and np.any([len(s)!=1 for s in self._sel1])) or \
                    (self._sel2 is not None and np.any([len(s)!=1 for s in self._sel2])):
                    print('mdmode only valid for single bond selections')
                    return
                for f in ['_sel1','_sel2']:
                    sel0=getattr(self,f)
                    if sel0 is not None:
                        sel=sel0[0][:0]
                        for s in sel0:
                            sel+=s
                        setattr(self,f,sel)
            elif value==False and self._mdmode:
                if self._sel1 is not None:
                    sel1=np.zeros(len(self._sel1),dtype=object)
                    for k in range(len(self._sel1)):sel1[k]=self._sel1[k:k+1]
                    self._sel1=sel1
                if self._sel2 is not None:
                    sel2=np.zeros(len(self._sel2),dtype=object)
                    for k in range(len(self._sel2)):sel2[k]=self._sel2[k:k+1]
                    self._sel2=sel2
                    
        if name=='repr_sel' and value is not None:
            if self.sel1 is not None and len(self.sel1)!=len(value):
                print('Warning: length of sel1 and repr_sel are not equal. This will cause errors in ChimeraX')
        if name in ['sel1','sel2','repr_sel']:
            # if name=='repr_sel':name='_repr_sel'
            _mdmode=self._mdmode
            self._mdmode=False
            name='_'+name
            if value is None:
                self._mdmode=_mdmode
                return
            elif isinstance(value,AtomGroup):
                sel=np.zeros(len(value),dtype=object)
                for k in range(len(value)):sel[k]=value[k:k+1]
                super().__setattr__(name,sel)
            elif isinstance(value,str):
                super().__setattr__(name,self.uni.select_atoms(value))
            else:
                out=np.zeros(len(value),dtype=object)
                for k,v in enumerate(value):out[k]=v
                super().__setattr__(name,out)
            self._mdmode=_mdmode
            return
        
        super().__setattr__(name,value)
    
    def clear_sel(self):
        self._sel1=None
        self._sel2=None
        self._repr_sel=None
        self._label=None
    
    @property
    def sel1(self):
        """
        Returns the selection corresponding to the first atoms of the bonds. 
        
        Can also be used to assign sel1. Assign to either an atom group,
        a valid MDAnalysis selection string, or a list of atoms groups.

        Returns
        -------
        Atom group / list

        """
        # if self._sel1 is None:
        #     return self.uni.atoms[:0] if self._mdmode else np.zeros(0,dtype=object)
        return self._sel1
    
    @property
    def sel2(self):
        """
        Returns the selection corresponding to the second atoms of the bonds. 
        
        Can also be used to assign sel2. Assign to either an atom group,
        a valid MDAnalysis selection string, or a list of atoms groups.

        Returns
        -------
        Atom group / list

        """
        # if self._sel2 is None:
        #     return self.uni.atoms[:0] if self._mdmode else np.zeros(0,dtype=object)
        return self._sel2
            
    @property
    def project(self):
        """
        Returns the associated project if one exists

        Returns
        -------
        Project
            pyDR Project object

        """
        return self._project if self._project is not None else self.molsys.project
    
    def __copy__(self):
        cls = self.__class__
        out = cls.__new__(cls)
        out.__dict__.update(self.__dict__)
        for f in ['sel1','sel2','_repr_sel','_label']:
            setattr(out,f,copy(getattr(self,f)))
        return out
        
        
    def select_bond(self,Nuc,resids=None,segids=None,filter_str:str=None,label=None):
        """
        Bond selection tool for proteins
        
        Select a bond according to 'Nuc' keywords
            '15N','N','N15': select the H-N in the protein backbone
            'CO','13CO','CO13': select the CO in the protein backbone
            'CA','13CA','CA13': select the CA in the protein backbone
            'ivl','ivla','ch3': Select methyl groups 
                (ivl: ILE,VAL,LEU, ivla: ILE,LEU,VAL,ALA, ch3: all methyl groups)
                (e.g. ivl1: only take one bond per methyl group)
                (e.g. ivlr,ivll: Only take the left or right methyl group)
            'sidechain' : Selects a vector representative of sidechain motion
        
        Note that it is possible to provide a list of keywords for Nuc in order
        to simultaneously evaluate multiple bond types. In this case, sorting
        will be such that bonds from the same residues/segments are grouped 
        together.

        Parameters
        ----------
        Nuc : str or list, optional
            Nucleus keyword or list of keywords. 
        resids : list/array/single element, optional
            Restrict selected residues. The default is None.
        segids : list/array/single element, optional
            Restrict selected segments. The default is None.
        filter_str : str, optional
            Restricts selection to atoms selected by the provided string. String
            is applied to the MDAnalysis select_atoms function. The default is None.
        label : list/array, optional
            Manually provide a label for the selected bonds. The default is None.

        Returns
        -------
        self

        """
        
        if isinstance(Nuc,list):
            sel1,sel2=self.uni.atoms[:0],self.uni.atoms[:0] #Create empty selection
            for Nuc0 in Nuc:
                sel10,sel20=selt.protein_defaults(Nuc=Nuc0,mol=self,resids=resids,segids=segids,filter_str=filter_str)
                sel1+=sel10
                sel2+=sel20
            i=np.lexsort([sel1.names,sel2.names,sel1.resids,sel1.segids])
            self.sel1,self.sel2=sel1[i],sel2[i]
                
        else:
            self.sel1,self.sel2=selt.protein_defaults(Nuc=Nuc,mol=self,resids=resids,segids=segids,filter_str=filter_str)
        
        _mdmode=self._mdmode
        self._mdmode=False
        repr_sel=list()        
        if hasattr(Nuc,'lower') and Nuc.lower() in ['15n','n15','n','co','13co','co13']:
            for s in self.sel1:
                repr_sel.append(s.residues[0].atoms.select_atoms('name H HN N CA'))
                resi=s.residues[0].resindex-1
                if resi>=0 and self.uni.residues[resi].segid==s[0].segid:
                    repr_sel[-1]+=self.uni.residues[resi].atoms.select_atoms('name C CA O')
        elif hasattr(Nuc,'lower') and Nuc.lower()[:3] in ['ivl','ch3'] and '1' in Nuc:
            for s in self.sel1:
                repr_sel.append(s+s.residues[0].atoms.select_atoms('name H* and around 1.4 name {}'.format(s.names[0])))
        elif hasattr(Nuc,'lower') and Nuc.lower()=='cacb':
            for s in self.sel1:
                repr_sel.append(s.residues[0].atoms.select_atoms('name N CA C CB'))
        elif hasattr(Nuc,'lower') and Nuc.lower()=='sidechain':
            for s in self.sel1:
                repr_sel.append(s.residues[0].atoms.select_atoms('not name N O H HN C'))
        else:
            for s1,s2 in zip(self.sel1,self.sel2):
                repr_sel.append(s1+s2)
        self.repr_sel=repr_sel
        self.set_label(label)
        self._mdmode=_mdmode
        
        return self

    @property
    def repr_sel(self):
        if self.sel1 is None:
            print('Warning: No selection currently set')
            return 
        
        if self._repr_sel is not None and len(self.sel1)!=len(self._repr_sel):
            self._repr_sel=None
        
        if self._repr_sel is None:
            if self.sel2 is not None:
                self.repr_sel=[s0+s1 for s0,s1 in zip(self.sel1,self.sel2)]
            else:
                self.repr_sel=[s0 for s0 in self.sel1]
        return self._repr_sel
    
    @property
    def details(self):
        out=self.molsys.details
        if len(self):
            out.append('Selection with {0} elements'.format(len(self)))
            out.append('Selection labels: '+', '.join([str(l) for l in self.label]))
        return out



    def set_selection(self,i,sel=None,resids=None,segids=None,fitler_str=None):
        """
        Define sel2 and sel2 separately. The first argument is 0 or 1, depending on
        which selection is to be set. Following arguments are:
            sel: Atom group, string
        """
        pass
    
    def del_sel(self,index) -> None:
        """
        Delete a bond or bonds from a list (argument is one or a list/array of
            integers)

        Parameters
        ----------
        index : int or int array (list or numpy array)
            Indices of the selections to be removed.

        Returns
        -------
        None
            DESCRIPTION.

        """
        if hasattr(index,'__len__'):
            index=np.sort(index)[::-1]
            for i in index:self.del_sel(i)
            return
        
        for f in ['sel1','sel2','_repr_sel','_label']:
            v=getattr(self,f)
            if v is not None and len(v):
                setattr(self,f,np.delete(v,index))
                    
    
    @property
    def box(self):
        return self.molsys.uni.dimensions[:3]
    
    @property
    def uni(self):
        return self.molsys.uni
    
    @property
    def traj(self):
        return self.molsys.traj
    
    @property
    def pos(self):
        assert self.sel1 is not None,"First, perform a selection"
        if hasattr(self.sel1[0],'__len__'):
            return np.array([s.positions.mean(0) for s in self.sel1],dtype=dtype)
        else:
            return self.sel1.positions.astype(dtype)
    
    @property
    def v(self):
        assert self.sel1 is not None and self.sel2 is not None,'vec requires both sel1 and sel2 to be defined'
        if hasattr(self.sel1[0],'__len__') and hasattr(self.sel2[0],'__len__'):
            out=np.array([s0.positions.mean(0)-s1.positions.mean(0) for s0,s1 in zip(self.sel1,self.sel2)])
        elif hasattr(self.sel1[0],'__len__'):
            out=np.array([s0.positions.mean(0)-s1.position for s0,s1 in zip(self.sel1,self.sel2)])
        elif hasattr(self.sel2[0],'__len__'):
            out=np.array([s0.position-s1.positions.mean(0) for s0,s1 in zip(self.sel1,self.sel2)])
        else:
            out=self.sel1.positions-self.sel2.positions
        return pbc_corr(out.astype(dtype),self.box)
        
    def __len__(self):
        if self.sel1 is None:return 0
        return len(self.sel1)
    
    @property
    def label(self):
        if self._label is None or len(self._label)!=len(self):
            self.set_label()
        return self._label
    
    def set_label(self,label=None):
        "We attempt to generate a unique label for this selection, while having labels of minimum length"
        if label is None:
            try: #TODO kinda hack-y. Not too pleased with this.
                mdmode=self._mdmode
                self._mdmode=True
            except:
                self.label=np.arange(len(self.sel1))
                self._mdmode=mdmode
                
            "An attempt to generate a unique label under various conditions"
            if hasattr(self, "sel1") and getattr(self,"sel1") is not None\
               and hasattr(self, "sel2") and getattr(self, "sel2") is not None: #QUICKFIX
              
               if hasattr(self.sel1[0],'__len__'):
                  #todo this here is causing the saving to crash when sel1 or sel2 is None, i make a quick fix to avoid it,
                  #  but i think it needs a more detailed view
                  label=list()
                  for s in self.sel1:     #See if each atom group corresponds to a single residue
                      if s is None:
                          continue
                      elif np.unique(s.resids).__len__()==1:
                          label.append(s.resids[0])
                      else:               #If more than residue, then
                          break
                  else:
                      label=np.array(label)
                      if np.unique(label).size==label.size:   #Keep this label if all labels are unique
                          self.label=np.array(label)
                          return
                  
                  label=list()
                  for s in self.sel1:     #See if each atom group corresponds to a single segment
                      if s is None:
                          continue
                      elif np.unique(s.segids).__len__()==1:
                          label.append(s.segids[0])
                      else:               #If more than residue, then
                          break
                  else:
                      
                      label=np.array(label)
                      if np.unique(label).size==label.size:
                          self.label=np.array(label)
                          return
                  self.label=np.arange(self.sel1.__len__())
                  
               else:
                  count,cont=0,True
                  while cont:
                      if count==0: #One bond per residue- just take the residue number
                          label=self.sel1.resids  
                      elif count==1: #Multiple segments with same residue numbers
                          label=np.array(['{0}_{1}'.format(s.segid,s.resid) for s in self.sel1])
                      elif count==2: #Same segment, but multiple bonds on the same residue (include names)
                          label=np.array(['{0}_{1}_{2}'.format(s1.resid,s1.name,s2.name) for s1,s2 in zip(self.sel1,self.sel2)])
                      elif count==3: #Multiple bonds per residue, and multiple segments
                          label=np.array(['{0}_{1}_{2}_{3}'.format(s1.segid,s1.resid,s1.name,s2.name) \
                                          for s1,s2 in zip(self.sel1,self.sel2)])
                      "We give up after this"
                      count+=1
                      if np.unique(label).size==label.size or count==4:
                          cont=False
                      
                  self.label=label
            self._mdmode=mdmode
        
    def copy(self):
        return copy(self)
        
        
    def compare(self,sel,mode='auto'):
        """
        Returns indices such that
        in12: self.sel1[in12],self.sel2[in12] is in sel.sel1,sel.sel2
        in21: sel.sel1[in21],self.sel2[in21] is in self.sel1,self.sel2
        
        Then, self.sel1[in12],self.sel1[in12] and self.sel2[in21],self.sel2[in21]
        are the same set (furthermore, they are in the same order. Selection in
        sel may be resorted to achieve ordering)
        
        Several modes of comparison exist:
            exact:  Requires that both selections are based of the same topology file,
                    and each selection has the same atoms in it
            a0: Requires that segids, resids, and atom names match for all selections
            a1: Requires that resids and atom names match for all selections
            a2: Requires that segids, resids, and atom names match for sel1 
            a3: Requires that resids, and atom names match for sel1             
            auto:Toggles between a0 and exact depending if the topology files match
        """
        if self is sel:return np.arange(len(self)),np.arange(len(self)) #Same selection
        
        assert hasattr(sel,'__class__') and str(self.__class__)==str(sel.__class__),"Comparison only defined for other MolSelect objects"
        assert self.sel1 is not None,"Selection not defined in self"
        assert sel.sel1 is not None,"Selection not defined in sel1"
        assert mode in ['exact','auto','a0','a1','a2','a3'],"mode must be 'auto','exact','a0','a1','a2', or 'a3'"
        
        if self.sel2 is not None and sel.sel2 is None:return np.zeros(0,dtype=int),np.zeros(0,dtype=int)
        if self.sel2 is None and sel.sel2 is not None:return np.zeros(0,dtype=int),np.zeros(0,dtype=int)
        
        if mode=='auto':
            mode='exact' if self.uni.filename==sel.uni.filename else 'a0'
        if self.sel2 is None:
            mode='a2' if mode in ['a0','a2'] else 'a3'
        
        if mode=='exact':
            if self.sel2 is not None:
                id11,id12,id21,id22=[[np.sort(s.ids) if hasattr(s,'ids') else s.id for s in sel] \
                      for sel in [self.sel1,self.sel2,sel.sel1,sel.sel2]]
                id1=[(i0,i1) for i0,i1 in zip(id11,id12)]
                id2=[(i0,i1) for i0,i1 in zip(id21,id22)]
            else:
                id1,id2=[[np.sort(s.ids) if hasattr(s,'ids') else s.id for s in sel] \
                      for sel in [self.sel1,sel.sel1]]
        elif mode in ['a0','a1']:
            id1=list()
            id2=list()
            for id0,sel0 in zip([id1,id2],[self,sel]):
                for s1,s2 in zip(sel0.sel1,sel0.sel2):
                    if hasattr(s1,'segids') and hasattr(s2,'segids'):
                        id0.append((s1.segids,s1.resids,s1.names,s2.segids,s2.resids,s2.names))
                    elif hasattr(s1,'segids'):
                        id0.append((s1.segids,s1.resids,s1.names,s2.segid,s2.resid,s2.name))
                    elif hasattr(s2,'segids'):
                        id0.append((s1.segid,s1.resid,s1.name,s2.segids,s2.resids,s2.names))
                    else:
                        id0.append((s1.segid,s1.resid,s1.name,s2.segid,s2.resid,s2.name))
            if mode=='a1':
                id1,id2=[[(*x[1:3],*x[4:6]) for x in id0] for id0 in [id1,id2]]
        else:
            id1=list()
            id2=list()
            for id0,sel0 in zip([id1,id2],[self,sel]):
                for s1 in sel0.sel1:
                    if hasattr(s1,'segids'):
                        id0.append((s1.segids,s1.resids,s1.names))
                    else:
                        id0.append((s1.segid,s1.resid,s1.name))
            if mode=='a3':
                id1,id2=[[x[1:] for x in id0] for id0 in [id1,id2]]

        in12=list()
        for i in id1:
            in12.append(np.any([np.array_equal(np.sort(i),np.sort(i1)) for i1 in id2]))
        in12=np.argwhere(in12)[:,0]
        
        in21=np.array([(np.argwhere([np.array_equal(np.sort(id1[k]),np.sort(i2)) for i2 in id2])[0,0]) for k in in12])
            
            
        # in21=np.array([np.argwhere([id1[k]==i2 for i2 in id2])[0,0] for k in in12])
                      
        return in12,in21
    
    def __eq__(self,sel):
        if sel is self:return True
        in21=self.compare(sel,mode='auto')[1]
        return len(in21)==len(self) and np.all(in21==np.sort(in21))

    def chimera(self,color:tuple=(1.,0.,0.,1.),x=None,index=None,norm:bool=False):
        """
        Opens the molecule in chimera. One may either highlight the selection
        (optionally provide color) or one may plot some data onto the molecule,
        provided by a parameter, x. 

        Parameters
        ----------
        color : tuple, optional
            Color for the selection. The default is (1.,0.,0.,1.).
        x : TYPE, optional
            Parameter to encode onto the molecule. Length should match the length
            of the selection object. Typically, x is normalized to a maximum
            of 1, although this is not requred. The default is None.
        index : list-like, optional
            Select which elements of selection on which to plot data. Should
            have the same length as x.
            The default is None.
        norm : bool, optional
            Determine whether to renormalize x, such that it spans from 0 to 1.

        Returns
        -------
        None.

        """
        
        CMXRemote=clsDict['CMXRemote']

        if self.project is not None:
            ID=self.project.chimera.CMXid
            if ID is None:
                self.project.chimera.current=0
                ID=self.project.chimera.CMXid
        else: #Hmm....how should this work?
            ID=CMXRemote.launch()


        # CMXRemote.send_command(ID,'open "{0}" maxModels 1'.format(self.molsys.topo))
        self.molsys.chimera()
        mn=CMXRemote.valid_models(ID)[-1]
        CMXRemote.command_line(ID,'sel #{0}'.format(mn))

        CMXRemote.send_command(ID,'style sel ball')
        CMXRemote.send_command(ID,'size sel stickRadius 0.2')
        CMXRemote.send_command(ID,'size sel atomRadius 0.8')
        CMXRemote.send_command(ID,'~ribbon')
        CMXRemote.send_command(ID,'show sel')
        CMXRemote.send_command(ID,'color sel tan')
        
        
        backbone_only=True
        for rs in self.repr_sel:
            if not(backbone_only):break
            for name in rs.names:
                if name not in ['N','HN','CA','H','O','C']:
                    backbone_only=False
                    break     
        if backbone_only:
            CMXRemote.send_command(ID,'~show sel')
            CMXRemote.send_command(ID,'show sel&@N,C,CA,C,O,H,HN')
            
        CMXRemote.send_command(ID,'~sel')
        
        if index is None:index=np.arange(len(self))
        
        if np.max(color)>1:color=[float(c/255) for c in color]
        if len(color)==3:color=[*color,1.]
        
        
        if self.project is not None and self.project.chimera.saved_commands is not None:
            for cmd in self.project.chimera.saved_commands:
                CMXRemote.send_command(ID,cmd)
        
        if self.sel1 is not None:
            ids=np.concatenate([s.indices for s in self.repr_sel[index]]).astype(int)
        
        if x is None:
            if self.sel1 is not None:
                CMXRemote.show_sel(ID,ids=ids,color=color)
        else:
            assert len(x)==len(self.sel1[index]),'Length of x must match the length of the selection'
            x=np.array(x)
            if x.ndim==1:x=np.atleast_2d(x).T
            if norm:
                x-=x.min()
                x/=x.max()
            ids=np.array([s.indices for s in self.repr_sel[index]],dtype=object)
            out=dict(R=np.abs(x),rho_index=np.arange(x.shape[1]),ids=ids,color=[int(c*255) for c in color])
            CMXRemote.add_event(ID,'Detectors',out)
            
    def new_molsys_from_sel(self,sel):
        """
        Creates a new MolSys and MolSelect object using a subset of the current
        topology. 
        
        This will also create a temporary folder either in the current directory
        or the project directory, if a project directory exists, to store
        the new topology.

        Parameters
        ----------
        sel : Atom Group
            Atom group to create new molsys from.

        Returns
        -------
        MolSelect

        """
        
        _mdmode=self._mdmode
        self._mdmode=False
        
        # Check for valid selection
        if self.sel1 is not None:
            ids=np.concatenate([s.indices for s in self.sel1])
            assert np.all(np.in1d(ids,sel.indices)),\
                'All atoms in self.sel1 must be in "sel"'
        if self.sel2 is not None:
            ids=np.concatenate([s.indices for s in self.sel2])
            assert np.all(np.in1d(ids,sel.indices)),\
                'All atoms in self.sel2 must be in "sel"'
            
        ms=self.molsys.new_molsys_from_sel(sel)
        
        select=MolSelect(molsys=ms,project=self.project)
        
        
        for key in ['sel1','sel2','repr_sel']:
            new=np.zeros(len(self),dtype=object)
            if getattr(self,key) is not None:
                for k,ag in enumerate(getattr(self,key)):
                    i0=np.in1d(ag.indices,sel.indices)
                    if i0.sum():
                        index=np.digitize(ag[i0].indices,sel.indices)-1
                        new[k]=ms.uni.atoms[index]
                    else:
                        new[k]=select.sel1[k]+select.sel2[k]
                setattr(select,key,new)
        
        self._mdmode=_mdmode    
        return select

    @property
    def _hash(self):
        #TODO remove this someday
        return hash(self)

    def __hash__(self):
        x = 0
        fields = ["sel1", "sel2"]
        for field in fields:
            if hasattr(self, field) and getattr(self, field) is not None:
                f = getattr(self, field)
                if hasattr(f,"ids"):
                    x += hash(f) #<atomgroup hashing
                elif hasattr(f, "__iter__"):
                    for i,y in enumerate(f):
                        x += hash(y)*(i+1)
        return x + self.molsys._hash
        
        