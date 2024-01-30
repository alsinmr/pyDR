#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 13:18:10 2024

@author: albertsmith
"""

from .. import clsDict
import os

class MovieSys():
    def __init__(self,molsys):
        self.molsys=molsys
        self._mdlnums=None
        self._ID=None
        self.CMX=clsDict['CMXRemote']

    @property
    def project(self):
        return self.molsys.project
    
    @property
    def directory(self):
        return self.molsys.directory

    @property
    def traj(self):
        return self.molsys.traj
    
    @property
    def CMXid(self):
        """
        ID of the current ChimeraX session. If there is no session, or the 
        previous session was closed or disconnected, then this will launch
        a new ChimeraX session and update the ID. 
        
        In case of a new launch, previous mdlnums are deleted since we have lost
        access to them.

        Returns
        -------
        int
            ChimeraX session ID

        """
        if self._ID is not None and not(self.CMX.isConnected(self._ID)):
            self._ID=None
            self._mdlnums=None
            
        if self._ID is None:
            if self.project is not None:
                if self.project.chimera.CMXid is None:
                    self.project.chimera.current=0
                self._ID=self.project.chimera.CMXid
            else:
                self._ID=self.CMX.launch()
        
        return self._ID
            
    def command_line(self,cmds):
        """
        Pass commands to the current chimeraX instance

        Parameters
        ----------
        cmds : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if cmds is None:return
        if isinstance(cmds,str): cmds=[cmds]
        for cmd in cmds:            
            self.CMX.send_command(self.CMXid,cmd)
        return self
    
    @property
    def mdlnums(self):
        if self._mdlnums is not None and self._mdlnums[0] not in self.CMX.valid_models(self.CMXid):
            self._mdlnums=None
        return self._mdlnums
    
    @mdlnums.setter
    def mdlnums(self,mdlnums):
        self._mdlnums=mdlnums

    def open_pdb(self):
        """
        Opens the current pdb in ChimeraX

        Returns
        -------
        self

        """
        
        if self.valid():return self
        
        CMX=self.CMX
        ID=self.CMXid
            
        om=CMX.how_many_models(ID)
        CMX.send_command(ID,f'open "{self.molsys.topo}" coordset true')
        while om==CMX.how_many_models(ID):
            pass
        self.mdlnums=CMX.valid_models(ID)[-1],CMX.how_many_models(ID)-1
        return self
    
    def valid(self):
        """
        Checks if the current model is still open in ChimeraX, as determined
        by the number of atoms in the model. At the moment, we don't worry if
        the model is replaced by another with the same number of atoms.
        
        If the model still is valid, True is returned
        
        If molsys is not currently connected to a model, then False is returned.

        Returns
        -------
        bool

        """
        if self.mdlnums is None:
            return False
        
        n_atoms=self.CMX.how_many_atoms(self.CMXid,self.mdlnums[1])
        if n_atoms==self.molsys.uni.atoms.__len__():
            return True
        return False
        
        
    def update_traj(self):
        """
        Replaces the current coordset in Chimera with whatever trajectory is
        currently stored in the molsys object

        Returns
        -------
        None.

        """
 
        
        nm=self.mdlnums[0]
        for file in self.traj.files:self.CMX.send_command(self.CMXid,f'open "{file}" structureModel #{nm}')
        
        return self
    
    def close(self):
        """
        Closes the model previously opened by this object

        Returns
        -------
        None.

        """
        if self.mdlnums is not None:
            self.CMX.send_command(self.CMXid,f"close #{self.mdlnums[0]}")
            self.mdlnums=None
        return self
    
    def __call__(self):
        """
        Starts a movie with the current trajectory. If a model is already 
        open in Chimera, then the trajectory is only updated, but the original
        model is retained

        Returns
        -------
        self

        """
        if not(self.valid()):
            self.open_pdb()
        self.update_traj()
        return self
    
    def record(self,filename:str,framerate=15):
        """
        Plays the current trajectory and records as an mp4 file. If the filename
        is provided without an absolute path, then it will be stored in a 
        subfolder of the project folder ({project_folder}/movies).
        
        Otherwise it will be stored in the current directory

        Returns
        -------
        filepath

        """
        if filename[-4:]!='.mp4':filename+='.mp4'
        if self.project is not None and self.project.directory is not None:
            filename=os.path.join(self.project.directory,filename)
        else:
            filename=os.path.abspath(filename)
            
        mn=self.mdlnums[0]
        
        cxc=os.path.join(self.directory,'temp.cxc')
        with open(cxc,'w') as f:
            f.write('movie record\n')
            f.write(f'coordset #{mn}\n')
            f.write(f'wait {len(self.molsys.traj)}\n')
            f.write(f'movie encode "{filename}" framerate {framerate}\n')
            
        self.command_line(f'open "{cxc}"')
        return self