# -*- coding: utf-8 -*-

import numpy as np
from .. import clsDict

sel_names={'ARG':['CZ','NE'],'HIS':['CG','CB'],'HSD':['CG','CB'],'LYS':['NZ','CE'],
           'ASP':['CG','CB'],'GLU':['CD','CG'],'SER':['OG','CB'],'THR':['CG2','CB'],
           'ASN':['OD1','CG'],'GLN':['OE1','CD'],'CYS':['SG','CB'],'VAL':['CB','CA'],
           'ILE':['CD','CG1'],'LEU':['CD1','CG'],'MET':['CE','SD'],'PHE':['CG','CB'],
           'TYR':['CG','CB'],'TRP':['CG','CB']}

class StateCounter:
    def __init__(self,select):
        self.select=select
        
        self.reset()
        
        
    def reset(self):
        """
        Sets various fields back to unitialized state

        Returns
        -------
        self

        """
        self._resi=None
        self._sel1=None
        self._sel2=None
        self._included=None
        self._FrObj=None
        
        return self
        
    @property
    def resi(self):
        if self._resi is None:
            if self.select.sel1 is None:return None
            
            self._resi=[]
            for s in self.select.sel1:
                self._resi.append(s.residues[0])
                
            self._resi=np.array(self._resi,dtype=object)
            
        return self._resi
                
    
    @property
    def sel1(self):
        """
        Returns the bond selection (sel1) for defining the rotameric state

        Returns
        -------
        numpy array

        """
        if self._sel1 is None:
            if self._resi is None:return None
            self._sel1=[]
            self._sel2=[]
            self._included=[]
            for res in self.resi:
                self._included.append(False)
                if res.resname in sel_names:
                    q=sel_names[res.resname]
                    if q[0] in res.atoms.names and q[1] in res.atoms.names:
                        self._included[-1]=True
                        self._sel1.append(res.atoms[res.atoms.names==q[0]])
                        self._sel2.append(res.atoms[res.atoms.names==q[1]])
                    else:
                        print(f'Residue {res.resname} found in topology, but one or more atoms ({q[0],q[1]}) were not found')
                        
            self._sel1=np.sum(self._sel1)
            self._sel2=np.sum(self._sel2)
            self._included=np.array(self._included,dtype=bool)
            
        return self._sel1
                         
    @property
    def sel2(self):
        """
        Returns the bond selection (sel1) for defining the rotameric state

        Returns
        -------
        numpy array

        """
        if self._sel2 is None:
            if self.sel1 is None:return None
        return self._sel2
    
    @property
    def included(self):
        """
        Returns a boolean index to indicate which residues were included in the
        entropy calculation
        
        e.g. Alanine, glycine, proline have fixed entropies and are therefore
        omitted. Unrecognized residues are also not included

        Returns
        -------
        boolean array

        """
        if self._included is None:
            if self.sel1 is None:return None
        return self._included
    
    @property
    def FrObj(self):
        """
        Returns the frame object for the side chain rotamers

        Returns
        -------
        None.

        """
        if self._FrObj is None:
            sel=clsDict['MolSelect'](self.select.molsys)
            sel._mdmode=True
            sel.sel1=self.sel1
            sel.sel2=self.sel2
            self._FrObj=clsDict['FrameObj'](sel)
            self._FrObj.tensor_frame(sel1=1,sel2=2)
            
        return self._FrObj
            
                
            