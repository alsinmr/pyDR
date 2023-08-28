# -*- coding: utf-8 -*-

import numpy as np



def color2hex(color):
    color=np.array(color)[:3]
    assert np.all(color<=1) and np.all(color>=0),\
        "Color should be between 0 and 1"
    return '0x'+(''.join('%02x'%i for i in np.uint8(color*255))).upper()
    

def color_calc(x,x0=None,colors=[[0,0,1],[0.82352941, 0.70588235, 0.54901961],[1,0,0]]):
    """
    Calculates color values for a list of values in x (x ranges from 0 to 1).
    
    These values are linear combinations of reference values provided in colors.
    We provide a list of N colors, and a list of N x0 values (if x0 is not provided,
    it is set to x0=np.linspace(0,1,N). If x is between the 0th and 1st values
    of x0, then the color is somewhere in between the first and second color 
    provided. Default colors are blue at x=0, tan at x=0.5, and red at x=1.
    
    color_calc(x,x0=None,colors=[[0,0,255,255],[210,180,140,255],[255,0,0,255]])
    """
    
    colors=np.array(colors)
    N=len(colors)
    if x0 is None:x0=np.linspace(0,1,N)
    x=np.atleast_1d(x)
    if x.min()<x0.min():
        x[x<x0.min()]=x0.min()
    if x.max()>x0.max():
        x[x>x0.max()]=x0.max()

    i=np.digitize(x,x0)
    i[i==len(x0)]=len(x0)-1
    clr=(((x-x0[i-1])*colors[i].T+(x0[i]-x)*colors[i-1].T)/(x0[i]-x0[i-1])).T
    return color2hex(clr[0])

def hex2list(hex_str):
    if len(hex_str)==7:hex_str=hex_str[1:]
    return [int(hex_str[k:k+2],base=16)/256 for k in range(0,6,2)]

class NglPlot():
    def __init__(self,repr_sel,x=None,color:tuple=(1.,0.,0.),norm:bool=False,sel_str:str=None):
        """
        Class for coloring data onto a molecule in NGLviewer.
        
        Note that changing repr_sel, x requires reinitialization.

        Parameters
        ----------
        repr_sel : TYPE
            List of atom groups to be colored/scaled.
        x : TYPE, optional
            List of values to give the color encoding (same length as repr_sel)
        color : tuple, optional
            Color corresponding to x=1. The default is (1.,0.,0.,1.).
            Default for zero is tan: [0.8203125 , 0.703125  , 0.546875  , 0.99609375]
        norm : bool, optional
            Normalize x to 1 before calculations. The default is False.
        sel_str : str, optional
            String (NGLViewer style) of base representaton. The default is None,
            which will show all segments found in repr_sel. Only the backbone
            is displayed if no non-backbone atoms found in repr_sel.

        Returns
        -------
        None.

        """
        
        self.zero_color=[0.8203125 , 0.703125  , 0.546875]
        
        self.repr_sel=repr_sel
        self.x=x
        self.color=color
        self.norm=norm
        self.sel_str=sel_str
        self.colorID=None
        
        # self._all_atoms=None
        # self._xavg=None
        
    def __setattr__(self,name,value):
        """
        Reset calculated attributes if parameters are changed

        Returns
        -------
        None.

        """
        super().__setattr__('_all_atoms',None)
        super().__setattr__('_xavg',None)
        super().__setattr__(name,value)
    
        
    @property
    def all_atoms(self):
        """
        Returns an atom group with all 

        Returns
        -------
        None.

        """
        if self._all_atoms is None:
            self._all_atoms=np.sum(np.unique(np.sum(self.repr_sel)))
        return self._all_atoms
    
    @property
    def segments(self) -> str:
        """
        Returns a string for selecting segments found in repr_sel

        Returns
        -------
        str.

        """
        

        segs=np.unique(self.all_atoms.segids)
        return ':'+','.join(*segs) if len(segs) else ''
    
    @property
    def atom_names(self) -> list:
        """
        Returns a list of atom names found in repr_sel

        Returns
        -------
        list
            all atoms found in repr_sel

        """
        
        return np.unique(self.all_atoms.names)
    
    @property
    def backbone_only(self) -> bool:
        """
        Returns True if all atoms are in the protein backbone (C,CA,H,N,O)

        Returns
        -------
        bool
            Are all atoms backbone atoms

        """
        
        return np.all([name in ['C','CA','H','HN','N','O'] for name in self.atom_names])
        
    @property
    def base_repr(self) -> dict:
        """
        Returns a dictionary for the base representation, e.g. the molecule 
        backbone.

        Returns
        -------
        dict

        """
        
        if self.sel_str is not None:
            sele=self.sel_str
        else:
            if self.backbone_only:
                atoms=np.unique(np.concatenate((['C','CA','N'],self.atom_names)))
            else:
                atoms=np.unique(self.all_atoms.universe.atoms.names)
            
            sele=' or '.join(self.segments+f'.{a}/0' for a in atoms)

                    
        # return {"type":"ball+stick","params":{"sele":sele,"color":color2hex(self.zero_color),
        #                                    "aspectRatio":1}}
        
        if self.colorID is None:self.color_JS()
        
        return {"type":"ball+stick","params":{"sele":sele,"color":self.colorID,
                                   "radius":"bfactor"}}
    
    @property
    def xavg(self):
        """
        Average values of x for all atoms found in repr_sel

        Returns
        -------
        None.

        """
        
        norm=1/np.max(self.x) if self.norm else 1
        
        if self._xavg is None:
            x=np.zeros(len(self.all_atoms))
            count=np.zeros(len(self.all_atoms))
            
            for rs,x0 in zip(self.repr_sel,self.x):
                i=np.isin(self.all_atoms,rs)
                count[i]+=1
                x[i]+=x0*norm
                
            self._xavg=x/count
        return self._xavg
    
    @property
    def atom_reprs(self):
        """
        List of representations for each atom

        Returns
        -------
        rep : list
            List of dictionaries for each atom and its color/aspectRatio.

        """
        rep=[]
        for atom,xavg in zip(self.all_atoms,self.xavg):
            color=color_calc(x=np.abs(xavg),colors=[self.zero_color,self.color])
            rep.append({"type":"ball+stick",
                        "params":{"sele":f"@{atom.index}/0",
                                  "color":color,"aspectRatio":10*np.abs(xavg)}})
        return rep
    
    # @property
    # def atom_reprs(self):
    #     if self.colorID is None:self.color_JS()
    #     return [{"type":"ball+stick","params":{"sele":' or '\
    #                 .join(f"@{atom.index}/0" for atom in self.all_atoms),
    #                 "color":self.colorID,"radius":"bfactor"}}]
        
    @property
    def representations(self):
        """
        Base represenation and atom representations

        Returns
        -------
        None.

        """
        out=[self.base_repr]
        out.extend(self.atom_reprs)
        return out
    
    def color_JS(self):
        """
        Creates the javascript code for a color and adds it to the Colormaker
        Registry

        Returns
        -------
        None.

        """
        if self.colorID is None:
            
            JScode="""this.atomColor = function (atom) {
  if """
            for atom,x, in zip(self.all_atoms,self.xavg):
                color=color_calc(x,colors=[self.zero_color,self.color])
                index=atom.index+1
                JScode+=f" (atom.serial == {index})"+"""{
  return """+f"{color}"+"""
  } else if """
            
            color=color_calc(0,colors=[self.zero_color,self.zero_color])
            JScode=JScode[:-3]
            JScode+=""" {
  return """ + f"{color}" + """
  }
 }
 """
            
            from nglview.color import ColormakerRegistry as cm
            
            self.colorID=f'color{np.random.randint(10000)}'
            self.colorID='color1'
            cm.add_scheme_func(self.colorID,JScode)
            self._JScode=JScode
        
        return self._JScode
        
        
    
    def __call__(self):
        """
        Returns the NGLview view object

        Returns
        -------
        None.

        """
        import nglview as nv
        
        self.color_JS()
        self.all_atoms.universe.atoms.tempfactors=.1
        self.all_atoms.tempfactors=self.xavg*.4+.1
        
        # return nv.show_mdanalysis(self.all_atoms.universe,representations=self.representations)
        
        return nv.show_mdanalysis(self.all_atoms.universe,representations=[self.base_repr])
        
        # v=nv.show_mdanalysis(self.all_atoms.universe,representations=[self.base_repr])
        
        # for atom,x in zip(self.all_atoms,self.xavg):
        #     color=hex2list(color_calc(np.abs(x),colors=[self.zero_color,self.color]))
        #     pos=atom.position.tolist()
        #     v.shape.add_sphere(pos,color,x)
            
        # return v
            
        
        
        
    