#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%% Init file for all of pyDR

clsDict=dict()
from pyDR.Defaults import Defaults

from pyDR import Selection
from pyDR.Selection.MolSys import MolSys,MolSelect

#from pyDR import FRET
from pyDR.Data import Data,Plotting
from pyDR import Fitting
from pyDR import MDtools
from pyDR import Sens
from pyDR.misc.tools import tools
import sys as _sys
if 'MDAnalysis' in _sys.modules:
    from pyDR import Frames
else:
    print('MDAnalysis not available, MD processing will not be possible')
from pyDR import IO
from pyDR.iRED.iRED import iRED, Data_iRED

if 'MDAnalysis' in _sys.modules:
    from pyDR.Frames.eval_fr import md2data,md2iRED
from pyDR.chimeraX.CMXRemote import CMXRemote
from pyDR.chimeraX.Movies import Movies

from pyDR.Project import Project,Source

clsDict.update({'Data':Data,'Data_iRED':Data_iRED,'Source':Source,'Info':Sens.Info,
         'Sens':Sens.Sens,'Detector':Sens.Detector,'NMR':Sens.NMR,'MD':Sens.MD,'SolnNMR':Sens.SolnNMR,
         'MolSys':MolSys,'MolSelect':MolSelect,'Project':Project,
         'FrameObj':Frames.FrameObj,'Ctcalc':MDtools.Ctcalc,
         'DataPlots':Plotting.DataPlots,'CMXRemote':CMXRemote,'Movies':Movies})



from matplotlib.axes import Subplot as _Subplot
from matplotlib.gridspec import SubplotSpec as _SubplotSpec
if hasattr(_SubplotSpec,'is_first_col'):
    def _fun(self):
        return self.get_subplotspec().is_first_col()
    _Subplot.is_first_col=_fun
    def _fun(self):
        return self.get_subplotspec().is_first_row()
    _Subplot.is_first_row=_fun
    def _fun(self):
        return self.get_subplotspec().is_last_col()
    _Subplot.is_last_col=_fun
    def _fun(self):
        return self.get_subplotspec().is_last_row()
    _Subplot.is_last_row=_fun