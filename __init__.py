#!/usr/bin/env python3
# -*- coding: utf-8 -*-


#%% Init file for all of pyDR

clsDict=dict()
from pyDR.Defaults import Defaults

from pyDR import Selection
from pyDR.Selection.MolSys import MolSys,MolSelect
from pyDR.iRED.iRED import iRED
#from pyDR import FRET
from pyDR.Data import Data,Plotting
from pyDR import Fitting
from pyDR import MDtools
from pyDR import Sens
from pyDR.misc.tools import tools
from pyDR import Frames
from pyDR import IO
from pyDR.chimeraX.CMXRemote import CMXRemote

from pyDR import Project

clsDict.update({'Data':Data,'Source':Project.Source,'Info':Sens.Info,
         'Sens':Sens.Sens,'Detector':Sens.Detector,'NMR':Sens.NMR,'MD':Sens.MD,
         'MolSys':MolSys,'MolSelect':MolSelect,'Project':Project.Project,
         'FrameObj':Frames.FrameObj,'Ctcalc':MDtools.Ctcalc,
         'DataPlots':Plotting.DataPlots,'CMXRemote':CMXRemote})