
import pyDR
import numpy as np


proj=pyDR.Project('/Users/albertsmith/Documents/Dynamics/HETs_Methyl_Loquet/dummy/',create=True)

pdb = "/Volumes/My Book/HETs/HETs_3chain.pdb"
xtc = "/Volumes/My Book/HETs/MDSimulation/HETs_MET_4pw.xtc"
molsys = pyDR.Selection.MolSys.MolSys(pdb, xtc, step =10)
molsel = pyDR.Selection.MolSys.MolSelect(molsys)
molsel.uni.residues.resids=np.tile(np.arange(219,290),3)

resids=np.concatenate([np.arange(224,250),np.arange(224+35,250+35)])
molsel.select_bond(Nuc="15N",resids=resids)


frames=pyDR.Frames.FrameObj(molsel)
ired=frames.md2iRED()
proj.append_data(ired.iRED2data())
proj.append_data(frames.md2data())
proj.detect.r_auto(7)
proj.fit()
proj['proc']['iREDmode'].modes2bonds()


proj['proc']['MD'].plot(style='bars')
proj['proc']['iREDbond'].plot()

proj['iREDbond'].plot_CC('all')