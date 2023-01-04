#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 21 09:03:18 2022

@author: albertsmith
"""

"""
For setup, you should have the usual installed: numpy, scipy, matplotlib
Also required is MDAnalysis (https://www.mdanalysis.org/pages/installation_quick_start/)

pyfftw is recommended for speed. Some warnings may also appear recommending numba,
although this is less important and can be ignored for now.

chimeraX is recommended for visualization (https://www.cgl.ucsf.edu/chimerax/download.html)
This is NOT installed as part of your python distribution, but just using the
chimeraX pre-compiled distribution. Note that to use it, you have to once specify
the location of the executable. Interfacing with chimeraX has not been tested on
Windows.
"""

import os
import sys
path=os.path.abspath(__file__)
for _ in range(4):path=os.path.split(path)[0]
print(path)
sys.path.append(path)
import pyDR

#%% Create a project
proj=pyDR.Project('N15data',create=True)    
proj=pyDR.Project()

#%% Load the experimental data
proj.append_data('HETs_15N.txt')
"""
Text file contains both parameters for the experiments and the
experimental data itself. 

The parameter entry is sandwiched between
INFO
...
END

Each line provides a parameter followed by its values (tab separated!!)

Units are the "obvious" units...we need to document
that, but anyway MHz (1H frequency) to describe the field strength, kHz for MAS
frequency and spin-lock field strength, ppm for CSA, Hz for dipole couplings.

Some nuclei have default settings, so you can override those if you want, but
if you give a known Nuc (15N, 13CO, 13CA, 13CHD2), then you can omit these

INFO
Type    R1  R1  R1p
Nuc     15N 15N 15N
v0      600 800 600
v1      0   0   15
vr      0   0   60 
END

Next comes the data
DATA
END

One may enter fields R, Rstd, S2, S2std,label (R, Rstd required)

Each field starts with a line just with the parameter, and then followed either
by a matrix (R, Rstd) or column (S2,S2std,label)

R is a matrix (tab separated columns) where each column corresponds to the experiments
given in INFO. Each row corresponds to a separate resonance– ideally a separate
residue, but since detectors are linear, you can also take overlapping peaks
as long as you properly acquire the average rate constant (initial decay rate)

Rstd gives the standard deviation of R (determined with your favorite method), 
and should be the same size as R

S2 is a column of order parameters (S2, not S!). Same number of rows as R. (optional)

S2std is a column the same size as S2, which gives the standard deviation of S2.
Required if S2 is given

label is a data label, with the same number of rows as R. Techically, this can
be labeled however you want, but in the example here, I assume you put in the
residue numbers which match residue numbers in your topology file.

"""

d=proj[0] #Projects can be indexed in various ways, but a single integer returns the corresponding data object

d.sens.plot_rhoz(norm=True)  #This will plot the sensitivities of the experiments you included
print(d.sens.info) #See the parameters describing your experiments. 
                    #Includes default parameter values not given. Not all parameters used for all experiments
#Your data is in d.R, d.Rstd, d.S2, d.S2std, with labels in d.label
                    

"""Now we add a selection to the data object (still a little clumsy). This isn't
required, but you do need it if you want to visualize data on the structure in
chimeraX. 
"""

"""Arguments are some form of topology and a trajectory file or a list of trajectory files
See MDAnalysis requirements for acceptable file types.
In this example, I assume you've already aligned you trajectory. Note that for
experimental data, you can also just provide a pdb. In this example, however,
we'll use the same molsys to also process the md data.
"""
molsys=pyDR.MolSys(topo='backboneB.pdb',traj_files='backboneB.xtc') 

d.select=pyDR.MolSelect(molsys) #This is a selection object, which we're attaching to our data

resids=d.label #Whether this works depends on what you've included as labels. Let me know if it fails
#btw, we can handle multiple assignments in the selection objection, but it's a little trickier
d.select.select_bond(Nuc='15N',segids='B',resids=resids)


#Now we set up the detectors for fitting

d.detect.r_auto(4)  #Automatically create 4 detectors to process the data
d.detect.inclS2()   #Create a 5th detector based on inclusion of S2 (REDOR) data. Comment if no S2 included!
d.detect.plot_rhoz() #Plot the sensitivity of the detectors

f=d.fit() #Fit the experimental data (assignment to f is optional. f is automatically appended to project)
print(proj) #See that the processed data is in the project

proj['proc'].plot(style='bar') #plot all processed data to single plot (here just one data set)

"We have to tell pyDR where chimera is located (uncomment and edit path. Run just once)"
# pyDR.chimeraX.chimeraX_funs.set_chimera_path('/Applications/ChimeraX-1.2.5.app/Contents/MacOS/ChimeraX') #Uncomment and add your chimera path

proj[1].chimera(scaling=20) #Launch a chimeraX session and plot detector responses
proj.chimera.command_line('~show ~/B@N,C,CA,O,H') #Message to pass to chimera command line (also accepts list of messages)
"""Mouse over the detector labels to display a given detector.

In this case, the detector responses are rather different in amplitude. I've
set scaling to 20 to show rho1 and rho2. You can also index the detector display,
as shown below, in which case the scaling is set automatically based on the
largest detector response in the selection.
"""

# proj.chimera.close() #Close the current molecule (Can also just close the chimera session, but faster to not re-launch)
# proj[1].chimera(rho_index=[1,2])
# proj.chimera.command_line('~show ~/B@N,C,CA,H')

# proj.chimera.close()
# proj[1].chimera(rho_index=[3,4],scaling=200)
# proj.chimera.command_line('~show ~/B@N,C,CA,H')
