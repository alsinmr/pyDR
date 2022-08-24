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

proj.chimera.close() #Close the current molecule (Can also just close the chimera session, but faster to not re-launch)
proj[1].chimera(rho_index=[1,2])
proj.chimera.command_line('~show ~/B@N,C,CA,H')

proj.chimera.close()
proj[1].chimera(rho_index=[3,4],scaling=200)
proj.chimera.command_line('~show ~/B@N,C,CA,H')

"""You can also open multiple detector displays in chimera for comparison, but
sometimes the second molecule isn't displaying when it should (just type 'show'
in chimera to show all atoms)
"""

"""
A note on project indexing:
    A project can be indexed by string as follows:
    Type (print(proj.Types) to see types)
    status (proj.statuses)
    title (proj.titles)
    additional_info (proj.additional info)
    short_file (proj.short_files)
    If none of the above are matched, a string will be interpreted as a regex, applied to the title
    
    We can also use slices or a single integer:
    slices ([0:5:2], [3:], [:-2], etc.)
    Single integer ([0])
    
Note that these return subprojects, except in the last case, which returns the
data object itself. Then, you can string together indices
proj['proc']['NMR'][:2] to apply multiple filters, since the first call returns a 
subproject, and the second call then indexes the subproject again.

You can also add together subprojects (just use a + sign between two subprojects)
sub=proj['NMR']['proc']+proj['MD'][1:3]
"""

#%% Process MD data
mdsel=pyDR.MolSelect(molsys) #A separate selection object for the MDdata
mdsel.select_bond(Nuc='15N',segids='B') #Process all residues in segment B (no resids specified)

molsys.traj.step=10 #Uncomment to get faster results (lower resolution at shorter correlation times!)
from time import time
t0=time()
proj.append_data(pyDR.md2data(mdsel)) #Calculate correlation functions for HN motion. Append result to project
print(time()-t0)
#482 seconds on 2018 Mac Mini @ 3.2 GHz (slower without pyfftw installed)
#If you don't want to wait so long, uncomment molsys.step=10 above (uses 50 ps step instead of 5 ps)

proj['MD'].detect.r_no_opt(15) #15 unoptimized detectors
proj['MD'].fit(bounds=False) #Fit, don't use bounds at this step
proj.save()
"""
The above is a sort of pre-processing. It's not necessary, but saves some time later.
Basically, we process the data with an excess of 'unoptimized' detectors. These 
fit the correlation functions to high accuracy (probably too much accuracy), and
store the fitting parameters. Then, we can capture the information in the correlation
function in the 15 parameters instead of in 100000 time points. This saves disk
space, plus now you can process those 15 parameters how you want and the resulting
fitting takes a lot less time than this initial fit.

Note, you can take the result of a detector analysis and fit it again to yield
a new detector analysis. However, the error on detector responses is correlated
in such a way that stringing together multiple detector analyses to yield a
final set of detectors will give a different result than if one goes directly
from raw data to the final set of detectors. Therefore, we should go directly
to the final desired set of detectors

On the other hand, a fit resulting from unoptimized detectors (above) does not
have the same correlation of the error (this is my rough explanation...). Then,
one obtains identical results using this two-step fitting with unoptimized detectors
vs. the direct fit. 

Note, a result of all this is that when you save a project, correlation functions
are by default not saved, to save disk space.
"""
proj.close_fig('all') #Close all the figures
proj['MD']['no_opt'].plot() #Show the unoptimized detectors (a giant mess of data!)


target=proj['NMR']['proc'].sens.rhoz #We'll try to make the MD detectors match the experimental detectors
target[0,99:]=0 #I zero out the rho0 sensitivity at long correlation times, since most motion in this detector comes from librations

"""
To match MD detector sensitivities to NMR, we have to use more detectors in the
MD than in the NMR. These are just sort of 'dummy' detectors, but they do influence
the quality of the back-calculation of the raw data.

We use r_target for matching a specific functional form, instead of r_auto
"""
proj=pyDR.Project('N15data/')
proj['MD']['no_opt'].detect.r_target(target=target,n=8)
#Plot MD detectors compared to the original experimental detectors.
#Failure to match microsecond detectors is expected, because we only have 500 ns simulation
ax=proj['MD']['no_opt'][0].detect.plot_rhoz(index=range(5))[0].axes
ax.plot(proj['NMR']['proc'].sens.z,proj['NMR']['proc'].sens.rhoz.T,color='black',linestyle=':')

proj['MD']['no_opt'].fit(bounds=False)
proj['MD']['proc'].opt2dist(rhoz_cleanup=True)
"""opt2dist forces a fit to be consistent with a normalized distribution of correlation times.
That is, we actually calculate a distribution to fit the set of detector responses. Then, 
from the fitted distribution, we back-calculate the responses and report those. If 
rhoz_cleanup=True, we also clean the shape of the detector shape and use the new
detectors for back calculation. 

The fitting basically takes care of noise in the correlation function that can
lead to a set of detector responses that aren't consistent with a multi-exponential
correlaiton function. This in itself shouldn't add bias, although if the 
C(t) isn't totally equilibrated, that is more problematic

rhoz_cleanup does not seem to introduce problems, but I'm less convinced. Still,
r_target introduces negative regions to the sensitivities, which can result in
negative detector responses. Technically, those negative responses aren't wrong,
but they are difficult to interpret, so that's why we have this option.
"""

#Compare experimental results vs. simulation
proj.close_fig('all') #You can also assign a new figure number (e.g. fig=2) in the next line if you don't want to close the current figures
proj['NMR']['proc'].plot(color='black')
proj['MD']['opt_fit'].plot(style='bar')

"""Plotting multiple data sets automatically omits detectors that don't match
up with the initial data set's sensitivities (imperfect agreement is allowed).
In this case, detectors 0-3 are plotted but 4 disagress too much and is omitted.

HOwever, rho3 from MD is going to be highly unreliable because the detector 
sensitivity extends to infinite correlation times, thus a lot of the MD-derived
detector response is contributions from S2.
"""
