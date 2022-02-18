# pyDR
pyDR is the latest implementation of the DIFRATE method of analyzing MD and NMR data. At the moment, the project is under development and is unlikely to function stably except in limited cases. 

Technical details can be found at:
"Optimized “detectors” for dynamics analysis in solid-state NMR"
A.A. Smith, M. Ernst, B.H. Meier
https://doi.org/10.1063/1.5013316

"Reducing bias in the analysis of solution-state NMR data with dynamics detectors"
A.A. Smith, M. Ernst, B.H. Meier, F. Ferrage
https://doi.org/10.1063/1.5111081

"Interpreting NMR dynamic parameters via the separation of reorientational motion in MD simulation"
A.A. Smith
https://arxiv.org/abs/2111.09224

There is no installation required for this module. Just place in a folder, navigate there and run. However, python3 and the following modules are required. 

Python v. 3.7.3 <br />
numpy v. 1.17.2
scipy v. 1.3.0,
pandas v. 0.25.1
MDAnalysis v. 0.19.2
matplotlib v. 3.0.3
pyQT5  (for GUI usage) 

Recommended (for speed in processing MD trajectories):
pyFFTW
numba

We recommend installing Anaconda: https://docs.continuum.io/anaconda/install/
The Anaconda installation includes Python, numpy, scipy, pandas, and matplotlib. 

MDAnalysis is installed by running:
conda config --add channels conda-forge
conda install mdanalysis
(https://www.mdanalysis.org/pages/installation_quick_start/)


All files are copyrighted under the GNU General Public License. A copy of the license has been provided in the file LICENSE


Copyright 2022 Albert Smith-Penzel, Kai Zumpfe
