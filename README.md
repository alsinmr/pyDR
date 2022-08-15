# pyDR
pyDR is the latest implementation of the DIFRATE method of analyzing MD and NMR data. At the moment, the project is under development and is unlikely to function stably except in limited cases. 

Technical details can be found at:<br />
"Optimized “detectors” for dynamics analysis in solid-state NMR"<br />
A.A. Smith, M. Ernst, B.H. Meier<br />
https://doi.org/10.1063/1.5013316

"Reducing bias in the analysis of solution-state NMR data with dynamics detectors"<br />
A.A. Smith, M. Ernst, B.H. Meier, F. Ferrage<br />
https://doi.org/10.1063/1.5111081

"Interpreting NMR dynamic parameters via the separation of reorientational motion in MD simulation"<br />
A.A. Smith<br />
https://arxiv.org/abs/2111.09224

There is no installation required for this module. Just place in a folder, navigate there and run. However, python3 and the following modules are required. <br />
Python v. 3.7.3 <br />
numpy v. 1.17.2 <br />
scipy v. 1.3.0 <br />
pandas v. 0.25.1 <br />
MDAnalysis v. 0.19.2 <br />
matplotlib v. 3.0.3 <br />
pyQT5  (for GUI usage) 

Recommended (for speed in processing MD trajectories): <br />
pyFFTW <br />
numba <br />

We recommend installing Anaconda: https://docs.continuum.io/anaconda/install/ <br />
The Anaconda installation includes Python, numpy, scipy, pandas, and matplotlib. 

MDAnalysis is installed by running:<br />
conda config --add channels conda-forge<br />
conda install mdanalysis<br />
(https://www.mdanalysis.org/pages/installation_quick_start/)


run the GUI with 'python3 -m pyDR'



Copyright 2022 Albert Smith-Penzel, Kai Zumpfe

All files are copyrighted under the GNU General Public License. A copy of the license has been provided in the file LICENSE

Funding for this project provided by:

Deutsche Forschungsgemeinschaft (DFG) grant SM 576/1-1

European Social Funds (ESF) and the Free State of Saxony (Junior Research Group UniDyn, Project No. SAB 100382164)
