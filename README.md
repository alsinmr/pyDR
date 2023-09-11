# pyDR
<img src="https://raw.githubusercontent.com/alsinmr/pyDR_tutorial/8da8d5daf4d9933fc192188ad7d6d9aa14f58e33/JupyterBook/logo.png" width="500"/>

pyDR is the latest implementation of the DIFRATE method of analyzing MD and NMR data. The project should still be considered to be in "testing" phase, but at this point we hope to retain most functionality, and most functions are stable (the PCA module and solution-state sensitivities are still subject to significant changes).<br />

We will eventually publish this software as a paper, but in the meantime, please consider citing:<br />

Technical details can be found at:<br />
"Optimized “detectors” for dynamics analysis in solid-state NMR"<br />
A.A. Smith, M. Ernst, B.H. Meier<br />
https://doi.org/10.1063/1.5013316

"Reducing bias in the analysis of solution-state NMR data with dynamics detectors"<br />
A.A. Smith, M. Ernst, B.H. Meier, F. Ferrage<br />
https://doi.org/10.1063/1.5111081

"Interpreting NMR dynamic parameters via the separation of reorientational motion in MD simulation"<br />
A.A. Smith<br />
https://doi.org/10.1016/j.jmro.2022.100045

"Model-free or Not?"<br />
K. Zumpfe, A.A. Smith<br />
https://doi.org/10.3389/fmolb.2021.727553

There is no installation required for this module. Just place in a folder, navigate there (or place on the path) and run. However, python3 and the following modules are required. <br />
Python v. 3.7.3 <br />
numpy v. 1.17.2 <br />
scipy v. 1.3.0 <br />
pandas v. 0.25.1 <br />
MDAnalysis v. 0.19.2 <br />
matplotlib v. 3.0.3 <br />
pyQT5  (for GUI usage) 

Recommended (for speed in processing MD trajectories): <br />
pyFFTW <br />

We recommend installing Anaconda: https://docs.continuum.io/anaconda/install/ <br />
The Anaconda installation includes Python, numpy, scipy, pandas, and matplotlib. 

MDAnalysis is installed by running:<br />
conda config --add channels conda-forge<br />
conda install mdanalysis<br />
(https://www.mdanalysis.org/pages/installation_quick_start/)



Copyright 2022 Albert Smith-Penzel, Kai Zumpfe

All files are copyrighted under the GNU General Public License. A copy of the license has been provided in the file LICENSE

Funding for this project provided by:

Deutsche Forschungsgemeinschaft (DFG) grant 450148812

European Social Funds (ESF) and the Free State of Saxony (Junior Research Group UniDyn, Project No. SAB 100382164)
