#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Copyright 2021 Albert Smith-Penzel

This file is part of Frames Theory Archive (FTA).

FTA is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

FTA is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with FTA.  If not, see <https://www.gnu.org/licenses/>.


Questions, contact me at:
albert.smith-penzel@medizin.uni-leipzig.de


Created on Thu Sep 24 14:52:56 2020

@author: albertsmith
"""


"""
This script generates an ideal trajectory (Fig. 5) with four motions:
    1) Fast wobbling-in-a-cone
    2) Three-site hopping
    3) Two-site hopping
    4) Isotropic tumbling
    
We take a step-by-step construction first of the motion, based on random
number generation, followed by deconstruction of the total motion using the frame
analysis.

"""
import numpy as np
import matplotlib.pyplot as plt
from pyDR.Frames import eval_fr as ef
from pyDR.MDtools import vft


#%% Simulation Parameters
n=int(1e6)          #Number of time points in simulated trajectory (reduce for faster calculations)
dt=5e-3             #Time step (ns)
t=np.arange(n)*dt   #Time axis (10 us)
t[0]=1e-3           #We plot on a log scale. We replace 0 with 1 ps to keep it on the scale (labeling later indicates this)

tau_f=0             #Fast motion (immediate decay)
tau_m=0.1           #Methyl rotation (hop time- not decay time constant of the correlation function)
tau_h=1             #Two-site hop time
dr=0.5              #Step size (in degrees) for rotation of the molecule in solution (determines correlation time)

theta_f=15          #Opening angle of fast motion (wobbling in a cone)
theta_m=30          #Symmetry axis for "methyl hop"
theta_h=45          #Angle of two-site hop



#%% Transistions
"state_m is a list of 0,1,2, indicating the state of the 3-site hopping (transitions are Poisson distributed)"
state_m=np.mod((np.random.binomial(1,dt/tau_m,n)*(1-2*np.random.binomial(1,0.5,n))).cumsum(),3).astype(int)
"state_h is a list of 0,1, indicating the state of 2-site hopping (Poisson distributed)"
state_h=np.mod(np.random.binomial(1,dt/tau_h,n).cumsum(),2).astype(int)

"gamma angle for fast, wobbling-in-a-cone motion"
gamma_f=np.random.rand(n)*2*np.pi
"beta angle for fast, wobbling-in-a-cone motion"
br=np.random.rand(n)    #First a random number on the range 0-1
#We map that random number to the beta angle. This second step is required because we need larger angles to be more likely
beta_f=np.arccos(1-br+br*np.cos(theta_f*np.pi/180))

"Random direction for step in tumbling in solution"
gamma_r=np.random.rand(n)*2*np.pi



#%% Build up the trajectory
"Direction of vector resulting from fast wobbling-in-a-cone"
vf=np.array([np.sin(beta_f)*np.cos(gamma_f),np.sin(beta_f)*np.sin(gamma_f),np.cos(beta_f)])

"Three vectors for possible directions resulting from 3-site hopping"
vm0=[np.sin(theta_m*np.pi/180),0,np.cos(theta_m*np.pi/180)]
vm1=[np.sin(theta_m*np.pi/180)*np.cos(2*np.pi/3),np.sin(theta_m*np.pi/180)*np.sin(2*np.pi/3),np.cos(theta_m*np.pi/180)]
vm2=[np.sin(theta_m*np.pi/180)*np.cos(-2*np.pi/3),np.sin(theta_m*np.pi/180)*np.sin(-2*np.pi/3),np.cos(theta_m*np.pi/180)]
"Vectors for 3-site hopping motion (index the three vectors above with state_m)"
vm=np.array([vm0,vm1,vm2]).T[:,state_m]
vmXZ=np.array([vm1,vm2,vm0]).T[:,state_m]

"Two vectors for possible directions resulting from 2-site hopping"
vh0=[0,0,1]
vh1=[np.sin(theta_h*np.pi/180),0,np.cos(theta_h*np.pi/180)]
"Vectors for 2-site hopping motion (index the twovectors above with state_h)"
vh=np.array([vh0,vh1]).T[:,state_h]

"Here we construct tumbling in solution"
vr=[np.array([0,0,1])]  #Start out with vector pointing along z
for k in range(n-1):    #Loop over all time points
    sc=vft.getFrame(vr[-1])     #Get frame of current vector direction
    """Below is the direction of the vector due to a step of dr degrees (see simulation parameters),
    occuring at an angle of gamma_r[k] (given in the frame of the current vector direction)
    """
    vr0=[np.cos(gamma_r[k])*np.sin(dr*np.pi/180),np.sin(gamma_r[k])*np.sin(dr*np.pi/180),np.cos(dr*np.pi/180)]
    "We rotate vr0 into the lab frame frome the frame of the vector direcion, and append this angle"
    vr.append(vft.R(vr0,*sc))
vr=np.array(vr).T #Convert into a numpy array

"Here the full set of motions"
scf=vft.getFrame(vf)  #Time-dependent angles for fast wobbling-in-a-cone motion
scm=vft.getFrame(vm)  #Time-dependent angles for 3-site hopping    
sch=vft.getFrame(vh)  #Time-dependent angles for 2-site hopping
scr=vft.getFrame(vr)  #Time-dependent angles for tumbling in solution

v=vft.R(vft.R(vft.R(vf,*scm),*sch),*scr) #Total motion a product of all four rotations (vft.R applies a rotation to a vector)
vXZ=vft.R(vft.R(vmXZ,*sch),*scr)    #v (above) defines z-axis, vXZ defines the XZ plane


#%% Get each vector in the lab frame
"""Below, we take vectors for each separate motion and rotate them into the lab frame
This is done by rotating each vector by all outer motions
"""
vf_LF=vft.R(vft.R(vft.R(vf,*scm),*sch),*scr)
vm_LF=vft.R(vft.R(vm,*sch),*scr)
vh_LF=vft.R(vh,*scr)
vr_LF=vr



#%% Try to reconstruct correlation function assuming timescale separation
"""
Below, we use the function Ct_D2inf, which is responsible for calculating components
of various correlation values and their equilibrium values

Usage is as follows
Ct,A = ef.Ct_D2inf(v,vXZ,nuZ_F,nuXZ_F,nuZ_f,nuXZ_f,cmpt,mode)

v,vXZ:          Defines the Z-axis and XZ-plane of the frame of the NMR tensor
nuZ_F,nuXZ_F:   Defines the Z-axis and XZ-plane of some outer frame, F, which we
                use to remove all motion of frame F. Optional- omitting will
                result in no outer motion being removed, i.e. calculating in
                lab frame)
nuZ_f,nuXZ_f:   Defines the Z-axis and XZ-plane of some inner frame, f, where we
                calculate the components of the correlation function resulting 
                from motion of f acting on the NMR tensor (where motion of F
                is removed if nuZ_F is provided). nuZ_f, nuXZ_f are optional, 
                and if omitted, calculation will return components of the 
                innermost motion, that is, the motion of the NMR tensor moving
                within frame F (or in the lab frame, if F is also omitted)
cmpt:           Choose which components of the correlation function and 
                equilibrium values to return. Provided as a string, with options
                being a specific component ('00','20','-10', etc.), or a range
                of components ('0p','p0', where the former returns a list of
                5 correlation functions, '0-2','0-1','00','01','02'). Note that
                if all 5 components are required, the '0p' or 'p0' option is much
                faster than looping over this function, since we may recycle 
                many of the calculations
mode:           String, specifying 'ct','D2inf', or 'both', where 'ct' returns
                only the correlation function, 'D2inf' only returns the equilibrium
                value of the correlation function, and 'both' returns both (ct 
                first, A, the equilibrium values, second). Again, running this
                function twice to get each parameter is slower than acquiring
                both simultaneously.
"""


"First we calculate total correlation function"
"No inner or outer frames, and only the '00' component required"
Ct=ef.Ct_D2inf(v,cmpt='00',mode='ct') 

"Construction correlation function from product of individual motions"

"""
Inner motion (librations).
Af is the residual tensor of fast motion, and Ct0pf
contains the correlation functions describing the evolution of the tensor
due to fast motion. Af is used in the subsequent step for calculating the 
correlation function for 3-site hopping.
"""
Ct0pf,Af=ef.Ct_D2inf(v,vXZ,nuZ_F=vm_LF,cmpt='0p')
"""
Ctp0f contains the 5 components of the correlation function for fast motions,
although we actually only need the 00 component
"""
Ctp0f=ef.Ct_D2inf(v,vXZ,nuZ_F=vm_LF,cmpt='p0',mode='ct')
"""
For the innermost motion, we only require the 00 component of the correlation function
"""
Ct_f=Ct0pf[2].real


"""
Intermediate motion (3-site hopping)
Am is the residual tensor of 3-site hopping and fast motion, and Ct0pm describes
the time evolution towards this residual tensor. Am is required in the subsequent
step for calculating the correlation function for 2-site hopping.
"""
Ct0pm,Am=ef.Ct_D2inf(v,vXZ,nuZ_F=vh_LF,cmpt='0p')
"""
Ctp0m contains the 5 components of the correlation function for 3-site hopping
"""
Ctp0m=ef.Ct_D2inf(v,vXZ,nuZ_F=vh_LF,nuZ_f=vm_LF,cmpt='p0',mode='ct')
"""
The correlation function for 3-site hopping is the sum of the products of the
5 components of the residual tensor from the fast motion and the 5 components of 
the correlation function for 3-site hopping
"""
Ct_m=np.sum(Af*Ctp0m.T,1).T.real/Af[2].real


"""
Intermediate motion (2-site hopping)
Ah is the residual tensor of 2-site hopping, 3-site hopping, and fast motion, 
and Ct0ph describes the time evolution towards this residual tensor. Ah is 
required in the subsequent step for calculating the correlation function for 
isotropic tumbling.
"""
Ct0ph,Ah=ef.Ct_D2inf(v,vXZ,nuZ_F=vr_LF,cmpt='0p')
"""
Ctp0h contains the 5 components of the correlation function for 2-site hopping
"""
Ctp0h=ef.Ct_D2inf(v,vXZ,nuZ_F=vr_LF,nuZ_f=vh_LF,cmpt='p0',mode='ct')
"""
The correlation function for 2-site hopping is the sum of the products of the
5 components of the residual tensor from the fast motion+3-site hopping and 
the 5 components of the correlation function for 2-site hopping
"""
Ct_h=np.sum(Am*Ctp0h.T,1).T.real/Am[2].real

"""
Outer motion (tumbling)
Ar is the residual tensor of tumbling 2-site hopping, 3-site hopping, and fast
motion (we don't actually need this, or plot it), and Ct0pr describes the time
evolution towards this residual tensor. 
"""
Ct0pr,Ar=ef.Ct_D2inf(v,vXZ,cmpt='0p')
"""
Ctp0r contains the 5 components of the correlation function for tumbling
"""
Ctp0r=ef.Ct_D2inf(v,vXZ,nuZ_f=vr_LF,cmpt='p0',mode='ct')
"""
The correlation function for tumbling is the sum of the products of the
5 components of the residual tensor from the fast motion+3-site hopping+2-site
hopping and the 5 components of the correlation function for tumbling
"""
Ct_r=np.sum(Ah*Ctp0r.T,1).T.real/Ah[2].real


"""
The total correlation function should be approximately equal to the product of
the correlation functions for the individual motions
"""
Ct_p=Ct_f*Ct_m*Ct_h*Ct_r




#%% Plot all correlation functions
"Create axes, store titles, text, legends for later"
fig=plt.figure('Ideal Frame Separation')
fig.clear()
n2=int(n/2)

ax=[fig.add_subplot(5,2,k) for k in [1,2,3,6,5,8,7,10,9]]

titles=['Total Correlation Function',
        'Components',
        'C(t) for f in F',
        'C(t) for tensor PAS in f',
        '','','','','']
legends=[[r'$C(t)$',r'$C_\mathrm{prod}(t)$'],
         [r'$C_\mathrm{f}(t)$',r'$C_\mathrm{hop}(t)$',r'$C_\mathrm{met}(t)$',r'$C_\mathrm{f}(t)$'],
         [r'$C_{00}(t)$',r'Re[$C_{10}(t)$]',r'Im[$C_{10}(t)$]',r'Re[$C_{20}(t)$]',r'Im[$C_{20}(t)$]'],
         [r'$C_{00}(t)$',r'Re[$C_{01}(t)$]',r'Im[$C_{01}(t)$]',r'Re[$C_{02}(t)$]',r'Im[$C_{02}(t)$]'],
         '','','','','']
text=['','',
      'PAS in ME\n(fast wobbling)',
      'PAS in ME',
      'ME in CC\n(rotation)',
      'PAS in CC',
      'CC in MO\n(hopping)',
      'PAS in MO',
      'MO in LF\n(tumbling)']


"Plot total correlation function"
ax[0].semilogx(t[:n2],Ct[:n2],color='red')
ax[0].semilogx(t[:n2],Ct_p[:n2],color='black',linestyle=':')


    
"Plot components of correlation function for each motion"
Ct_x=[Ct_f,Ct_m,Ct_h,Ct_r]          #Correlation functions for given motion
for ct in Ct_x:ax[1].semilogx(t[:n2],ct[:n2])

"Plot components of correlation function"
Ct0p=[Ct0pf,Ct0pm,Ct0ph,Ct0pr]      #Correlation functions for the equilibration of residual tensors (right plots)
Ctp0=[Ctp0f,Ctp0m,Ctp0h,Ctp0r]      #Correlation functions to construct total correlation function of given motion

for k,(ctx,ct0p,ctp0) in enumerate(zip(Ct_x,Ct0p,Ctp0)):
    "Ctp0"
    for m,ct in enumerate(ctp0[2:]):
        ax[2*k+2].semilogx(t[:n2],ct[:n2].real)
        if m!=0:
            ax[2*k+2].semilogx(t[:n2],ct[:n2].imag,linestyle=':')
    ax[2*k+2].semilogx(t[:n2],ctx[:n2],color='black')        
    
    "Ct0p"
    if k<3:
        for m,ct in enumerate(ct0p[2:]):
            ax[2*k+3].semilogx(t[:n2],ct[:n2].real)
            if m!=0:
                ax[2*k+3].semilogx(t[:n2],ct[:n2].imag,linestyle=':')
                    
"Add legends, titles, texts, axes"
for a,ttl,l,txt in zip(ax,titles,legends,text):
    a.set_title(ttl)
    
    if len(l):a.legend(l,loc='upper center')
    a.text(50,0.75,txt)
    
    a.set_ylim([-.05,1.05])
    a.set_yticks(np.linspace(0,1,5))
    a.set_ylabel('C(t)')
    a.set_yticklabels([0,None,.5,None,1])

    a.set_xticks(np.logspace(-3,3,7))
    a.set_xlim([1e-3,t[n2]])
    if a.get_subplotspec().is_last_row():
        a.set_xlabel('t')
        a.set_xticklabels(['0 s','','100 ps','','10 ns','',r'1 $\mu$s'])
    else:
        a.set_xticklabels([])

plt.show()


#%% Plotting the residual tensors in chimeraX for comparison
from pyDIFRATE.chimera.chimeraX_funs import draw_tensors,get_path,set_chimera_path,chimera_path
import os
#set_chimera_path('your path to chimeraX executable here')  #uncomment and fill in path
if os.path.exists(os.path.join(get_path(),'ChimeraX_program_path.txt.txt')) and os.path.exists(chimera_path()):
    text0='Residual tensor for {} motion at {0} ps'
    t0=[50,200,100000]  #Times in ps to show the tensor (we'll show t=0 simultaneously for comparison)
    titles=['Fast wobbling','Three-site hopping','Two-site hopping','Tumbling']
    for ct0p,title in zip(Ct0p,titles):
#        DR.chimeraX
        for t1 in t0:
            b=np.argmin(np.abs(t1-t*1000))
#            draw_tensors(np.concatenate(([[0,0,1,0,0]],[ct0p[:,b]])),chimera_cmds=\
#                ["2dlabel text '{0} at 0 and {1} ps' size 26 x .03 y .92".format(title,t1),'turn x 90'],
#                save_opts='transparentBackground True',
#                fileout='{0}_{1}ps.png'.format(title.replace(' ','_').replace('-','_'),t1))  #Uncomment to save result to file
            draw_tensors(np.concatenate(([[0,0,1,0,0]],[ct0p[:,b]])),chimera_cmds=\
                ["2dlabel text '{0} at 0 and {1} ps' size 26 x .03 y .92".format(title,t1),'turn x 90'])
else:
    print('To visualize residual tensors, install chimeraX and provide the path with DR.chimeraX.set_chimera_path()')
    print('https://www.cgl.ucsf.edu/chimerax/download.html')

