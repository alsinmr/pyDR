import matplotlib.pyplot as plt

from Full_Analysis import KaiMarkov
import pyDIFRATE as DR
from pyDIFRATE.data.load_nmr import load_NMR
from sys import platform
import numpy as np
from matplotlib.pyplot import get_cmap
from os.path import exists

cmap = get_cmap("tab10")

het = load_NMR("nmr/HET-s/13CD2H_double_assign.txt")

het.del_data_pt(range(12,17))
#het.del_data_pt([0,1,2,3,4,11])
n_dets =4
print(het.label)
if "linux" in platform:
    het.detect.r_auto3(int(n_dets), inclS2=True, Normalization='MP')

het = het.fit()
fig = plt.figure()
ax0 = fig.add_subplot(5,1,1)
ax0.plot(het.sens.rhoz().T)
ax = [fig.add_subplot(5,1,2+i) for i in range(4)]
for i in range(4):
    for res in range(het.R.shape[0]):
        ax[i].bar(x=res,height=het.R[res,i], color=cmap(i))

ax[-1].set_xticks(range(het.R.shape[0]))
ax[-1].set_xticklabels(het.label)
targ = het.sens.rhoz()[:4]
targ[0,84:]=0
indices =[]
labels = []
colors = ["red", "orange", "green"]
M = KaiMarkov(simulation=0)
counts = []
for label in het.label:
    counts.append(len(label.split(",")))
    for lab in label.split(","):
        for key in M.sim_dict["residues"].keys():
            print(key[:3], int(key[3:])+147)
            if int(lab[:3]) == int(key[3:])+147:
                print(M.sim_dict['residues'][key]['ct_vecs'])
                for ctlab in M.sim_dict["residues"][key]["ct_vecs"]:
                    if "met-to-plane" == ctlab['name']:
                        print(key, ctlab)
                        indices.append(ctlab['id'])
                        labels.append(key)
print(het.label,len(het.label))
print(labels,len(labels))
print(counts,len(counts),sum(counts))
print(indices,len(indices))


indices = np.array(indices)
sims = [3,0,4,2,1]
legend = []
closes_val = np.zeros((len(het.label),3))
col = np.zeros((len(het.label),3)).astype(int)

fig2 = plt.figure()
bx = fig2.add_subplot(111)
chi_sq = np.zeros(len(sims))
for j,sim in enumerate(sims):
    M = KaiMarkov(simulation=sim)
    legend.append(M.sel_sim_name)
    if not exists("sim{}_R.npy".format(sim)):
        M.calc_new()
        cts = M.cts[indices]
        D = DR.data()
        D.load(Ct={'Ct': cts
            , 't': np.linspace(0, int(M.length / 1000) * M.universe.trajectory.dt, M.length)})
        n_dets = 8
        D.detect.r_target(target=targ,n=12)#r_auto3(n=n_dets)
        md = D.fit()
        md.label = labels
        #np.save("sim{}_R.npy".format(sim),md.R)
        R = md.R
    else:
        R = np.load("sim{}_R.npy".format(sim))
    #legend[j]+=str(cts.shape[1])
    R_avg = np.zeros((len(het.label),3))
    s = 0
    for m,n in enumerate(counts):
        if n == 1:
            R_avg[m,:] = R[s,:3]
        elif n==2:
            R_avg[m,:] = (R[s,:3]+R[s+1,:3])/2
        elif n==3:
            R_avg[m,:] = (R[s,:3]+R[s+1,:3]+R[s+2,:3])/3
        s+=n
    for i in range(3):
        if j > 1:
            ax[i].plot(R_avg[:,i],color=cmap(j),marker="x",linewidth=2)
        for k in range(len(het.label)):
#            if i ==2:
            chi_sq[j]+=(het.R[k,i]-R_avg[k,i])**2/(het.R_std[k,i]**2)
            if np.abs(het.R[k,i]-R_avg[k,i]) < np.abs(het.R[k,i] - closes_val[k,i]):
                closes_val[k,i] = R_avg[k,i]
                col[k,i] = j
print(het.label)
print(labels)

for i in range(3):
    for k in range(len(het.label)):
        ax[i].bar(k+0.3,height=closes_val[k,i], color = cmap(col[k,i]), width=0.2,edgecolor="black")
fig.legend(legend)
for i in range(len(sims)):
    bx.bar(i,chi_sq[i])
fig2.legend(legend)

plt.show()