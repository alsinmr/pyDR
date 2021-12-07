from os import listdir, system, mkdir
from os.path import join, exists, abspath, dirname
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import MDAnalysis as MDA
from calcs import *
import sys
sys.path.append('/Users/albertsmith/Documents/GitHub')
import pyDIFRATE as DR

# setting default parameters for matplotlib plots
matplotlib.rcParams.update({"lines.linewidth": 1,
                            "axes.labelsize": 8,
                            "xtick.labelsize": 8,
                            "ytick.labelsize": 8,
                            "axes.titlesize": 10,
                            'font.size': 8})
TEXTBOX = {'facecolor': 'lightblue',
           'alpha': 0.5}


class KaiMarkov():
    def __init__(self, **kwargs):
        self.n = 200
        self.length = None
        self.labels = []
        self.full_chains = []
        self.plot_hydrogens_rama_3D = []  # get a better name for that, for ramachandran 3d plot
        self.hydrogens_dihedral = []
        self.chi1_groups_dihedral = []
        self.chi2_groups_dihedral = []
        self.ct_vector_groups = []  # Atomgroups to get coordinates for calculation of correlation function
        self.n_dets = None  # number of detectors for analysis
        self.part = None  # the number of frames that will be in one fraction defined by length/n
        self.residues = []  # residues to examine, can be preset in kwargs
        self.pdb = None  # pdb file (path) that will fit to the selected simulation
        self.sel_sim = None  # filename of selected simulation
        self.sel_sim_name = None  # same just with removed xtc
        self.universe = None  # will be MDA.Universe object
        self.default_float = "float32"  # used for all kinds of calculations and saves some space compared to float 64
        self.save_float = "float16"  # saves space for saving issues, since md accuracy is only 0.01 it shouldn't matter
        self.v = False  # verbose, if activated, print
        self.full_dict = {}
        self.dir = dirname(__file__)
        if not exists(join(self.dir, "cts")):
            mkdir(join(self.dir, "cts"))
        for key in kwargs:
            setattr(self, key, kwargs[key])

        self.sim_dict = {'cts':None,
                         "S2s":None,
                         'dihedrals':None,
                         'residues':{}}
        self.dihedral_atomgroups = []
        self.vector_atomgroups = []

        self.select_simulation(kwargs.get("simulation"))
        self.get_methyl_groups()
        self.create_data_container()

        return

        # the length of self.dih is 3 times the number of methylgroups, as well as the length of areas and hops
        self.dih = np.zeros((len(self.hydrogens_dihedral), self.length), dtype=self.default_float)
        self.dih_chi1 = np.zeros((len(self.chi1_groups_dihedral), self.length), dtype=self.default_float)
        self.dih_chi2 = np.zeros((len(self.chi2_groups_dihedral), self.length), dtype=self.default_float)

        self.areas = np.zeros((len(self.hydrogens_dihedral), self.length), dtype="uint8")
        self.chi1_areas = np.zeros((len(self.chi1_groups_dihedral), self.length), dtype="uint8")
        self.chi2_areas = np.zeros((len(self.chi2_groups_dihedral), self.length), dtype="uint8")
        self.coords_3Dplot = np.zeros((len(self.hydrogens_dihedral), 2000, 3), dtype=self.default_float)
        # creating arrays for all occuring hops over the trajectory and for the hop probabilities ocuring in single
        # fragments (n)
        self.hops = np.zeros((len(self.hydrogens_dihedral), self.length), dtype=bool)
        self.hopability = np.zeros((len(self.hydrogens_dihedral), self.n), dtype=self.default_float)
        self.hops_chi1 = np.zeros((len(self.chi1_groups_dihedral), self.length), dtype=bool)
        self.hopability_chi1 = np.zeros((len(self.chi1_groups_dihedral), self.n), dtype=self.default_float)
        self.hops_chi2 = np.zeros((len(self.chi2_groups_dihedral), self.length), dtype=bool)
        self.hopability_chi2 = np.zeros((len(self.chi2_groups_dihedral), self.n), dtype=self.default_float)

        self.ct_vectors = np.zeros((len(self.ct_vector_groups), self.length, 3), dtype=self.default_float)
        self.cts = np.ones((len(self.ct_vectors), self.length), dtype=self.default_float)
        self.S2s = np.zeros((len(self.ct_vectors)), self.default_float)

    def create_data_container(self):
        self.sim_dict["dihedrals"] = self.dihedrals = \
            np.zeros((len(self.dihedral_atomgroups), self.length), dtype= self.default_float)
        self.areas = np.zeros((len(self.dihedral_atomgroups), self.length), dtype=self.default_float)
        self.hops = np.zeros((len(self.dihedral_atomgroups), self.length), dtype=bool)
        self.hopability = np.zeros((len(self.dihedral_atomgroups), self.length), dtype= self.default_float)

        self.ct_vectors = np.zeros((len(self.vector_atomgroups), self.length, 3), dtype=self.default_float)
        self.sim_dict["cts"] = self.cts = np.ones((len(self.vector_atomgroups), self.length), dtype=self.default_float)
        self.sim_dict["S2s"] = self.S2s = np.zeros(len(self.vector_atomgroups), dtype=self.default_float)

    def is_res_in_dict(self,resid):
        '''check if the residue is already in the calculation dicitonary, if not, create a key and the important lists'''
        resname = self.universe.residues[resid-1].resname + str(resid)
        if not resname in self.sim_dict['residues'].keys():
            self.sim_dict['residues'][resname] = {"dihedrals":[], "ct_vecs":[]}
        return resname

    def add_dihedral_by_ids(self, atomlist, name=""):
        #todo put this and the next function in one and add argument for choosing dihedral or vector
        assert len(atomlist)==4, "atomlist has to contain 4 atom ids, contains {}".format(len(atomlist))
        if isinstance(atomlist, list) or isinstance(atomlist,np.ndarray):
            #todo something is goofing around here, but i dont understand what
            atomlist = MDA.AtomGroup(self.universe.atoms[atomlist-1])
        resname = self.is_res_in_dict(atomlist[0].residue.resid)
        dih = {"atom_ids":[], "name":"", "id":None}
        dih["atom_ids"] = atomlist.ids
        dih["name"] = name
        self.sim_dict['residues'][resname]["dihedrals"].append(dih)
        self.dihedral_atomgroups.append(atomlist)
        dih["id"] = len(self.dihedral_atomgroups)-1


    def add_vector_for_ct_by_ids(self, atomlist, name=""):
        assert len(atomlist)==4 or len(atomlist)==5, "atomlist ahs to contain 4 or 5 atom ids, contains {}".format(len(atomlist))
        if isinstance(atomlist, list) or isinstance(atomlist,np.ndarray):
            #todo something is goofing around here, but i dont understand what
            atomlist = MDA.AtomGroup(self.universe.atoms[atomlist-1])
        resname = self.is_res_in_dict(atomlist[0].residue.resid)
        vec = {"atom_ids":[], "name":"", "id":None}
        vec["atom_ids"] = atomlist.ids
        vec["name"] = name
        self.sim_dict['residues'][resname]["ct_vecs"].append(vec)
        self.vector_atomgroups.append(atomlist)
        vec["id"] = len(self.vector_atomgroups)-1

    def get_methyl_groups(self):
        make_AG = lambda alist: MDA.AtomGroup([this[_] for _ in alist])
        assert self.sel_sim and self.pdb, "You have to select a PDB file AND a simulation before "
        if self.universe is None:
            self.universe = MDA.Universe(self.pdb, self.sel_sim)
        uni = self.universe  # just to shorten this
        # TODO this will just work for HETs right now, make it more general
        seg = 1 if len(uni.segments) == 3 else 2 if len(uni.segments) == 5 else 0
        # TODO add segment selection to select simulation function
        segment = uni.segments[seg]
        residues = [segment.residues[r] for r in self.residues] if len(self.residues) else segment.residues
        dix = self.full_dict
        for i, res in enumerate(residues):
            methyls = search_methyl_groups(res, v=self.v)
            if len(methyls):
                group_label = "{}_{}".format(res.resname, res.resnum + 147)
                this = dix[group_label] = {}
                this["ct_indices"] = []
                this["ct_labels"] = []
                print(i, res.resname, res.resnum + 147, "contains", len(methyls), "methyl groups")
                chain_for_3d_plot = dix[group_label]["chain"] = ["C", "CA", "N", "CA", "CB"]
                this["N"] = res.atoms[0]
                this["H1"] = methyls[0][0]  # this is only useful if you have only one H
                this["H12"] = methyls[0][1]
                this["H13"] = methyls[0][2]
                this["C"] = methyls[0][-1]
                this["CA"] = methyls[0][-2]
                this["CB"] = methyls[0][-3]
                self.add_vector_for_ct_by_ids(make_AG(["CB", "N", "CA", "C"]),r'$\chi_{1,lib.}$' if "ALA" not in res.resname else r'$CH_{3,lib.}$')
                # todo add methionin
                if "ILE" in res.resname:
                    for _ in ["CG2", "CB", "CG1", "CD"]:
                        chain_for_3d_plot.append(_)
                    this["CG2"] = methyls[0][-4]
                    this["CG1"] = methyls[1][4]
                    this["CD"] = methyls[1][3]
                    this["H2"] = methyls[1][0]
                    #TODO maybe uncomment this sometime, will give you total movement of the CH vector compared to the
                    #TODO peptide plane
                    self.add_dihedral_by_ids(make_AG(["C","CA","CB","CG1"]), "chi1")
                    self.add_dihedral_by_ids(make_AG(["CA", "CB", "CG1", "CD"]), "chi2")
                    self.add_vector_for_ct_by_ids(make_AG(["CG1", "C", "CB", "CA"]), r'$\chi_{1,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["CG1", "CA", "CB", "CG2"]), r'$\chi_{2,lib.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["CD", "CB", "CG1", "CG2"]), r'$\chi_{2,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "CB", "CG2", "CG1"]), r'$CH_{3,rot.}^1$')
                    self.add_vector_for_ct_by_ids(make_AG(["H2", "CG1", "CD", "CB"]), r'$CH_{3,rot.}^2$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1","N","CA","C","CG2"]),"met-to-plane")
                    #self.add_vector_for_ct_by_ids(make_AG(["H2", "N", "CA", "C", "CD"]), "met-to-plane_2")
                elif "VAL" in res.resname:
                    for _ in ["CG1", "CB", "CG2"]:
                        chain_for_3d_plot.append(_)
                    this["CG1"] = methyls[0][-4]  # in "VAL" this is CG1
                    this["CG2"] = methyls[1][3]  # in "VAL this is CG2
                    this["H2"] = methyls[1][0]
                    self.add_vector_for_ct_by_ids(make_AG(["CG1","C","CB","CA"]),r'$\chi_{1,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["CG2", "CG1", "CB", "CA"]), r'$CH_{3,lib.}^2$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "CB", "CG1", "CG2"]), r'$CH_{3,rot.}^1$')
                    self.add_vector_for_ct_by_ids(make_AG(["H2", "CB", "CG2", "CG1"]), r'$CH_{3,rot.}^2$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1","N","CA","C","CG1"]), "met-to-plane")
                    self.add_vector_for_ct_by_ids(make_AG(["H2","N","CA","C","CG2"]), "met-to-plane_2")
                    self.add_dihedral_by_ids(make_AG(["C","CA","CB","CG2"]),"chi1")
                elif "LEU" in res.resname:
                    for _ in ["CG", "CD1", "CG", "CD2"]:
                        chain_for_3d_plot.append(_)
                    this["CG"] = methyls[0][4]
                    this["CD1"] = methyls[0][3]
                    this["CD2"] = methyls[1][3]
                    this["H2"] = methyls[1][0]
                    self.add_vector_for_ct_by_ids(make_AG(["CG", "CA", "CB", "C"]),r'$\chi_{1,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["CD1", "CA", "CG", "CB"]),r'$\chi_{2,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "CG", "CD1", "CD2"]),r'$CH_{3,rot.}^1$')
                    self.add_vector_for_ct_by_ids(make_AG(["H2", "CG", "CD2", "CD1"]),r'$CH_{3,rot.}^2$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1","N","CA","C","CD1"]),"met-to-plane")
                    self.add_vector_for_ct_by_ids(make_AG(["H2","N","CA","C","CD2"]),"met-to-plane_2")
                    self.add_dihedral_by_ids(make_AG(["C","CA","CB","CG"]),"chi1")
                    self.add_dihedral_by_ids(make_AG(["CA","CB","CG","CD2"]),"chi2")
                elif "THR" in res.resname:
                    this["CG2"] = methyls[0][3]
                    chain_for_3d_plot.append("CG2")
                    self.add_vector_for_ct_by_ids(make_AG(["CG2", "CA", "CB", "C"]),r'$\chi_{1,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "CA", "CG2", "CB"]),r'$CH_{3,rot.}$')
                    #self.add_vector_for_ct_by_ids(make_AG(["H1","N","CA","C","CG2"]),"met-to-plane")
                    self.add_dihedral_by_ids(make_AG(["C","CA","CB","CG2"]))
                elif "ALA" in res.resname:
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "CA", "CB", "C"]),r'$CH_{3,rot.}$')
                    self.add_vector_for_ct_by_ids(make_AG(["H1","N","CA","C","CB"]),"met-to-plane")
                    self.add_vector_for_ct_by_ids(make_AG(["H1", "H12", "CB", "CA"]),r'$C-H_{lib.}$')
                this["labels"] = []
                for j, m in enumerate(methyls):
                    met_label = group_label + ("_B" if j else ("_A" if len(methyls) == 2 else ""))
                    this["labels"].append(met_label)
                    this[met_label] = {}
                    sdix = this[met_label]  # subdictionary
                    sdix["index"] = len(self.hydrogens_dihedral)
                    sdix["carbon"] = m[3]
                    group = MDA.AtomGroup(m)
                    self.full_chains.append(group)
                    self.plot_hydrogens_rama_3D.append(group[[-1, -2, -3, 0]])
                    self.plot_hydrogens_rama_3D.append(group[[-1, -2, -3, 1]])
                    self.plot_hydrogens_rama_3D.append(group[[-1, -2, -3, 2]])
                    # One group appended for every hydrogen
                    # it is a little weird but still this was the fastest way to calculate the dih for this
                    self.hydrogens_dihedral.append(
                        group[[0, 3, 4, 5]])  # presetting this saves much time for calculations
                    self.hydrogens_dihedral.append(group[[1, 3, 4, 5]])
                    self.hydrogens_dihedral.append(group[[2, 3, 4, 5]])
                    self.add_dihedral_by_ids(group[[0,3,4,5]],"CH3_{}".format(j+1))

        assert len(self.hydrogens_dihedral), "No methylgroups detected"
        print("Total:", int(len(self.hydrogens_dihedral) / 3), "methylgroups detected")
        if self.v:
            for key in dix.keys():
                print(key)
                print(dix[key])

    def select_simulation(self, number=None):
        xtcdir = join(self.dir, "xtcs")
        pdbdir = join(self.dir, "pdbs")
        if not exists(xtcdir) or not exists(pdbdir):
            mkdir(xtcdir)
            mkdir(pdbdir)
            assert 0, "folders for xtcs and pdbs were created, plz fill with trajectories and pdbfiles"

        if not self.sel_sim:
            simulations = [f for f in listdir(xtcdir) if f.endswith(".xtc")]
            assert len(simulations), "please copy (or link) some simulations in your xtcs/ folder"
            assert len(listdir(pdbdir)), "of course you also need a pdb file (in pdbs/)"
            if number is None:
                print("Available Simulations:")
                for i, sim in enumerate(simulations):
                    print(i, sim)
                i = int(input("Choosen: "))
            else:
                i = number
            try:
                self.sel_sim = join(xtcdir, simulations[i])
                self.sel_sim_name = simulations[i].split(".")[0]
            except:
                print("not a valid simulation")
                self.select_simulation()
        else:
            self.sel_sim_name = self.sel_sim.split(".")[0].split("/")[-1]
        for pdb in listdir(pdbdir):
            f = join(pdbdir, pdb)
            print(f)
            try:
                self.universe = MDA.Universe(f, self.sel_sim)
                self.pdb = f
                break
            except:
                self.universe = None

        assert self.universe, "Couldnt find a fitting pdb for this simulation, please copy one in the pdb folder"
        print("Simulation", self.sel_sim_name, "with timestep", self.universe.trajectory.dt, "and", self.pdb,
              "selected")
        self.dt = self.universe.trajectory.dt * 10 ** (-12)  # 5 ps
        if self.length == None:
            self.length = len(self.universe.trajectory)
        # todo put segment selection here if more than one segment is available

    def load_new(self):
        dihedral_calc_indices = np.ones(len(self.dihedral_atomgroups))
        ct_calc_indices = np.ones(len(self.vector_atomgroups))
        fn = join("calced_data",self.sel_sim_name+".npy")
        if exists(fn):
            loaded = np.load(fn, allow_pickle=True).item()
            for key in self.sim_dict['residues'].keys():
                if not key in loaded:
                    continue
                for dih in self.sim_dict['residues'][key]['dihedrals']:
                    for dih2 in loaded[key]['dihedrals']:
                        if (dih['atom_ids'] == dih2['atom_ids']).sum() == len(dih['atom_ids']) \
                                and dih2['values'].shape[0] >= self.length:
                            self.sim_dict['dihedrals'][dih['id']] = dih2['values'][:self.length,]
                            dihedral_calc_indices[dih['id']]=0
                for ct in self.sim_dict['residues'][key]['ct_vecs']:
                    for ct2 in loaded[key]["ct_vecs"]:
                        if len(ct['atom_ids']) == len(ct2['atom_ids']):
                            if (ct['atom_ids'] == ct2['atom_ids']).sum() == len(ct['atom_ids']) \
                                    and ct2['values'].shape[0] >= self.length:
                                self.sim_dict['cts'][ct['id']] = ct2['values'][:self.length,]
                                self.sim_dict['S2s'][ct['id']] = ct2['S2']
                                ct_calc_indices[ct['id']] = 0
        return dihedral_calc_indices, ct_calc_indices

    def save_new(self):
        #assert 0, "change before save again!"
        #todo append to existing structure instead of overwriting
        #rearrange the dictionary, just save the 'residues' part and remove the index, put instead the values
        #directly to the thing

        #todo add sparse values to dict
        for res in self.sim_dict['residues'].keys():
            for dih in self.sim_dict['residues'][res]['dihedrals']:
                dih["values"] = self.sim_dict['dihedrals'][dih['id']].astype("float16")
            for ct in self.sim_dict['residues'][res]['ct_vecs']:
                ct["values"] = self.sim_dict['cts'][ct['id']].astype("float16")
                ct["S2"] = self.sim_dict['S2s'][ct['id']].astype("float16")

        self.sim_dict["cts"] = self.sim_dict["cts"].astype("float16")
        self.sim_dict["dihedrals"] = self.sim_dict["dihedrals"].astype("float16")
        self.sim_dict["S2s"] = self.sim_dict["S2s"].astype("float16")
        np.save(join("calced_data",self.sel_sim_name+".npy"), self.sim_dict['residues'])

    def calc_new(self, **kwargs):
        traj = self.universe.trajectory
        assert self.length <= len(traj), "Length {} exceeding length of trajectory ({}), set length to {}" \
                                         "".format(self.length, len(traj), len(traj))
        dihedral_calc, ct_calc = self.load_new()
        if kwargs.get("force_calc"):
            dihedral_calc[:] = 1
            ct_calc[:] = 1
        print(dihedral_calc.sum(), ct_calc.sum())
        if dihedral_calc.sum() or ct_calc.sum():
            for i in range(self.length):
                if i%10000==0: print(i)
                traj[i]
                for j, group in enumerate(self.dihedral_atomgroups):
                    if dihedral_calc[j]:
                        self.dihedrals[j,i] = fastest_dihedral(group.positions)
                for j, group in enumerate(self.vector_atomgroups):
                    if ct_calc[j]:
                        pos_xyz_o(self.ct_vectors[j, i], *group.positions)

            np.save("ct_vecs_3pw_400000.npy",self.ct_vectors)
            if ct_calc.sum():
                get_ct_S2(self.cts, self.S2s, self.ct_vectors, ct_calc,
                          kwargs.get("sparse") if kwargs.get("sparse") is not None else 1)
            print("please dont save!")
            #self.save_new()
        print(self.sim_dict["S2s"], self.sim_dict["S2s"].shape)


    def calc(self, **kwargs):
        '''doing all importand calculations for the anylsis
        available kwargs:
        "ct_off" disabeling the calculation of ct, for time saving purposes
        "recalc_ct" forcing to recalculate cts even if they are already stored on drive
        "sparse" for ct calculation, low sparse value decreases computation time and increases noise'''
        traj = self.universe.trajectory
        assert self.length <= len(traj), "Length {} exceeding length of trajectory ({}), set length to {}" \
                                         "".format(self.length, len(traj), len(traj))
        if not self.load_state():
            # iterating over all timepoints defined by length and calculating dihedrals for all methyl hydrogens,
            # chi_1 and chi_2 bonds. Furthermore calculating the vertices for later C(t) calculation
            for i in range(self.length):
                if i % 10000 == 0: print(i)
                traj[i]
                for j, group in enumerate(self.hydrogens_dihedral):  # first hydrogen
                    self.dih[j, i] = fastest_dihedral(group.positions)
                for j, group in enumerate(self.chi1_groups_dihedral):
                    self.dih_chi1[j, i] = fastest_dihedral(group.positions)
                for j, group in enumerate(self.chi2_groups_dihedral):
                    self.dih_chi2[j, i] = fastest_dihedral(group.positions)
                for j, group in enumerate(self.ct_vector_groups):
                    pos_xyz_o(self.ct_vectors[j, i],*group.positions)

            get_ct_S2(self.cts, self.S2s, self.ct_vectors,
                      kwargs.get("sparse") if kwargs.get("sparse") is not None else 1)
            self.save_state()

        for i, _ in enumerate(range(0, self.length, int(self.length/2000))):
            # todo calculate a good number of points for the plot
            if _ % 10000 == 0: print(_)
            traj[_]
            if i == 2000:
                break
            for j, group in enumerate(self.plot_hydrogens_rama_3D):  # first hydrogen
                self.coords_3Dplot[j, i] = get_x_y_z(group.positions)
        # calculating in which of three areas a methyl hydrogen appears at every timepoint
        # furthermore check if a hop between timepoints occured by comparing the areas
        self.areas[:, :] += ((self.dih[:, :] >= 0) == (self.dih[:, :] < 120)).astype('uint8')
        self.areas[:, :] += ((self.dih[:, :] < 0) == (self.dih[:, :] >= -120)).astype('uint8') * 2
        self.hops[:, 1:] = self.areas[:, :-1] != self.areas[:, 1:]  # the first element of hops is by definition 0
        # same for chi1
        self.chi1_areas[:, :] += ((self.dih_chi1[:, :] >= 0) == (self.dih_chi1[:, :] < 120)).astype('uint8')
        self.chi1_areas[:, :] += ((self.dih_chi1[:, :] < 0) == (self.dih_chi1[:, :] >= -120)).astype('uint8') * 2
        self.hops_chi1[:, 1:] = self.chi1_areas[:, :-1] != self.chi1_areas[:, 1:]
        # and chi2
        self.chi2_areas[:, :] += ((self.dih_chi2[:, :] >= 0) == (self.dih_chi2[:, :] < 120)).astype('uint8')
        self.chi2_areas[:, :] += ((self.dih_chi2[:, :] < 0) == (self.dih_chi2[:, :] >= -120)).astype('uint8') * 2
        self.hops_chi2[:, 1:] = self.chi2_areas[:, :-1] != self.chi2_areas[:, 1:]
        # how often does it happen that 2 hydrogens appear in the same area?
        print(np.sum(self.areas[0::3, :] == self.areas[1::3, :]))  # TODO i dont know what todo with it
        print(np.sum(self.areas[0::3, :] == self.areas[2::3, :]))  # TODO
        print(np.sum(self.areas[1::3, :] == self.areas[2::3, :]))  # TODO
        self.part = part = int(self.length / self.n)
        # calculating the hop probabilty for every methylgroup and chi1 and chi2 bond for every fraction n
        for i in range(self.n):
            for j in range(0, len(self.hydrogens_dihedral), 3):
                self.hopability[j, i] = np.sum(self.hops[j, i * part:(i + 1) * part]) / part
            for j in range(len(self.chi1_groups_dihedral)):
                self.hopability_chi1[j, i] = np.sum(self.hops_chi1[j, i * part:(i + 1) * part]) / part
            for j in range(len(self.chi2_groups_dihedral)):
                self.hopability_chi2[j, i] = np.sum(self.hops_chi2[j, i * part:(i + 1) * part]) / part

    def plot_single_res(self, label):
        def tc_str(z):
            unit = 's'
            if z <= -12: z, unit = z + 15, 'fs'
            if z <= -9: z, unit = z + 12, 'ps'
            if z <= -6: z, unit = z + 9, 'ns'
            if z <= -3: z, unit = z + 6, r'$\mu$s'
            if z <= 0: z, unit = z + 3, 'ms'
            tc = np.round(10 ** z, -1 if z >= 2 else (0 if z >= 1 else 1))
            return '{:2} '.format(tc if tc < 10 else int(tc)) + unit
        fig = plt.figure()
        ax = fig.add_subplot(211)
        ax.plot(np.array([self.cts[vec['id']] for vec in self.sim_dict['residues'][label]['ct_vecs']]).T)
        ax.legend([ct['name'] for ct in self.sim_dict['residues'][label]['ct_vecs']])
        bx = fig.add_subplot(212)
        bx.plot(np.array([np.sort(self.dihedrals[dih['id']]) for dih in self.sim_dict['residues'][label]['dihedrals']]).T)
        bx.legend([dih['name'] for dih in self.sim_dict['residues'][label]['dihedrals']])

        fig.suptitle(label)
        cts = self.cts[np.array([vec['id'] for vec in self.sim_dict['residues'][label]['ct_vecs']])]
        D = DR.data()
        D.load(Ct={'Ct': cts
            , 't': np.linspace(0, int(self.length / 1000) * self.universe.trajectory.dt, self.length)})
        n_dets = self.n_dets if self.n_dets else 8
        remove_S2 = 1
        D.detect.r_auto3(n=n_dets)
        fit = D.fit()
        tick_labels = ['~' + tc_str(z0) for z0 in fit.sens.info.loc['z0']]
        tick_labels[0] = '<' + tick_labels[1][1:]
        # tick_labels[-1] = '>' + tick_labels[-2][1:]
        tick_labels = tick_labels[:n_dets - 1]  # removed last label because no data were there
        page_2 = plt.figure()
        h = int((cts.shape[0] + 1) / 2) + 1
        page_2.set_size_inches(8.3, 11.7 / 8 * h)  # reduce page size for less cts
        ax0 = page_2.add_subplot(h, 1, 1)
        ax0.plot(D.sens.z(), fit.sens.rhoz().T)
        ax0.set_xlim(-14, -3)
        ax0.set_title("Detector Sensitivities")
        ax0.set_ylabel(r'$(1-S^2)$')
        ax0.set_xlabel(r'$\log_{10}(\tau_c/s)$')
        ax0.set_yticks([0, 1])
        ax = [page_2.add_subplot(h, 2, 3 + i) for i in range(cts.shape[0])]
        for d in range(n_dets - remove_S2):
            for r in range(cts.shape[0]):
                ax[r].bar(d, fit.R[r, d])
        for r in range(cts.shape[0]):
            maxR = np.max(fit.R[r, :-remove_S2])
            ylim = np.round(maxR + .06, 1) if maxR < .5 else .75 if maxR < .72 else 1
            print(ylim)
            ax[r].set_ylim(0, ylim)
            ax[r].text(n_dets / 2, ylim * .75, self.sim_dict['residues'][label]['ct_vecs'][r]['name'], bbox=TEXTBOX)
            ax[r].set_xticks(range(n_dets - remove_S2))
            if r < cts.shape[0] - 2:
                ax[r].set_xticklabels([])
            else:
                ax[r].set_xticklabels(tick_labels, rotation=66)
            if r % 2 == 0:
                ax[r].set_ylabel(r'$\rho_n^{(\Theta,S)}$')

    def plot_all(self):
        """this function creates the plots for all examined residues depending on the number of methyl groups and
        if a chi1 and/or chi2 bond is contained. All plots will be saved in a predefined folder (now test/). If only
        a single residue is examined the plot will be immediately shown (but still saved)"""
        def plot_hopability_with_chi_states(axis, index, meth_type, chi_num, chi_type, hoptype="methyl"):
            """there wasnt a very smooth way to put this calculation in the self.calc function, so i put it here
            if a chi2 bond is existing, it will separate the hop probability in three bars to show in which state of
            chi2 the hops are occuring. furthermore, if a shift_axis is given, it will compare if the hop probability
            is changing by the state itself"""
            #todo rename the meth type because now it can be chi or methyl
            #todo this function can be rebuilt anyway
            hop_state_0 = np.zeros(self.n, dtype=self.default_float)
            hop_state_1 = np.zeros(self.n, dtype=self.default_float)
            hop_state_2 = np.zeros(self.n, dtype=self.default_float)
            chi_areas = self.chi1_areas if chi_type == 1 else self.chi2_areas
            hops = self.hops if "methyl" in hoptype else self.hops_chi2 if "chi" in hoptype else None
            for i in range(self.n):
                hop_state_0[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 0)).sum() / part
                hop_state_1[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 1)).sum() / part
                hop_state_2[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 2)).sum() / part
            axis.bar(np.arange(n) + .5, hop_state_0 + hop_state_1 + hop_state_2, color="b")
            axis.bar(np.arange(n) + .5, hop_state_0 + hop_state_1, color="g")
            axis.bar(np.arange(n) + .5, hop_state_0, color="r")
            axis.set_xlim(0, n)
            if hoptype == "methyl":
                axis.text(self.n / 5, axis.get_ylim()[1] * .8,
                      r'$CH_3^{{{}}} with \chi_{{{}}}-states$'.format(meth_type, chi_type), bbox=TEXTBOX)
            else:
                axis.text(self.n / 5, axis.get_ylim()[1] * .8,
                          r'$\chi_2 with \chi_1-states$', bbox=TEXTBOX)
            axis.set_xticklabels([])

            #TODO remove
            #if meth_type==2:
            #    axis.set_ylim(0,1)
            #if hoptype != "methyl":
            #    axis.set_ylim(0,0.2)
            #todo remove

        def plot_detectors():
            def tc_str(z):
                unit = 's'
                if z <= -12: z, unit = z + 15, 'fs'
                if z <= -9: z, unit = z + 12, 'ps'
                if z <= -6: z, unit = z + 9, 'ns'
                if z <= -3: z, unit = z + 6, r'$\mu$s'
                if z <= 0: z, unit = z + 3, 'ms'
                tc = np.round(10 ** z, -1 if z >= 2 else (0 if z >= 1 else 1))
                return '{:2} '.format(tc if tc < 10 else int(tc)) + unit
            D = DR.data()
            cts = self.cts[dix[key]["ct_indices"]]
            D.load(Ct={'Ct':cts
                       ,'t': np.linspace(0,int(self.length/1000)*self.universe.trajectory.dt, self.length)})
            n_dets = self.n_dets if self.n_dets else 8
            remove_S2 = 1
            D.detect.r_auto3(n=n_dets)
            fit = D.fit()
            tick_labels = ['~' + tc_str(z0) for z0 in fit.sens.info.loc['z0']]
            tick_labels[0] = '<' + tick_labels[1][1:]
            #tick_labels[-1] = '>' + tick_labels[-2][1:]
            tick_labels = tick_labels[:n_dets-1]  # removed last label because no data were there
            page_2 = plt.figure()
            h = int((cts.shape[0]+1)/2)+1
            page_2.set_size_inches(8.3, 11.7/8*h)  # reduce page size for less cts
            ax0 = page_2.add_subplot(h, 1, 1)
            ax0.plot(D.sens.z(), fit.sens.rhoz().T)
            ax0.set_xlim(-14, -3)
            ax0.set_title("Detector Sensitivities")
            ax0.set_ylabel(r'$(1-S^2)$')
            ax0.set_xlabel(r'$\log_{10}(\tau_c/s)$')
            ax0.set_yticks([0, 1])
            ax = [page_2.add_subplot(h, 2, 3 + i) for i in range(cts.shape[0])]
            for d in range(n_dets - remove_S2):
                for r in range(cts.shape[0]):
                    ax[r].bar(d, fit.R[r, d])
            for r in range(cts.shape[0]):
                maxR = np.max(fit.R[r, :-remove_S2])
                ylim = np.round(maxR + .06,1) if maxR < .5 else .75 if maxR < .72 else 1
                print(ylim)
                ax[r].set_ylim(0, ylim)
                ax[r].text(n_dets/2, ylim * .75, dix[key]["ct_labels"][r], bbox=TEXTBOX)
                ax[r].set_xticks(range(n_dets - remove_S2))
                if r < cts.shape[0]-2:
                    ax[r].set_xticklabels([])
                else:
                    ax[r].set_xticklabels(tick_labels, rotation=66)
                if r % 2 == 0:
                    ax[r].set_ylabel(r'$\rho_n^{(\Theta,S)}$')
            page_2.suptitle("{}_{}".format(key, self.sel_sim_name))
            page_2.tight_layout()
            pdf.savefig(page_2)
            plt.close(page_2)
        dix = self.full_dict
        part = self.part
        n = self.n
        if not exists(join(self.dir, "plots")):
            mkdir(join(self.dir, "plots"))

        for _, key in enumerate(dix):
            if not "ILE_231" in key:
                continue
            with PdfPages(join(self.dir, "plots", "{}_{}.pdf".format(key, self.sel_sim_name))) as pdf:
                num = dix[key][dix[key]["labels"][0]]["index"]  # index of the first (or only) methylgroup
                if len(dix[key]["labels"]) == 2:
                    two = True
                    num2 = dix[key][dix[key]["labels"][1]]["index"]  # index of the second methylgroup
                else:
                    two = False
                chi1 = dix[key].get("chi1")  # index for chi1
                chi2 = dix[key].get("chi2")  # index for chi2
                ### Initializing figure and axes
                page_1 = plt.figure()
                page_1.set_size_inches(8.3, 11.7)  # (14, 10)
                if chi1 is not None or chi2 is not None:
                    ax_3d_plot = page_1.add_subplot(321, projection='3d')  # for plotting 3d distribution of methyl hydrogens
                    rama_ax = page_1.add_subplot(322)
                    if two:  # check if two methylgroups are available in the residue
                        hopability_methyl_1_axis = page_1.add_subplot(625)  # if yes, create two hopability plots
                        hopability_methyl_2_axis = page_1.add_subplot(627)
                        if chi2 is not None:
                            hopability_methyl_1_axis_B = page_1.add_subplot(6, 2, 9)  # if yes, create two hopability plots
                            hopability_methyl_2_axis_B = page_1.add_subplot(6, 2, 11)
                    else:
                        hopability_single_methyl_axis = page_1.add_subplot(323)  # if not, only one (surprise)
                    if chi2 is not None:  # same here if chi1 and chi2 are available
                        ct_plot = page_1.add_subplot(326)
                        chi1_hopability_plot = page_1.add_subplot(626)  # then make two plots
                        chi2_hopability_plot = page_1.add_subplot(628)
                        pie_chi2 = page_1.add_subplot(645)
                        pie_chi1 = page_1.add_subplot(646)
                        pie_chi2.pie([(self.chi2_areas[chi2]==0).sum(), (self.chi2_areas[chi2]==1).sum(),
                                      (self.chi2_areas[chi2]==2).sum()], colors=["r", "g", "b"])
                        pie_chi2.text(0,0,r'$\chi_2$', ha="center", va="center", bbox=TEXTBOX)
                    else:
                        ct_plot = page_1.add_subplot(313)
                        chi1_hopability_ax_solo = page_1.add_subplot(324)  # or only one
                        #tau_legend.append(r'$\chi_1 hops$')
                        pie_chi1 = page_1.add_subplot(645)
                    pie_chi1.pie([(self.chi1_areas[chi1]==0).sum(), (self.chi1_areas[chi1]==1).sum(),
                                  (self.chi1_areas[chi1]==2).sum()], colors=["r", "g", "b"])
                    pie_chi1.text(0,0,r'$\chi_1$', ha="center", va="center", bbox=TEXTBOX)
                else:
                    # this part is mostly for Alanine
                    ax_3d_plot = page_1.add_subplot(321, projection='3d')
                    ct_plot = page_1.add_subplot(313)
                    hopability_single_methyl_axis = page_1.add_subplot(312)  # if not, only one (surprise)
                ### start to plot, first the 3d scatter plot of the residue chain
                ax_3d_plot.set_axis_off()
                ax_3d_plot.scatter(self.coords_3Dplot[num, :, 0], self.coords_3Dplot[num, :, 1],
                                   self.coords_3Dplot[num, :, 2], s=.5, c="g")  # areas[num])
                carbons = np.array([get_x_y_z_old(dix[key]["C"].position, dix[key]["CA"].position,
                                                  dix[key]["CB"].position, dix[key][_].position) for _ in
                                    dix[key]["chain"]])
                ax_3d_plot.plot(*carbons.T, marker="o", markersize=10)
                for c, label in zip(carbons, dix[key]["chain"]):
                    ax_3d_plot.text(*c.T, label)
                ax_3d_plot.view_init(elev=0, azim=45)
                if len(dix[key]["ct_indices"]):
                    ct_plot.semilogx(np.arange(0, self.length),
                                     self.cts[dix[key]["ct_indices"]].T)
                    for l,index in enumerate(dix[key]["ct_indices"]):
                        dix[key]["ct_labels"][l] += " S²={:.2f}".format(self.S2s[index])
                    ct_plot.legend(dix[key]["ct_labels"])
                    ct_plot.set_xlabel("timepoints")
                    # todo calculate the correlation time by dt and stuff and set the xticks
                    ct_plot.set_ylabel("C(t)")
                    ct_plot.set_yticks([0, .5, 1])
                    ct_plot.set_ylim(-.1, 1)
                    ct_plot.set_title("Correlation functions")

                if two:  # todo this is still clumsy
                    ax_3d_plot.scatter(self.coords_3Dplot[num2, :, 0], self.coords_3Dplot[num2, :, 1],
                                       self.coords_3Dplot[num2, :, 2], s=.5, c="b")  # 3D Rama plot
                    if chi2 is not None:
                        plot_hopability_with_chi_states(hopability_methyl_1_axis, num, 1, chi2, 2)
                        plot_hopability_with_chi_states(hopability_methyl_2_axis, num2, 2, chi2, 2)
                        plot_hopability_with_chi_states(hopability_methyl_1_axis_B, num, 1, chi1, 1)
                        plot_hopability_with_chi_states(hopability_methyl_2_axis_B, num2, 2, chi1, 1)
                    elif chi1 is not None:
                        plot_hopability_with_chi_states(hopability_methyl_1_axis, num, 1, chi1, 1)
                        plot_hopability_with_chi_states(hopability_methyl_2_axis, num2, 2, chi1, 1)
                    else:
                        hopability_methyl_1_axis.bar(np.arange(n), self.hopability[num, :], color="black")
                        hopability_methyl_2_axis.bar(np.arange(n), self.hopability[num2, :], color="black")
                else:
                    hopability_single_methyl_axis.bar(np.arange(n) + .5, self.hopability[num], color="black")  # hopability
                    hopability_single_methyl_axis.set_xlim(0, n)
                    if chi1 is not None:
                        plot_hopability_with_chi_states(hopability_single_methyl_axis, num, 1, chi1, 1)
                if chi1 is not None or chi2 is not None:
                    if chi2 is not None:
                        rama_ax.hist2d((self.dih_chi1[chi1] + 240) % 360, (self.dih_chi2[chi2] + 240) % 360,
                                       range=[[0, 360], [0, 360]], bins=[180, 180], cmin=0.1)
                        rama_ax.set_xlabel(r'$\chi_1$')
                        rama_ax.set_ylabel(r'$\chi_2$')
                        rama_ax.set_ylim(0, 360)
                        rama_ax.set_yticks([0, 60, 180, 300, 360])
                        chi1_hopability_plot.bar(np.arange(n) + .5, self.hopability_chi1[chi1], color="black")
                        chi1_hopability_plot.set_xlim(0, n)
                        chi1_hopability_plot.set_xticklabels([])
                        chi1_hopability_plot.text(n / 10, chi1_hopability_plot.get_ylim()[1] * .8,
                                                  r'$\chi_1 hop-probability$', bbox=TEXTBOX)
                        plot_hopability_with_chi_states(chi2_hopability_plot,chi2,2,chi1,1,"chi")
                    else:
                        chi1_hopability_ax_solo.bar(np.arange(n) + .5, self.hopability_chi1[chi1], color="black")
                        chi1_hopability_ax_solo.set_xlim(0, n)
                        chi1_hopability_ax_solo.text(n / 10, chi1_hopability_ax_solo.get_ylim()[1] * .8,
                                                     r'$\chi_1 hop-probability$', bbox=TEXTBOX)
                        rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(0, 120, 3),
                                     range=range(120), color="r")
                        rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(120, 240, 3),
                                     range=range(120), color="b")
                        rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(240, 360, 3),
                                     range=range(120), color="g")
                        rama_ax.set_xlabel(r'$\chi_1$')
                        rama_ax.set_ylabel("occurrence")
                        rama_ax.set_yticklabels([])
                    # final configuration of the plots
                    rama_ax.set_xlim(0, 360)
                    rama_ax.set_xticks([0, 60, 180, 300, 360])

                ct_plot.set_xlim(1, self.length)
                page_1.suptitle("{}_{}".format(key, self.sel_sim_name))
                page_1.tight_layout()
                pdf.savefig(page_1)
                plt.close(page_1)
                plot_detectors()
                print("Save", key, "plot")

    def plot_det_gui(self, residue):
        '''temporary funciton to show the funcitonality of the GUI'''
        def plot_hopability_with_chi_states(axis, index, meth_type, chi_num, chi_type, hoptype="methyl"):
            """there wasnt a very smooth way to put this calculation in the self.calc function, so i put it here
            if a chi2 bond is existing, it will separate the hop probability in three bars to show in which state of
            chi2 the hops are occuring. furthermore, if a shift_axis is given, it will compare if the hop probability
            is changing by the state itself"""
            #todo rename the meth type because now it can be chi or methyl
            #todo this function can be rebuilt anyway
            hop_state_0 = np.zeros(self.n, dtype=self.default_float)
            hop_state_1 = np.zeros(self.n, dtype=self.default_float)
            hop_state_2 = np.zeros(self.n, dtype=self.default_float)
            chi_areas = self.chi1_areas if chi_type == 1 else self.chi2_areas
            hops = self.hops if "methyl" in hoptype else self.hops_chi2 if "chi" in hoptype else None
            for i in range(self.n):
                hop_state_0[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 0)).sum() / part
                hop_state_1[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 1)).sum() / part
                hop_state_2[i] = (hops[index, i * part:(i + 1) * part] *
                                  (chi_areas[chi_num, i * part:(i + 1) * part] == 2)).sum() / part
            axis.bar(np.arange(n) + .5, hop_state_0 + hop_state_1 + hop_state_2, color="b")
            axis.bar(np.arange(n) + .5, hop_state_0 + hop_state_1, color="g")
            axis.bar(np.arange(n) + .5, hop_state_0, color="r")
            axis.set_xlim(0, n)
            if hoptype == "methyl":
                axis.text(self.n / 5, axis.get_ylim()[1] * .8,
                      r'$CH_3^{{{}}} with \chi_{{{}}}-states$'.format(meth_type, chi_type), bbox=TEXTBOX)
            else:
                axis.text(self.n / 5, axis.get_ylim()[1] * .8,
                          r'$\chi_2 with \chi_1-states$', bbox=TEXTBOX)
            axis.set_xticklabels([])

            #TODO remove
            #if meth_type==2:
            #    axis.set_ylim(0,1)
            #if hoptype != "methyl":
            #    axis.set_ylim(0,0.2)
            #todo remove

        def plot_detectors():
            def tc_str(z):
                unit = 's'
                if z <= -12: z, unit = z + 15, 'fs'
                if z <= -9: z, unit = z + 12, 'ps'
                if z <= -6: z, unit = z + 9, 'ns'
                if z <= -3: z, unit = z + 6, r'$\mu$s'
                if z <= 0: z, unit = z + 3, 'ms'
                tc = np.round(10 ** z, -1 if z >= 2 else (0 if z >= 1 else 1))
                return '{:2} '.format(tc if tc < 10 else int(tc)) + unit
            D = DR.data()
            cts = self.cts[dix[residue]["ct_indices"]]
            D.load(Ct={'Ct':cts
                       ,'t': np.linspace(0,int(self.length/1000)*self.universe.trajectory.dt, self.length)})
            n_dets = self.n_dets if self.n_dets else 8
            remove_S2 = 1
            D.detect.r_auto3(n=n_dets)
            fit = D.fit()
            tick_labels = ['~' + tc_str(z0) for z0 in fit.sens.info.loc['z0']]
            tick_labels[0] = '<' + tick_labels[1][1:]
            #tick_labels[-1] = '>' + tick_labels[-2][1:]
            tick_labels = tick_labels[:n_dets-1]  # removed last label because no data were there
            page_2 = plt.figure()
            h = int((cts.shape[0]+1)/2)+1
            page_2.set_size_inches(8.3, 11.7/8*h)  # reduce page size for less cts
            ax0 = page_2.add_subplot(h, 1, 1)
            ax0.plot(D.sens.z(), fit.sens.rhoz().T)
            ax0.set_xlim(-14, -3)
            ax0.set_title("Detector Sensitivities")
            ax0.set_ylabel(r'$(1-S^2)$')
            ax0.set_xlabel(r'$\log_{10}(\tau_c/s)$')
            ax0.set_yticks([0, 1])
            ax = [page_2.add_subplot(h, 2, 3 + i) for i in range(cts.shape[0])]
            for d in range(n_dets - remove_S2):
                for r in range(cts.shape[0]):
                    ax[r].bar(d, fit.R[r, d])
            for r in range(cts.shape[0]):
                maxR = np.max(fit.R[r, :-remove_S2])
                ylim = np.round(maxR + .06,1) if maxR < .5 else .75 if maxR < .72 else 1
                print(ylim)
                ax[r].set_ylim(0, ylim)
                ax[r].text(n_dets/2, ylim * .75, dix[residue]["ct_labels"][r], bbox=TEXTBOX)
                ax[r].set_xticks(range(n_dets - remove_S2))
                if r < cts.shape[0]-2:
                    ax[r].set_xticklabels([])
                else:
                    ax[r].set_xticklabels(tick_labels, rotation=66)
                if r % 2 == 0:
                    ax[r].set_ylabel(r'$\rho_n^{(\Theta,S)}$')
            page_2.suptitle("{}_{}".format(residue, self.sel_sim_name))
            page_2.tight_layout()
            return page_2

        dix = self.full_dict
        part = self.part
        n = self.n
        if not exists(join(self.dir, "plots")):
            mkdir(join(self.dir, "plots"))
        num = dix[residue][dix[residue]["labels"][0]]["index"]  # index of the first (or only) methylgroup
        if len(dix[residue]["labels"]) == 2:
            two = True
            num2 = dix[residue][dix[residue]["labels"][1]]["index"]  # index of the second methylgroup
        else:
            two = False
        chi1 = dix[residue].get("chi1")  # index for chi1
        chi2 = dix[residue].get("chi2")  # index for chi2
        ### Initializing figure and axes
        page_1 = plt.figure()
        page_1.set_size_inches(8.3, 11.7)  # (14, 10)
        if chi1 is not None or chi2 is not None:
            ax_3d_plot = page_1.add_subplot(321, projection='3d')  # for plotting 3d distribution of methyl hydrogens
            rama_ax = page_1.add_subplot(322)
            if two:  # check if two methylgroups are available in the residue
                hopability_methyl_1_axis = page_1.add_subplot(625)  # if yes, create two hopability plots
                hopability_methyl_2_axis = page_1.add_subplot(627)
                if chi2 is not None:
                    hopability_methyl_1_axis_B = page_1.add_subplot(6, 2, 9)  # if yes, create two hopability plots
                    hopability_methyl_2_axis_B = page_1.add_subplot(6, 2, 11)
            else:
                hopability_single_methyl_axis = page_1.add_subplot(323)  # if not, only one (surprise)
            if chi2 is not None:  # same here if chi1 and chi2 are available
                ct_plot = page_1.add_subplot(326)
                chi1_hopability_plot = page_1.add_subplot(626)  # then make two plots
                chi2_hopability_plot = page_1.add_subplot(628)
                pie_chi2 = page_1.add_subplot(645)
                pie_chi1 = page_1.add_subplot(646)
                pie_chi2.pie([(self.chi2_areas[chi2]==0).sum(), (self.chi2_areas[chi2]==1).sum(),
                              (self.chi2_areas[chi2]==2).sum()], colors=["r", "g", "b"])
                pie_chi2.text(0,0,r'$\chi_2$', ha="center", va="center", bbox=TEXTBOX)
            else:
                ct_plot = page_1.add_subplot(313)
                chi1_hopability_ax_solo = page_1.add_subplot(324)  # or only one
                #tau_legend.append(r'$\chi_1 hops$')
                pie_chi1 = page_1.add_subplot(645)
            pie_chi1.pie([(self.chi1_areas[chi1]==0).sum(), (self.chi1_areas[chi1]==1).sum(),
                          (self.chi1_areas[chi1]==2).sum()], colors=["r", "g", "b"])
            pie_chi1.text(0,0,r'$\chi_1$', ha="center", va="center", bbox=TEXTBOX)
        else:
            # this part is mostly for Alanine
            ax_3d_plot = page_1.add_subplot(321, projection='3d')
            ct_plot = page_1.add_subplot(313)
            hopability_single_methyl_axis = page_1.add_subplot(312)  # if not, only one (surprise)
        ### start to plot, first the 3d scatter plot of the residue chain
        ax_3d_plot.set_axis_off()
        ax_3d_plot.scatter(self.coords_3Dplot[num, :, 0], self.coords_3Dplot[num, :, 1],
                           self.coords_3Dplot[num, :, 2], s=.5, c="g")  # areas[num])
        carbons = np.array([get_x_y_z_old(dix[residue]["C"].position, dix[residue]["CA"].position,
                                          dix[residue]["CB"].position, dix[residue][_].position) for _ in
                            dix[residue]["chain"]])
        ax_3d_plot.plot(*carbons.T, marker="o", markersize=10)
        for c, label in zip(carbons, dix[residue]["chain"]):
            ax_3d_plot.text(*c.T, label)
        ax_3d_plot.view_init(elev=0, azim=45)
        if len(dix[residue]["ct_indices"]):
            ct_plot.semilogx(np.arange(0, self.length),
                             self.cts[dix[residue]["ct_indices"]].T)
            for l,index in enumerate(dix[residue]["ct_indices"]):
                dix[residue]["ct_labels"][l] += " S²={:.2f}".format(self.S2s[index])
            ct_plot.legend(dix[residue]["ct_labels"])
            ct_plot.set_xlabel("timepoints")
            # todo calculate the correlation time by dt and stuff and set the xticks
            ct_plot.set_ylabel("C(t)")
            ct_plot.set_yticks([0, .5, 1])
            ct_plot.set_ylim(-.1, 1)
            ct_plot.set_title("Correlation functions")

        if two:  # todo this is still clumsy
            ax_3d_plot.scatter(self.coords_3Dplot[num2, :, 0], self.coords_3Dplot[num2, :, 1],
                               self.coords_3Dplot[num2, :, 2], s=.5, c="b")  # 3D Rama plot
            if chi2 is not None:
                plot_hopability_with_chi_states(hopability_methyl_1_axis, num, 1, chi2, 2)
                plot_hopability_with_chi_states(hopability_methyl_2_axis, num2, 2, chi2, 2)
                plot_hopability_with_chi_states(hopability_methyl_1_axis_B, num, 1, chi1, 1)
                plot_hopability_with_chi_states(hopability_methyl_2_axis_B, num2, 2, chi1, 1)
            elif chi1 is not None:
                plot_hopability_with_chi_states(hopability_methyl_1_axis, num, 1, chi1, 1)
                plot_hopability_with_chi_states(hopability_methyl_2_axis, num2, 2, chi1, 1)
            else:
                hopability_methyl_1_axis.bar(np.arange(n), self.hopability[num, :], color="black")
                hopability_methyl_2_axis.bar(np.arange(n), self.hopability[num2, :], color="black")
        else:
            hopability_single_methyl_axis.bar(np.arange(n) + .5, self.hopability[num], color="black")  # hopability
            hopability_single_methyl_axis.set_xlim(0, n)
            if chi1 is not None:
                plot_hopability_with_chi_states(hopability_single_methyl_axis, num, 1, chi1, 1)
        if chi1 is not None or chi2 is not None:
            if chi2 is not None:
                rama_ax.hist2d((self.dih_chi1[chi1] + 240) % 360, (self.dih_chi2[chi2] + 240) % 360,
                               range=[[0, 360], [0, 360]], bins=[180, 180], cmin=0.1)
                rama_ax.set_xlabel(r'$\chi_1$')
                rama_ax.set_ylabel(r'$\chi_2$')
                rama_ax.set_ylim(0, 360)
                rama_ax.set_yticks([0, 60, 180, 300, 360])
                chi1_hopability_plot.bar(np.arange(n) + .5, self.hopability_chi1[chi1], color="black")
                chi1_hopability_plot.set_xlim(0, n)
                chi1_hopability_plot.set_xticklabels([])
                chi1_hopability_plot.text(n / 10, chi1_hopability_plot.get_ylim()[1] * .8,
                                          r'$\chi_1 hop-probability$', bbox=TEXTBOX)
                plot_hopability_with_chi_states(chi2_hopability_plot,chi2,2,chi1,1,"chi")
            else:
                chi1_hopability_ax_solo.bar(np.arange(n) + .5, self.hopability_chi1[chi1], color="black")
                chi1_hopability_ax_solo.set_xlim(0, n)
                chi1_hopability_ax_solo.text(n / 10, chi1_hopability_ax_solo.get_ylim()[1] * .8,
                                             r'$\chi_1 hop-probability$', bbox=TEXTBOX)
                rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(0, 120, 3),
                             range=range(120), color="r")
                rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(120, 240, 3),
                             range=range(120), color="b")
                rama_ax.hist((self.dih_chi1[chi1] + 240) % 360, bins=range(240, 360, 3),
                             range=range(120), color="g")
                rama_ax.set_xlabel(r'$\chi_1$')
                rama_ax.set_ylabel("occurrence")
                rama_ax.set_yticklabels([])
            # final configuration of the plots
            rama_ax.set_xlim(0, 360)
            rama_ax.set_xticks([0, 60, 180, 300, 360])

        ct_plot.set_xlim(1, self.length)
        page_1.suptitle("{}_{}".format(residue, self.sel_sim_name))
        page_1.tight_layout()
        page_2 = plot_detectors()
        return page_1, page_2

def MethylCOrrealation():
    M = KaiMarkov(n=500,  simulation=2, residues=[])
    M.calc(sparse=10)
    M.universe.trajectory[0]
    #M.plot_all()
    #return
    fig = plt.figure()
    fig2 = plt.figure()
    fig3 = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    bx = fig2.add_subplot(111,projection="3d")
    cx = fig3.add_subplot(111, projection="3d")
    with open("MetRotCorr.bild","w+") as f:
        for key in M.full_dict.keys():
            for label in M.full_dict[key]["labels"]:
                pos = M.full_dict[key][label]["carbon"].position
                ax.scatter(*pos,color="black")
                ax.text(*pos,label)
                num = M.full_dict[key][label]["index"]
                for key2 in M.full_dict.keys():
                    for label2 in M.full_dict[key2]["labels"]:
                        if label ==  label2:
                            continue
                        pos2 = M.full_dict[key2][label2]["carbon"].position
                        num2 = M.full_dict[key2][label2]["index"]
                        corr = np.corrcoef(M.hopability[num],M.hopability[num2])[1][0]
                        if np.linalg.norm(pos-pos2) < 20 and np.abs(corr)>.35:
                            ax.plot([pos[0],pos2[0]],[pos[1],pos2[1]],[pos[2],pos2[2]], color =((np.abs(corr),0,0) if corr < 0 else (0,np.abs(corr),0)),
                                    alpha=corr**2)
                            f.write(".color {} {} {}\n".format(np.abs(corr) if corr < 0 else 0, corr if corr > 0 else 0, 0))
                            #f.write(".transparency {}\n".format(np.abs(corr)))
                            f.write(".cylinder {} {} {} {} {} {} {}\n".format(pos[0],pos[1],pos[2],pos2[0],pos2[1],pos2[2],np.abs(corr-.2)/4))
                chi1 = M.full_dict[key].get("chi1")
                if chi1 is not None:
                    corr = np.corrcoef(M.hopability[num],M.hopability_chi1[chi1])[1][0]
                    bx.scatter(*pos, color=(np.abs(corr),0,0) if corr < 0 else (0,corr,0), s=np.abs(corr)*100)
                    bx.text(*pos, label)
                chi2 = M.full_dict[key].get("chi2")
                if chi2 is not None:
                    corr = np.corrcoef(M.hopability[num],M.hopability_chi2[chi2])[1][0]
                    cx.scatter(*pos, color=(np.abs(corr),0,0) if corr < 0 else (0,corr,0), s=np.abs(corr)*100)
                    cx.text(*pos, label)



def get_detectors_for_simulation():
    M = KaiMarkov(simulation=0)
    #

    indices=[]
    labels = []

    for key in M.full_dict.keys():
        for i, label in enumerate(M.full_dict[key]['ct_labels']):
            if "CH" in label and "rot" in label:
                indices.append(M.full_dict[key]['ct_indices'][i])
                l = ""
                l+= key[4:7]
                l+= key[:3]
                if "1" in label:
                    l+="1"
                elif "2" in label:
                    l+="2"
                else:
                    l+="0"
                labels.append(l)
                print(M.full_dict[key]['ct_indices'][i],label)
    print(indices)
    print(len(indices))
    print(labels)
    indices = np.array(indices)
    M.calc()
    cts = M.cts[indices]
    print(cts.shape)


    D = DR.data()
    D.load(Ct={'Ct': cts
        , 't': np.linspace(0, int(M.length / 1000) * M.universe.trajectory.dt, M.length)})
    n_dets = 8
    D.detect.r_auto3(n=n_dets)
    fit = D.fit()
    fit.label = labels
    print(fit.R.shape)
    marked_atom = {"ala0":"CB",
                   "ile1":"CG2",
                   "ile2":"CD",
                   "leu1":"CD1",
                   "leu2":"CD2",
                   "val1":"CG1",
                   "val2":"CG2",
                   "thr0":"CG2"}

    with open("det2.txt","w+") as f:
        for i,lab in enumerate(fit.label):
            try:
                print(lab)
                f.write(str(int(lab[:3])-147))
                f.write("-")
                f.write(marked_atom[lab[3:].lower()])
                f.write(":")
                for R in fit.R[i,:]:
                    f.write(str(R.astype("float16")))
                    f.write(";")
                f.write("\n")
            except:
                pass


@time_runtime
def main():
    #M = KaiMarkov(simulation=1)
    #M.calc_new(sparse=0)
    #plt.plot(M.cts.T)
    #for key in M.sim_dict['residues'].keys():
    #    M.plot_single_res(key)

    M = KaiMarkov(simulation=0, residues=[12,58,9,57])
    M.calc_new(sparse=10,force_calc=True)
    # plt.plot(M.cts.T)
    for key in M.sim_dict['residues'].keys():
        M.plot_single_res(key)


if __name__ == "__main__":
    main()
