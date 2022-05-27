import pyDR

def main():
    pdb = "../HETs_clean.pdb"
    xtc = "../xtcs/MET_3pw.xtc"
    molsys = pyDR.Selection.MolSys.MolSys(pdb, xtc, tf=100000,step =1)
    molsel = pyDR.Selection.MolSys.MolSelect(molsys)
    molsel.select_bond(Nuc="15N", segids="B")

    try:
        frames = pyDR.Frames.FrameObj(molsel)
        x = frames.md2iRED()
        print(x)
        print(frames)
        return
    except Exception as E:
        print(E)

    print(molsys)
    print(molsel)



    ired = pyDR.iRED(molsel)
    #ired.get_vecs()
    ired.full_analysis()
    print(ired)


main()