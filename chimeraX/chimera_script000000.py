from multiprocessing.connection import Listener,Client
from time import time
t0=time()
while time()-t0<10:
	try:
		client=Client(("localhost",7000),authkey=b"pyDIFRATE password")
		break
	except:
		pass
sel=list()
for k,mdl in enumerate(session.models):
	if mdl.selected:
		sel.append({"model":k})
		a0,a1=mdl.bonds[mdl.bonds.selected].atoms
		sel[-1]["b0"]=a0.coord_indices
		sel[-1]["b1"]=a1.coord_indices
		sel[-1]["a"]=mdl.atoms[mdl.atoms.selected].coord_indices
client.send(sel)
client.close()
