import sys
from chimerax.core.commands import run as rc

rc(session,"remotecontrol rest start port 60958")
sys.path.append("/Users/albertsmith/Documents/GitHub/pyDR/chimeraX")
from RemoteCMXside import CMXReceiver as CMXR
import RemoteCMXside
cmxr=CMXR(session,7000,rc_port0=60958)
rc(session,"ui mousemode right select")
rc(session,"open pdbs/processed.pdb")
rc(session,"~ribbon ~/B")
rc(session,"style ball")
rc(session,"hide H")
