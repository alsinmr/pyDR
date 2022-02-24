import sys
from chimerax.core.commands import run as rc

rc(session,"remotecontrol rest start port 60959")
sys.path.append("/home/kaizumpfe/PycharmProjects/pyDR/chimeraX")
from RemoteCMXside import CMXReceiver as CMXR
import RemoteCMXside
cmxr=CMXR(session,7001,rc_port0=60959)
rc(session,"ui mousemode right select")
rc(session,"open pdbs/processed.pdb")
rc(session,"~ribbon ~/B")
rc(session,"show")
rc(session,"~show ~/B")
rc(session,"style ball")
rc(session,"hide H")
