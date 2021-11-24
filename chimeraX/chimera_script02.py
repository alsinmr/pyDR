import sys
from chimerax.core.commands import run as rc

rc(session,"remotecontrol rest start port 60960")
sys.path.append("/Users/albertsmith/Documents/GitHub/pyDR/chimeraX")
from RemoteCMXside import CMXReceiver as CMXR
import RemoteCMXside
cmxr=CMXR(session,7002)
rc(session,"ui mousemode right select")
rc(session,"open 2kj3")
rc(session,"~ribbon")
rc(session,"show #1.1@N,C,CA")
rc(session,"style ball")
