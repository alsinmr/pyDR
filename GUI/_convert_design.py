#!/usr/bin/python3
from os import system
# creating the python files from the files created with QtDesigner
system("pyuic5 designer/MainWindow.ui > designer2py/mymainwindow.py")
system("pyuic5 designer/data_widget.ui > designer2py/data_widget.py")
system("pyuic5 designer/sensitivity_widget.ui > designer2py/sensitivity_widget.py")
system("pyuic5 designer/detectors_widget.ui > designer2py/detectors_widget.py")
system("pyuic5 designer/ired_widget.ui > designer2py/ired_widget.py")
system("pyuic5 designer/frames_widget.ui > designer2py/frames_widget.py")
system("pyuic5 designer/selection_widget.ui > designer2py/selection_widget.py")
system("pyuic5 designer/mdsimulation_widget.ui > designer2py/mdsimulation_widget.py")
