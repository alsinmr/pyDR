import pyDR.Project as Project
from pyDR.GUI import QMainWindow   # this is important to import because of the get_project function! Maybe there is a
                                    # better solution
from pyDR.GUI import MyWindow
from PyQt5.QtWidgets import QApplication, QMessageBox, QInputDialog
from pyDR.GUI.other.elements import openFileNameDialog
import sys
import os.path


if __name__ == '__main__':
    """
    Open pyDR via commandline with 'python3 -m pyDR'
    Optional 'python3 -m pyDR PROJECTNAME' will skip the process the starting wizard
    
    This will create the GUI which will ask you to load an existing or create a new project.
    
    """
    project = None
    if len(sys.argv) > 1:
        #print(sys.argv)
        if os.path.exists(sys.argv[1]):
            project = Project(sys.argv[1])
        else:
            print("A project with this name doesn't exist")
    app = QApplication(sys.argv)

    # to initialize the GUI we need a project first. This will open a MessageBox to load an existing one or create
    # a new one. Skip this, if the project name was the first agument on program call
    if project is None:
        msg = QMessageBox()  # create an instance of it
        msg.setIcon(QMessageBox.Information)  # set icon
        msg.setText("Open an existing project or create a new one")  # set text
        msg.setInformativeText("Press Ok to create a new one")  # set information under the main text
        msg.setWindowTitle("Run pyDR")  # set title
        msg.setDetailedText("The details are as follows:")  # a button for more details will add in
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Open)  # type of buttons associated
        #msg.buttonClicked.connect(myfunc)  # connect clicked signal
        return_value = msg.exec_()  # get the return value

        if return_value == 1024:   # this is the Ok Button
            project_name, ok = QInputDialog().getText(None, "ProjectName","Name",text="GlpG_pyDR")
            assert ok, "If you don't want to..."
            assert len(project_name), "Project Name must have some characters"
            create = True
            if os.path.exists(project_name):
                QMessageBox(icon=QMessageBox.Warning, text="A project with this name already exists and will be loaded").exec_()
                create = False

            #todo think about making a default project folder which will be scanned always
            project = Project(project_name, create=create)

        elif return_value == 8192:  # this is the open button
            # todo add file dialog to open existing project
            print("Open File Dialog and load project")
            filename = openFileNameDialog(folder=True)
            assert filename, "You have to select a project file"
            # todo this here might get a little tricky since there is not directly a file to open right now
            #  we might consider opening a folder, but I havent make up my mind how we solve that properly

    # leve this line for testing purposes right now
    # print("value of pressed message box button:", str(return_value))

    # after the project was initialized, finally build the GUI and connect the project
    Window = QMainWindow()
    ui = MyWindow()
    ui.setupUi(Window, project)
    Window.setWindowTitle("pyDR")
    Window.show()
    sys.exit(app.exec_())
