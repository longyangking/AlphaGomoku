import numpy as np 
import threading
from . import nativeUI

import sys
import Config
from PyQt5.QtWidgets import QWidget, QApplication,QDesktopWidget
from PyQt5.QtCore import * 
from PyQt5.QtGui import *

class UI(threading.Thread):
    def __init__(self,pressaction,chessboardinfo,sizeunit=50):
        threading.Thread.__init__(self)
        self.ui = None
        self.app = None

        self.chessboardinfo = chessboardinfo
        self.sizeunit = sizeunit
        self.pressaction = pressaction
    
    def run(self):
        print('Init UI...')
        self.app = QApplication(sys.argv)
        self.UI = nativeUI.nativeUI(pressaction=self.pressaction,chessboardinfo=self.chessboardinfo)
        self.app.exec_()

    def setchessboard(self,chessboardinfo):
        return self.UI.setchessboard(chessboardinfo)
    
    def gameend(self,role):
        self.UI.gameend(winner=role)