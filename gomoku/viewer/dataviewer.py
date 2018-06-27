import numpy as np 
import sys
import Config
import threading

from PyQt5.QtWidgets import *
from PyQt5.QtCore import * 
from PyQt5.QtGui import *

class Dataplot(QWidget):
    def __init__(self, gamedata, sizeunit=50):
        super(Dataplot,self).__init__(None)
        
        self.gamedata = gamedata
        self.sizeunit = sizeunit
        self.R = 0.4*sizeunit

        winner, totalsteps = self.gamedata.getinfo()
        self.index = 0
        self.initUI()

    def initUI(self):
        (Nx,Ny) = self.gamedata.getdatashape()
        screen = QDesktopWidget().screenGeometry()
        size =  self.geometry()

        self.setGeometry((screen.width()-size.width())/2, 
                        (screen.height()-size.height())/2,
                        Nx*self.sizeunit, Nx*self.sizeunit)
        self.setWindowTitle("Gomoku Dataviwer: {index}".format(index=self.index+1))

        self.setFixedSize(Nx*self.sizeunit, Nx*self.sizeunit)
        # sld = QSlider(Qt.Horizontal, self)
        # sld.setFocusPolicy(Qt.NoFocus)
        # sld.setGeometry((Nx-3)*self.sizeunit, (Ny-2)*self.sizeunit, 2*self.sizeunit, self.sizeunit)

        # set Background color
        palette =  QPalette()
        palette.setColor(self.backgroundRole(), QColor(211, 169, 105))
        self.setPalette(palette)

        self.setMouseTracking(True)
        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawChessboard(qp)
        self.drawChesses(qp)
        # self.chooseChess(qp)
        qp.end()

    def drawChessboard(self,qp):
        (Nx,Ny) = self.gamedata.getdatashape()
        qp.setPen(QColor(0, 0, 0))
        for i in range(Nx):
            qp.drawLine((i+0.5)*self.sizeunit, 0, (i+0.5)*self.sizeunit,Ny*self.sizeunit)   
        for j in range(Ny):
            qp.drawLine(0, (j+0.5)*self.sizeunit, Ny*self.sizeunit, (j+0.5)*self.sizeunit) 

    def drawChesses(self, qp):
        (Nx,Ny) = self.gamedata.getdatashape()
        chessboardinfo = self.gamedata.getdata(self.index)
        qp.setPen(0)
        for i in range(Nx):
            for j in range(Ny):
                if chessboardinfo[i,j] == 1:
                    qp.setBrush(QColor(0, 0, 0))
                elif chessboardinfo[i,j] == -1:
                    qp.setBrush(QColor(255, 255, 255))
                if chessboardinfo[i,j] != 0:
                    qp.drawEllipse((i+0.5)*self.sizeunit-self.R, (j+0.5)*self.sizeunit-self.R, 2*self.R, 2*self.R)

    def keyPressEvent(self,e):
        winner, totalsteps = self.gamedata.getinfo()
        if e.key() == Qt.Key_Right:
            self.index += 1
            if self.index >= totalsteps:
                self.index = totalsteps - 1
        elif e.key() == Qt.Key_Left:
            self.index -= 1
            if self.index < 0:
                self.index = 0

        self.setWindowTitle("Gomoku Dataviwer: {index}".format(index=self.index+1))
        self.update()

class Dataviewer(threading.Thread):
    def __init__(self,gamedata,sizeunit=50):
        threading.Thread.__init__(self)
        self.ui = None
        self.app = None

        self.gamedata = gamedata
        self.sizeunit = sizeunit
    
    def run(self,verbose=False):
        if verbose:
            print('Init Dataviwer UI...')
        self.app = QApplication(sys.argv)
        self.UI = Dataplot(gamedata=self.gamedata,sizeunit=self.sizeunit)
        self.app.exec_()

