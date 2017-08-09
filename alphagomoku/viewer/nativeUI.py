import numpy as np 
import sys
import Config
from PyQt5.QtWidgets import QWidget, QApplication,QDesktopWidget
from PyQt5.QtCore import QRect
from PyQt5.QtGui import QPainter, QColor, QBrush, QPalette

class nativeUI(QWidget):
    def __init__(self,chessboardinfo,sizeunit,role="Human"):
        super().__init__()
        self.chessboardinfo = chessboardinfo
        self.sizeunit = sizeunit
        self.R = 0.4*sizeunit

        self.mousex = 0
        self.mousey = 0

        self.chooseX = 0
        self.chooseY = 0
        self.playstatus = True
        self.chessvalue = Config.ChessInfo['Human']
        self.initUI()

    def getchessboardinfo(self):
        return self.chessboardinfo

    def setchessboard(self,chessboardinfo):
        self.chessboardinfo = chessboardinfo
        self.playstatus = True

    def initUI(self):
        (Nx,Ny) = self.chessboardinfo.shape
        screen = QDesktopWidget().screenGeometry()
        size =  self.geometry()

        self.setGeometry((screen.width()-size.width())/2, 
                        (screen.height()-size.height())/2,
                        Nx*self.sizeunit, Ny*self.sizeunit)
        self.setWindowTitle("Gomoku")

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
        self.chooseChess(qp)
        if not self.playstatus:
            self.waitforanother(qp)
        qp.end()

    def waitforanother(self,qp):
        size =  self.geometry()
        qp.setPen(0)
        qp.setBrush(QColor(200, 200, 200, 180))
        width = size.width()/4*3
        height = size.height()/3
        qp.drawRect(size.width()/2-width/2, size.height()/2-height/2, width, height)

        qp.setPen(QColor(0,0,0))
        font = qp.font()
        font.setPixelSize(60)
        qp.setFont(font)
        qp.drawText(QRect(size.width()/2-width/2, size.height()/2-height/2, width, height),	0x0004|0x0080,str("Waiting..."))
       
    def mouseMoveEvent(self,e):
        self.mousex = int(e.x()/self.sizeunit)
        self.mousey = int(e.y()/self.sizeunit)
        self.update() 
    
    def mousePressEvent(self,e):
        X = int(e.x()/self.sizeunit)
        Y = int(e.y()/self.sizeunit)
        if (self.chessboardinfo[X,Y] == 0) and self.playstatus:
            self.chooseX = X
            self.chooseY = Y
            self.chessboardinfo[X,Y] = self.chessvalue
            self.playstatus = False
            self.update()

    def chooseChess(self,qp):
        #qp.setBrush(QColor(0, 0, 0))
        qp.setPen(QColor(255, 0, 0))
        qp.setBrush(0)
        qp.drawEllipse((self.mousex+0.5)*self.sizeunit-self.R,
                (self.mousey+0.5)*self.sizeunit-self.R, 
                2*self.R, 2*self.R)

    def drawChessboard(self,qp):
        (Nx,Ny) = self.chessboardinfo.shape
        qp.setPen(QColor(0, 0, 0))
        for i in range(Nx):
            qp.drawLine((i+0.5)*self.sizeunit, 0, (i+0.5)*self.sizeunit,Ny*self.sizeunit)   
        for j in range(Ny):
            qp.drawLine(0, (j+0.5)*self.sizeunit, Ny*self.sizeunit, (j+0.5)*self.sizeunit) 

    def drawChesses(self, qp):
        (Nx,Ny) = self.chessboardinfo.shape
        qp.setPen(0)
        for i in range(Nx):
            for j in range(Ny):
                if self.chessboardinfo[i,j] == 1:
                    qp.setBrush(QColor(0, 0, 0))
                elif self.chessboardinfo[i,j] == -1:
                    qp.setBrush(QColor(255, 255, 255))
                if self.chessboardinfo[i,j] != 0:
                    qp.drawEllipse((i+0.5)*self.sizeunit-self.R, (j+0.5)*self.sizeunit-self.R, 2*self.R, 2*self.R)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    chessboardinfo = np.random.randint(-1,2,size=(10,10))
    sizeunit = 50
    ex = nativeUI(chessboardinfo=chessboardinfo,sizeunit=sizeunit)
    sys.exit(app.exec_())

