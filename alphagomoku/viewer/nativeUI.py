import numpy as np 
import sys
from PyQt5.QtWidgets import QWidget, QApplication,QDesktopWidget
from PyQt5.QtGui import QPainter, QColor, QBrush, QPalette

class nativeUI:
    def __init__(self,chessboardinfo,sizeunit):
        self.chessboardinfo = chessboardinfo
        self.sizeunit = sizeunit

    def show():
        pass

class Chessboard(QWidget):
    def __init__(self,chessboardinfo,sizeunit):
        super().__init__()
        self.chessboardinfo = chessboardinfo
        self.sizeunit = sizeunit
        self.R = 0.4*sizeunit
        self.initUI()

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

        self.show()

    def paintEvent(self, e):
        qp = QPainter()
        qp.begin(self)
        self.drawChessboard(qp)
        self.drawChesses(qp)
        qp.end()

    def drawChessboard(self,qp):
        (Nx,Ny) = self.chessboardinfo.shape
        qp.setPen(QColor(0, 0, 0))
        for i in range(Nx):
            qp.drawLine((i+0.5)*self.sizeunit, 0, (i+0.5)*self.sizeunit,Ny*self.sizeunit)   
        for j in range(Ny):
            qp.drawLine(0, (j+0.5)*self.sizeunit, Ny*self.sizeunit, (j+0.5)*self.sizeunit) 

    def drawChesses(self, qp):
        (Nx,Ny) = self.chessboardinfo.shape
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
    ex = Chessboard(chessboardinfo=chessboardinfo,sizeunit=sizeunit)
    sys.exit(app.exec_())

