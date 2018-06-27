import sys
sys.path.append("..")

import naiveai
import Chessboard
import Gameengine
import viewer

#from PyQt5.QtWidgets import QApplication
#from PyQt5.QtCore import * 
#from PyQt5.QtGui import *
from Computer import Computer
from Human import Human

#class Playwithnaiveai(QApplication):
class Playwithnaiveai:
    def __init__(self):
        #super().__init__(sys.argv)
        #self.playsignal = pyqtSignal(tuple) 
        self.computer = naiveai.naivecomputer2.Computer()

        players = [Computer(), Human()]
        self.chessboard = Chessboard.Chessboard(players=players)

        self.waitingforplay = True

    def playerplay(self,chesspos):
        #print("Human")
        self.chessboard.playchess(chesspos,role='Human')
        self.waitingforplay = False
        
    def init(self):
        self.ui = viewer.UI(pressaction=self.playerplay,chessboardinfo=self.chessboard.get_chessboard())
        self.ui.start()

        status,winner = self.chessboard.victoryjudge(role='Computer')
        while not status:
            while self.waitingforplay:
                pass 
            status,winner = self.chessboard.victoryjudge(role='Human')
            if status:
                self.endgame(role='Human')

            chesspos = self.computer.play(self.chessboard.get_chessboard())
            self.chessboard.playchess(chesspos,role='Computer')
            status,winner = self.chessboard.victoryjudge(role='Computer')
            if status:
                self.endgame(role='Computer')
            
            self.ui.setchessboard(self.chessboard.get_chessboard())
            self.waitingforplay = True
        
    def endgame(self,role):
        print(role + " Win")
        self.ui.gameend(role)
        sys.exit()

if __name__=='__main__':
    test = Playwithnaiveai()
    test.init()
    #test.exec_()