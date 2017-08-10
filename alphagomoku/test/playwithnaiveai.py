import sys
sys.path.append("..")

import naivecomputer
import Chessboard
import Gameengine
import viewer

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import * 
from PyQt5.QtGui import *

class Playwithnaiveai(QApplication):
    def __init__(self):
        super().__init__(sys.argv)
        #self.playsignal = pyqtSignal(tuple) 
        self.computer = naivecomputer.Computer()
        self.chessboard = Chessboard.Chessboard()

    def init(self):
        self.ui = viewer.nativeUI.nativeUI(pressaction=self.playerplay, chessboardinfo=self.chessboard.chessboardinfo())
 
    def playerplay(self,chesspos):
        #print("Human")
        self.chessboard.playchess(chesspos,role='Human')
        status,winner = self.chessboard.victoryjudge(role='Human')
        if status:
            print("Human Win")
            sys.exit()

        chesspos = self.computer.play(self.chessboard.chessboardinfo())
        self.chessboard.playchess(chesspos,role='Computer')
        status,winner = self.chessboard.victoryjudge(role='Computer')
        if status:
            print("Computer Win")
            sys.exit()
        
        self.ui.setchessboard(self.chessboard.chessboardinfo())
        
    def start(self,op=True):
        #self.ui.run()

        #roles = ['Human','Computer']
        #if not op:
        #    roles = ['Computer','Human']
        #if op:
        #    self.ui.setchessboard(self.chessboard.chessboardinfo())
        status = False
        playstatus = False

        #self.connect(self.ui, SIGNAL("PlayerPlayed"), self.updateUi) 
 

        while not status:
            for i in range(len(roles)):
                #if op and (i==0) and computerplayed:
                #    playstatus,chesspos = self.ui.getplaystatus()
                #    while not playstatus:
                #        playstatus,chesspos = self.ui.getplaystatus()
                #elif (not op) and (i!=0) and computerplayed:
                #    playstatus,chesspos = self.ui.getplaystatus()
                #    while not playstatus:
                #        playstatus,chesspos = self.ui.getplaystatus()
                #else:
                #    chessboardinfo = self.ui.getchessboardinfo()
                #    chesspos = self.computer.play(chessboardinfo)
                #    computerplayed = True
                
                
                self.chessboard.playchess(chesspos,role=roles[i])
                status,winner = self.chessboard.victoryjudge(roles[i])
                if computerplayed:
                    self.ui.setchessboard(self.chessboard.chessboardinfo())
                if status:
                    break
        return winner

if __name__=='__main__':
    test = Playwithnaiveai()
    test.init()
    test.exec_()