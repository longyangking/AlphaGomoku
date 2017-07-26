import numpy as np 
import Config

class Chessboard:
    def __init__(self):
        self.chessboardheight = Config.ChessBoardHeight
        self.chessboardwidth = Config.ChessBoardWidth
        self.chessboard = np.zeros((self.chessboardwidth,self.chessboardheight))

    def playchess(self,pos,role):
        if self.chessboard[pos] == 0:
            self.chessboard[pos] = Config.ChessInfo[role]
        else:
            return False
        return True

    def chessboardinfo(self):
        return self.chessboard

    def victoryjudge(self,role):
        count = 0
        for i in range(self.chessboardwidth):
            for j in range(self.chessboardheight):
                count = 
                if count == 5*Config.ChessInfo[role]