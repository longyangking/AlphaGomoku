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
        flag = False
        for i in range(self.chessboardwidth):
            for j in range(self.chessboardheight):
                if ((i-4 >= 0) and (j-4 >= 0)):
                    count = chessboard[i,j] \
                        + chessboard[i-1,j-1] \
                        + chessboard[i-2,j-2] \
                        + chessboard[i-3,j-3] \
                        + chessboard[i-4,j-4]
                    if count == 5*Config.ChessInfo[role]:
                        flag =  True
                        return flag,role

                if (j-4 >= 0):
                    count = chessboard[i,j] \
                        + chessboard[i,j-1] \
                        + chessboard[i,j-2] \
                        + chessboard[i,j-3] \
                        + chessboard[i,j-4]
                    if count == 5*Config.ChessInfo[role]:
                        flag =  True
                        return flag,role

                if ((i+4 < self.chessboardwidth) and (j-4 >= 0)):
                    count = chessboard[i,j] \
                        + chessboard[i+1,j-1] \
                        + chessboard[i+2,j-2] \
                        + chessboard[i+3,j-3] \
                        + chessboard[i+4,j-4]
                    if count == 5*Config.ChessInfo[role]:
                        flag =  True
                        return flag,role

                if (i+4 < self.chessboardwidth):
                    count = chessboard[i,j] \
                        + chessboard[i+1,j] \
                        + chessboard[i+2,j] \
                        + chessboard[i+3,j] \
                        + chessboard[i+4,j]
                    if count == 5*Config.ChessInfo[role]:
                        flag =  True
                        return flag,role
                    
                if ((i+4 < self.chessboardwidth) and (j+4 < self.chessboardheight)):
                    count = chessboard[i,j] \
                        + chessboard[i+1,j+1] \
                        + chessboard[i+2,j+2] \
                        + chessboard[i+3,j+3] \
                        + chessboard[i+4,j+4]
                    if count == 5*Config.ChessInfo[role]:
                        flag =  True
                        return flag,role

                if (j+4 < self.chessboardheight):
                    count = chessboard[i,j] \
                        + chessboard[i,j+1] \
                        + chessboard[i,j+2] \
                        + chessboard[i,j+3] \
                        + chessboard[i,j+4]
                    if count == 5*Config.ChessInfo[role]:
                        flag =  True
                        return flag,role
                    
                if ((i-4 >= 0) and (j+4 < self.chessboardheight)):
                    count = chessboard[i,j] \
                        + chessboard[i-1,j+1] \
                        + chessboard[i-2,j+2] \
                        + chessboard[i-3,j+3] \
                        + chessboard[i-4,j+4]
                    if count == 5*Config.ChessInfo[role]:
                        flag =  True
                        return flag,role

                if (i-4 >= 0):
                    count = chessboard[i,j] \
                        + chessboard[i-1,j] \
                        + chessboard[i-2,j] \
                        + chessboard[i-3,j] \
                        + chessboard[i-4,j]
                    if count == 5*Config.ChessInfo[role]:
                        flag =  True
                        return flag,role
        
        return flag,None