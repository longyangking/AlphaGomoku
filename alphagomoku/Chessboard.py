from __future__ import absolute_import

import numpy as np 
import Config
from Gamedata import Gamedata
import copy

class Chessboard:
    def __init__(self,players):
        self.players = players

        self.chessboardheight = Config.ChessBoardHeight
        self.chessboardwidth = Config.ChessBoardWidth
        self.shape = (self.chessboardwidth,self.chessboardheight)
        self.chessboard = np.zeros(self.shape)
        self.gamedata = Gamedata(board_shape=(self.chessboardheight,self.chessboardwidth))
        self.is_gameend = False
        
    def init(self):
        self.chessboard = np.zeros(self.shape)
        self.gamedata.init(board_shape=self.shape)
        self.is_gameend = False

    def get_shape(self):
        return self.shape

    def playchess(self,pos,role):
        if self.is_gameend:
            return False
        if self.chessboard[pos] == 0:
            self.chessboard[pos] = Config.ChessInfo[role]
        else:
            return False
        self.gamedata.append(chessboardinfo=self.chessboard)
        return True

    def get_game_data(self):
        return self.gamedata

    def get_data(self,indexs):
        return self.gamedata.getdata(indexs)

    def get_state(self,steps=3):
        return self.gamedata.getstate(steps)

    def get_chessboardinfo(self):
        return copy.deepcopy(self.chessboard)
        
    def is_available(self):
        return np.sum(self.chessboard==0)>0

    def get_status(self,role):
        for player in self.players:
            flag,role = self.victoryjudge(role=player.role)
            if flag:
                break
        return flag, role

    def get_roles(self):
        roles = [player.role for player in self.players]
        return roles

    def victoryjudge(self,role):
        flag,role = self._evaluate(role=role)
        if flag:
            self.is_gameend = True
            self.gamedata.gameend(winner=role)
        return flag, role

    def _evaluate(self,role):
        count = 0
        flag = False
        chessboard = self.chessboard
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