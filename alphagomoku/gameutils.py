from __future__ import absolute_import

import numpy as np 
import Config

import h5py
import numpy as np 
import time
import copy

class Gamedata:
    def __init__(self,board_shape,data=None):
        self.date =  time.strftime("%Y-%m-%d(%H:%M:%S)", time.localtime())

        self.list = data
        if data is None:
            self.list = list()
        self.totalsteps = 0
        self.board_shape = board_shape
        self.winner = None

    def init(self,board_shape,data=None):
        self.__init__(board_shape=board_shape,data=data)

    def append(self,chessboardinfo):
        self.list.append(copy.deepcopy(chessboardinfo))
        self.totalsteps += 1

    def gameend(self,winner):
        self.winner = winner

    def getinfo(self):
        return self.winner, self.totalsteps

    def getstepinfo(self):
        return self.totalsteps
        
    def getdatas(self,indexs):
        data = list()
        for i in indexs:
            if (i<0) or (i>=self.totalsteps):
                data.append(np.zeros(self.board_shape))
            else:
                data.append(self.list[i])
        #if (min(indexs) < 0) and (max(index) >= self.totalsteps):
        #    return None
        #data = [self.list[i] for i in indexs]
        return data

    def getdata(self,index):
        if (index<0) or (index>=self.totalsteps):
            return None
        else:
            return copy.deepcopy(self.list[index])

    def getdatashape(self):
        return self.board_shape

    def getalldata(self):
        return self.winner, copy.deepcopy(self.list)

    def getstate(self,steps=3):
        '''
        Get state vector (for neural network)
        '''
        state = list()
        if self.totalsteps == 0:
            for _ in range(steps):
                state.append(np.zeros(self.board_shape))
            return state
        if self.totalsteps < steps:
            for _ in range(steps-self.totalsteps):
                state.append(np.zeros(self.board_shape))
            for i in range(self.totalsteps):
                state.append(self.list[i])
            return state
        for i in range(self.totalsteps-steps,self.totalsteps):
            state.append(self.list[i])
        return state

    def getwinner(self):
        return self.winner

    def load(self,filename):
        gamefile = h5py.File(filename,"r")
        self.date = gamefile["date"][...]
        self.playerA = gamefile["playerA"][...]
        self.playerB = gamefile["playerB"][...]
        self.winner = gamefile["winner"][...]
        self.totalsteps = gamefile["totalsteps"][...]
        self.list = list()
        for i in range(self.totalsteps):
            chessboard = gamefile["chessboards"]["step_{i}".format(i=i)][...]
            self.list.append(chessboard)
        gamefile.close()

    def save(self,filename=None,description=""):
        '''
        Save all useful information into HDF5-form file:
            + Date
            + Player info
            + Winner info
            + Total play steps
            + Chessboard info for each step
            + Descriptions/Remarks
        '''
        if filename is None:
            filename = self.date

        gamefile = h5py.File(filename+".hdf5", "w")
        gamefile.create_dataset("date", data=self.date)
        gamefile.create_dataset("playerA",data=self.playerB)
        gamefile.create_dataset("playerB",data=self.playerB)
        gamefile.create_dataset("winner",data=self.winner)
        gamefile.create_dataset("totalsteps",data=self.totalsteps)
        chessboards = gamefile.create_group("chessboards")
        for i in range(self.list):
            chessboards["step_{i}".format(i=i)] = self.list[i]

        gamefile.create_dataset("description",data=description)
        gamefile.close()

class Chessboard:
    def __init__(self, players, board_size=None):
        self.players = players

        self.chessboardheight = Config.ChessBoardHeight
        self.chessboardwidth = Config.ChessBoardWidth
        if board_size is not None:
            self.chessboardheight, self.chessboardwidth = board_size
        
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

    def rec2pos(self, rec_pos):
        #if type(rec_pos) is not list:
        #    return (rec_pos%self.chessboardwidth, int(rec_pos/self.chessboardwidth))
        x = [pos%self.chessboardwidth for pos in rec_pos]
        y = [int(pos/self.chessboardwidth) for pos in rec_pos]
        return [x,y]

    def pos2rec(self, pos):
        x,y = pos
        rec_pos = x + y*self.chessboardwidth
        return rec_pos

    def playchess_rec(self,pos_rec,role):
        pos = (pos_rec%self.chessboardwidth,int(pos_rec/self.chessboardwidth))
        if self.is_gameend:
            return False
        if self.chessboard[pos] == 0:
            self.chessboard[pos] = Config.ChessInfo[role]
        else:
            return False
        self.gamedata.append(chessboardinfo=self.chessboard)
        return True

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

    def get_stepinfo(self):
        return self.gamedata.getstepinfo()

    def get_state(self,steps=3):
        return self.gamedata.getstate(steps)

    def get_chessboardinfo(self,role):
        chess_value = Config.ChessInfo[role]
        if chess_value < 0:
            return copy.deepcopy(-self.chessboard)
        return copy.deepcopy(self.chessboard)

    def get_chessboard(self):
        return copy.deepcopy(self.chessboard)
        
    def is_available(self):
        return np.sum(self.chessboard==0)>0
    
    def get_availables(self):
        positions = np.transpose(np.where(self.chessboard==0))
        rec_pos = positions[:,0] + self.chessboardwidth*positions[:,1]
        return positions, rec_pos

    def get_status(self):
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

class GameEngine:
    def __init__(self,playerA,playerB,verbose=False):
        self.playerA = playerA
        self.playerB = playerB
        self.chessboard = Chessboard.Chessboard(players=[playerA,playerB])
        self.verbose = verbose

    def init(self):
        self.chessboard.init()

    def run(self,op='A'):
        # Offensive Position, default: Player A
        if op == 'A':
            players = [self.playerA,self.playerB]
        else:
            players = [self.playerB,self.playerA]

        status,winner = self.chessboard.get_status()
        # The game will continue to battle it out
        while not status:
            for i in range(len(players)):
                chesspos = players[i].play(self.chessboard)
                while not self.chessboard.playchess(chesspos,role=players[i].role):
                    chesspos = players[i].play(self.chessboard)
                # Save chessboard info
                status,winner = self.chessboard.victoryjudge(role=players[i].role)
                if status:
                    break
                if self.verbose:
                    print("{step}th Self-play step ...".format(step=self.chessboard.get_stepinfo()))
        return self.chessboard.get_game_data()
