from __future__ import absolute_import

import numpy as np 
import Config

import h5py
import numpy as np 
import time
import copy



class ChessBoard:
    def __init__(self, board_shape, 
        chessboard=None,
        current_player=0
    ):
        self.board_shape = board_shape

        if chessboard is not None:
            self.chessboard = chessboard
        else:
            self.chessboard = np.zeros(board_shape)

        self.player_chesses = [1, -1]
        self.current_player = current_player

    def update_current_player(self):
        self.current_player += 1
        self.current_player %= 2

    def get_shape(self):
        return np.copy(self.board_shape)

    def get_board(self):
        return np.copy(self.chessboard)

    def clone(self):
        '''
        Clone Chess Board
        '''
        chessboard = ChessBoard(
            board_shape=self.get_shape(),
            chessboard=self.get_board(),
            current_player=self.current_player
            )
        return chessboard

    def play(self, pos):
        chessboardwidth, chessboardheight = self.board_shape
        pos = (pos%chessboardwidth,int(pos/chessboardwidth))
        if self.chessboard[pos] == 0:
            self.chessboard[pos] = self.player_chesses[self.current_player]
        else:
            return False

        flag = self.__evaluate()
        if not flag:
            self.update_current_player()
        return flag

    def get_availables(self):
        chessboardwidth, chessboardheight = self.board_shape
        positions = np.transpose(np.where(self.chessboard==0))
        actions = positions[:,0] + chessboardwidth*positions[:,1]
        return np.copy(actions)

    def __evaluate(self):
        count = 0
        flag = False

        chessboardwidth, chessboardheight = self.board_shape
        chessboard = self.chessboard
        chess_value = self.player_chesses[self.current_player]

        for i in range(chessboardwidth):
            for j in range(chessboardheight):
                if ((i-4 >= 0) and (j-4 >= 0)):
                    count = chessboard[i,j] \
                        + chessboard[i-1,j-1] \
                        + chessboard[i-2,j-2] \
                        + chessboard[i-3,j-3] \
                        + chessboard[i-4,j-4]
                    if count == 5*chess_value:
                        flag =  True
                        return flag

                if (j-4 >= 0):
                    count = chessboard[i,j] \
                        + chessboard[i,j-1] \
                        + chessboard[i,j-2] \
                        + chessboard[i,j-3] \
                        + chessboard[i,j-4]
                    if count == 5*chess_value:
                        flag =  True
                        return flag

                if ((i+4 < chessboardwidth) and (j-4 >= 0)):
                    count = chessboard[i,j] \
                        + chessboard[i+1,j-1] \
                        + chessboard[i+2,j-2] \
                        + chessboard[i+3,j-3] \
                        + chessboard[i+4,j-4]
                    if count == 5*chess_value:
                        flag =  True
                        return flag

                if (i+4 < chessboardwidth):
                    count = chessboard[i,j] \
                        + chessboard[i+1,j] \
                        + chessboard[i+2,j] \
                        + chessboard[i+3,j] \
                        + chessboard[i+4,j]
                    if count == 5*chess_value:
                        flag =  True
                        return flag
                    
                if ((i+4 < chessboardwidth) and (j+4 < chessboardheight)):
                    count = chessboard[i,j] \
                        + chessboard[i+1,j+1] \
                        + chessboard[i+2,j+2] \
                        + chessboard[i+3,j+3] \
                        + chessboard[i+4,j+4]
                    if count == 5*chess_value:
                        flag =  True
                        return flag

                if (j+4 < chessboardheight):
                    count = chessboard[i,j] \
                        + chessboard[i,j+1] \
                        + chessboard[i,j+2] \
                        + chessboard[i,j+3] \
                        + chessboard[i,j+4]
                    if count == 5*chess_value:
                        flag =  True
                        return flag
                    
                if ((i-4 >= 0) and (j+4 < chessboardheight)):
                    count = chessboard[i,j] \
                        + chessboard[i-1,j+1] \
                        + chessboard[i-2,j+2] \
                        + chessboard[i-3,j+3] \
                        + chessboard[i-4,j+4]
                    if count == 5*chess_value:
                        flag =  True
                        return flag

                if (i-4 >= 0):
                    count = chessboard[i,j] \
                        + chessboard[i-1,j] \
                        + chessboard[i-2,j] \
                        + chessboard[i-3,j] \
                        + chessboard[i-4,j]
                    if count == 5*chess_value:
                        flag =  True
                        return flag

        return flag

# class GameEngine:
#     def __init__(self,playerA,playerB,verbose=False):
#         self.playerA = playerA
#         self.playerB = playerB
#         self.chessboard = Chessboard.Chessboard(players=[playerA,playerB])
#         self.verbose = verbose

#     def init(self):
#         self.chessboard.init()

#     def run(self,op='A'):
#         # Offensive Position, default: Player A
#         if op == 'A':
#             players = [self.playerA,self.playerB]
#         else:
#             players = [self.playerB,self.playerA]

#         status,winner = self.chessboard.get_status()
#         # The game will continue to battle it out
#         while not status:
#             for i in range(len(players)):
#                 chesspos = players[i].play(self.chessboard)
#                 while not self.chessboard.playchess(chesspos,role=players[i].role):
#                     chesspos = players[i].play(self.chessboard)
#                 # Save chessboard info
#                 status,winner = self.chessboard.victoryjudge(role=players[i].role)
#                 if status:
#                     break
#                 if self.verbose:
#                     print("{step}th Self-play step ...".format(step=self.chessboard.get_stepinfo()))
#         return self.chessboard.get_game_data()

# class Gamedata:
#     def __init__(self,board_shape,data=None):
#         self.date =  time.strftime("%Y-%m-%d(%H:%M:%S)", time.localtime())

#         self.list = data
#         if data is None:
#             self.list = list()
#         self.totalsteps = 0
#         self.board_shape = board_shape
#         self.winner = None

#     def init(self,board_shape,data=None):
#         self.__init__(board_shape=board_shape,data=data)

#     def append(self,chessboardinfo):
#         self.list.append(copy.deepcopy(chessboardinfo))
#         self.totalsteps += 1

#     def gameend(self,winner):
#         self.winner = winner

#     def getinfo(self):
#         return self.winner, self.totalsteps

#     def getstepinfo(self):
#         return self.totalsteps
        
#     def getdatas(self,indexs):
#         data = list()
#         for i in indexs:
#             if (i<0) or (i>=self.totalsteps):
#                 data.append(np.zeros(self.board_shape))
#             else:
#                 data.append(self.list[i])
#         #if (min(indexs) < 0) and (max(index) >= self.totalsteps):
#         #    return None
#         #data = [self.list[i] for i in indexs]
#         return data

#     def getdata(self,index):
#         if (index<0) or (index>=self.totalsteps):
#             return None
#         else:
#             return copy.deepcopy(self.list[index])

#     def getdatashape(self):
#         return self.board_shape

#     def getalldata(self):
#         return self.winner, copy.deepcopy(self.list)

#     def getstate(self,steps=3):
#         '''
#         Get state vector (for neural network)
#         '''
#         state = list()
#         if self.totalsteps == 0:
#             for _ in range(steps):
#                 state.append(np.zeros(self.board_shape))
#             return state
#         if self.totalsteps < steps:
#             for _ in range(steps-self.totalsteps):
#                 state.append(np.zeros(self.board_shape))
#             for i in range(self.totalsteps):
#                 state.append(self.list[i])
#             return state
#         for i in range(self.totalsteps-steps,self.totalsteps):
#             state.append(self.list[i])
#         return state

#     def getwinner(self):
#         return self.winner

#     def load(self,filename):
#         gamefile = h5py.File(filename,"r")
#         self.date = gamefile["date"][...]
#         self.playerA = gamefile["playerA"][...]
#         self.playerB = gamefile["playerB"][...]
#         self.winner = gamefile["winner"][...]
#         self.totalsteps = gamefile["totalsteps"][...]
#         self.list = list()
#         for i in range(self.totalsteps):
#             chessboard = gamefile["chessboards"]["step_{i}".format(i=i)][...]
#             self.list.append(chessboard)
#         gamefile.close()

#     def save(self,filename=None,description=""):
#         '''
#         Save all useful information into HDF5-form file:
#             + Date
#             + Player info
#             + Winner info
#             + Total play steps
#             + Chessboard info for each step
#             + Descriptions/Remarks
#         '''
#         if filename is None:
#             filename = self.date

#         gamefile = h5py.File(filename+".hdf5", "w")
#         gamefile.create_dataset("date", data=self.date)
#         gamefile.create_dataset("playerA",data=self.playerB)
#         gamefile.create_dataset("playerB",data=self.playerB)
#         gamefile.create_dataset("winner",data=self.winner)
#         gamefile.create_dataset("totalsteps",data=self.totalsteps)
#         chessboards = gamefile.create_group("chessboards")
#         for i in range(self.list):
#             chessboards["step_{i}".format(i=i)] = self.list[i]

#         gamefile.create_dataset("description",data=description)
#         gamefile.close()