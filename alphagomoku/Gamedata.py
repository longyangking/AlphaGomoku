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