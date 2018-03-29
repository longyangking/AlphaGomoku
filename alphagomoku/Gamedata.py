import h5py
import numpy as np 
import time

class Gamedata:
    def __init__(self,data=None):
        self.date =  time.strftime("%Y-%m-%d(%H:%M:%S)", time.localtime())

        self.list = data
        if data is None:
            self.list = list()
        self.totalsteps = 0
        self.winner = None

    def init(self,data=None):
        self.__init__(data=data)

    def append(self,chessboardinfo):
        self.list.append(chessboardinfo)
        self.totalsteps += 1

    def gameend(self,winner):
        self.winner = winner

    def getinfo(self):
        return self.winner, self.totalsteps
        
    def getdata(self,indexs):
        if (min(indexs) < 0) and (max(index) >= self.totalsteps):
            return None
        data = [self.list[i] for i in indexs]
        return data
        
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