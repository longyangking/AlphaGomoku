import numpy as np 

import sys
sys.path.append("..")
import Config

HV = Config.ChessInfo['Human']
CV = Config.ChessInfo['Computer']

unCover4 = [
    [0,CV,CV,CV,CV,0]
]

unCover3 = [
    [0,CV,CV,CV,0,0],
    [0,0,CV,CV,CV,0],
    [0,CV,0,CV,CV,0],
    [0,CV,CV,0,CV,0]
]

unCover2 = [
    [0,0,CV,CV,0,0],
    [0,CV,0,CV,0,0],
    [0,0,CV,0,CV,0],
    [0,CV,CV,0,0,0],
    [0,0,0,CV,CV,0],
    [0,CV,0,0,CV,0]
]

unCover1 = [
    [0,CV,0,0,0,0],
    [0,0,CV,0,0,0],
    [0,0,0,CV,0,0],
    [0,0,0,0,CV,0]
]

Cover4 = [
    [HV,CV,0,CV,CV,CV],
    [HV,CV,CV,0,CV,CV],
    [HV,CV,CV,CV,0,CV],
    [HV,CV,CV,CV,CV,0],
    [0,CV,CV,CV,CV,HV],
    [CV,0,CV,CV,CV,HV],
    [CV,CV,0,CV,CV,HV],
    [CV,CV,CV,0,CV,HV]
]

Cover3 = [
    [HV,CV,CV,CV,0,0],
    [HV,CV,CV,0,CV,0],
    [HV,CV,0,CV,CV,0],
    [0,0,CV,CV,CV,HV],
    [0,CV,0,CV,CV,HV],
    [0,CV,CV,0,CV,HV],
    [HV,CV,0,CV,0,CV,HV],
    [HV,0,CV,CV,CV,0,HV],
    [HV,CV,CV,0,0,CV,HV],
    [HV,CV,0,0,CV,CV,HV]
]

Cover2 = [
    [HV,CV,CV,0,0,0],
    [HV,CV,0,CV,0,0],
    [HV,CV,0,0,CV,0],
    [HV,0,CV,CV,0,0],
    [HV,0,CV,0,CV,0],
    [HV,0,0,CV,CV,0],
    [0,CV,CV,0,0,HV],
    [0,CV,0,CV,0,HV],
    [0,CV,0,0,CV,HV],
    [0,0,CV,CV,0,HV],
    [0,0,CV,0,CV,HV],
    [0,0,0,CV,CV,HV],
    [HV,CV,CV,0,0,0,HV],
    [HV,CV,0,CV,0,0,HV],
    [HV,CV,0,0,CV,0,HV],
    [HV,CV,0,0,0,CV,HV],
    [HV,0,CV,CV,0,0,HV],
    [HV,0,CV,0,CV,0,HV],
    [HV,0,CV,0,0,CV,HV],
    [HV,0,0,CV,CV,0,HV],
    [HV,0,0,CV,0,CV,HV],
    [HV,0,0,0,CV,CV,HV]
]

class Computer:
    '''
    Naive Computer with minimal cost full table-form strategy
    '''
    def __init__(self):
        self.chessboardinfo = None
        pass

    def play(self,chessboardinfo):
        self.chessboardinfo = chessboardinfo
        (width,height) = chessboardinfo.shape
        values = self.__values()
        flatpos = np.argmax(values)
        return (int(flatpos/width),flatpos%width)

    def __getchess(self,pos):
        x,y = pos
        (width,height) = self.chessboardinfo.shape
        if (x < 0) or (x >= width) or (y < 0) or (y >= height):
            return HV
        else:
            return self.chessboardinfo[x,y]

    def __patternrecognizer(self,pos):
        chessboardinfo = self.chessboardinfo
        (width,height) = chessboardinfo.shape
        x,y = pos

        value = HV
        for bias in range(-4,0):
            valuevertical = 0
            valuehorizon = 0
            valuecross = 0
            for i in range(5):
                pos = [x + bias + i, y]
                valuevertical += self.__getchess(pos)
                pos = [x, y + bias +i]
                valuehorizon += self.__getchess(pos)
                pos = [x + bias + i, y + bias + i]
                valuecross += self.__getchess(pos)
            value = np.max([value, valuevertical, valuehorizon, valuecross])
        return value
        
    def __values(self):
        (width,height) = self.chessboardinfo.shape
        values = np.zeros(self.chessboardinfo.shape)
        for i in range(width):
            for j in range(height):
                if self.chessboardinfo[i,j] !=0:
                    values[i,j] = -1
                else:
                    values[i,j] = self.__patternrecognizer([i,j])
                    
        return values        