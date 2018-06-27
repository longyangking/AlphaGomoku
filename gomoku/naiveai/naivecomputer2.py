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
        print('Naive Computer with minimal cost full table-form strategy')
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
            return -2
        else:
            return self.chessboardinfo[x,y]

    def __patternrecognizer(self,pos,direction):
        x,y = pos
        dx,dy = 1,0
        # Pattern Table
        if direction == 0: dx,dy = 1,0
        if direction == 1: dx,dy = 1,1
        if direction == 2: dx,dy = 0,1
        if direction == 3: dx,dy = -1,1
        if direction == 4: dx,dy = -1,0
        if direction == 5: dx,dy = -1,-1
        if direction == 6: dx,dy = 0,-1
        if direction == 7: dx,dy = 1,-1

        # The rules to win

        # ?CCCC?
        if (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV) \
            and (self.__getchess([x+3*dx,y+3*dy])==CV) and (self.__getchess([x+4*dx,y+4*dy])==CV) \
            and (self.__getchess([x+5*dx,y+5*dy])==0):
            return 0

        # CCCC?
        if (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV) \
           and (self.__getchess([x+3*dx,y+3*dy])==CV) and (self.__getchess([x+4*dx,y+4*dy])==CV) \
           and (self.__getchess([x+5*dx,y+5*dy])==HV):
            return 1
        
        # CCC?C
        if (self.__getchess([x-dx,y-dy])==CV) and (self.__getchess([x+dx,y+dy])==CV) \
            and (self.__getchess([x+2*dx,y+2*dy])==CV) and (self.__getchess([x+3*dx,y+3*dy])==CV):
            return 2

        # CC?CC
        if (self.__getchess([x-2*dx,y-2*dy])==CV) and (self.__getchess([x-dx,y-dy])==CV) \
            and (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV):
            return 3
        
        # ??CCC??
        if (self.__getchess([x-dx,y-dy])==0) and (self.__getchess([x+dx,y+dy])==CV) \
            and (self.__getchess([x+2*dx,y+2*dy])==CV) and (self.__getchess([x+3*dx,y+3*dy])==CV) \
            and (self.__getchess([x+4*dx,y+4*dy])==0) and (self.__getchess([x+5*dx,y+5*dy])==0):
            return 4

        # CCC??
        if (self.__getchess([x-dx,y-dy])==0) and (self.__getchess([x+dx,y+dy])==CV) \
            and (self.__getchess([x+2*dx,y+2*dy])==CV) and (self.__getchess([x+3*dx,y+3*dy])==CV):
            return 5
        
        # ?C?CC?
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==CV) \
            and (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0):
            return 6

        # C??CC
        if (self.__getchess([x-2*dx,y-2*dy])==CV) and (self.__getchess([x-dx,y-dy])==0) \
            and (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV):
            return 7

        # C?C?C
        if (self.__getchess([x-dx,y-dy])==CV) and (self.__getchess([x+dx,y+dy])==CV) \
            and (self.__getchess([x+2*dx,y+2*dy])==0) and (self.__getchess([x+3*dx,y+3*dy])==CV):
            return 8

        # ???CC???
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==0) \
            and (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0) and (self.__getchess([x+4*dx,y+4*dy])==0) \
            and (self.__getchess([x+5*dx,y+5*dy])==0):
            return 9
        
        # CC???
        if (self.__getchess([x-2*dx,y-2*dy])==CV) and (self.__getchess([x-dx,y-dy])==CV) \
            and (self.__getchess([x+dx,y+dy])==0) and (self.__getchess([x+2*dx,y+2*dy])==0):
            return 10

        # ??C?C??
        if (self.__getchess([x-3*dx,y-3*dy])==0) and (self.__getchess([x-2*dx,y-2*dy])==0) \
            and (self.__getchess([x-dx,y-dy])==CV) and (self.__getchess([x+dx,y+dy])==CV) \
            and (self.__getchess([x+2*dx,y+2*dy])==0) and (self.__getchess([x+3*dx,y+3*dy])==0):
            return 11
        
        # ?C??C?
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==CV) \
            and (self.__getchess([x+dx,y+dy])==0) and (self.__getchess([x+2*dx,y+2*dy])==CV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0):
            return 12

        # The rules to destroy rivals

         # ?HHHH?
        if (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV) \
            and (self.__getchess([x+3*dx,y+3*dy])==HV) and (self.__getchess([x+4*dx,y+4*dy])==HV) \
            and (self.__getchess([x+5*dx,y+5*dy])==0):
            return 0

        # HHHH?
        if (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV) \
           and (self.__getchess([x+3*dx,y+3*dy])==HV) and (self.__getchess([x+4*dx,y+4*dy])==HV) :
            return 1
        
        # HHH?H
        if (self.__getchess([x-dx,y-dy])==HV) and (self.__getchess([x+dx,y+dy])==HV) \
            and (self.__getchess([x+2*dx,y+2*dy])==HV) and (self.__getchess([x+3*dx,y+3*dy])==HV):
            return 2

        # HH?HH
        if (self.__getchess([x-2*dx,y-2*dy])==HV) and (self.__getchess([x-dx,y-dy])==HV) \
            and (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV):
            return 3
        
        # ??HHH??
        if (self.__getchess([x-dx,y-dy])==0) and (self.__getchess([x+dx,y+dy])==HV) \
            and (self.__getchess([x+2*dx,y+2*dy])==HV) and (self.__getchess([x+3*dx,y+3*dy])==HV) \
            and (self.__getchess([x+4*dx,y+4*dy])==0) and (self.__getchess([x+5*dx,y+5*dy])==0):
            return 4

        # HHH??
        if (self.__getchess([x-dx,y-dy])==0) and (self.__getchess([x+dx,y+dy])==HV) \
            and (self.__getchess([x+2*dx,y+2*dy])==HV) and (self.__getchess([x+3*dx,y+3*dy])==HV):
            return 5
        
        # ?H?HH?
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==HV) \
            and (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0):
            return 6

        # H??HH
        if (self.__getchess([x-2*dx,y-2*dy])==HV) and (self.__getchess([x-dx,y-dy])==0) \
            and (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV):
            return 7

        # H?H?H
        if (self.__getchess([x-dx,y-dy])==HV) and (self.__getchess([x+dx,y+dy])==HV) \
            and (self.__getchess([x+2*dx,y+2*dy])==0) and (self.__getchess([x+3*dx,y+3*dy])==HV):
            return 8

        # ???HH???
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==0) \
            and (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0) and (self.__getchess([x+4*dx,y+4*dy])==0) \
            and (self.__getchess([x+5*dx,y+5*dy])==0):
            return 9
        
        # HH???
        if (self.__getchess([x-2*dx,y-2*dy])==HV) and (self.__getchess([x-dx,y-dy])==HV) \
            and (self.__getchess([x+dx,y+dy])==0) and (self.__getchess([x+2*dx,y+2*dy])==0):
            return 10

        # ??H?H??
        if (self.__getchess([x-3*dx,y-3*dy])==0) and (self.__getchess([x-2*dx,y-2*dy])==0) \
            and (self.__getchess([x-dx,y-dy])==HV) and (self.__getchess([x+dx,y+dy])==HV) \
            and (self.__getchess([x+2*dx,y+2*dy])==0) and (self.__getchess([x+3*dx,y+3*dy])==0):
            return 11
        
        # ?H??H?
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==HV) \
            and (self.__getchess([x+dx,y+dy])==0) and (self.__getchess([x+2*dx,y+2*dy])==HV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0):
            return 12

        return -1
        
    def __values(self):
        (width,height) = self.chessboardinfo.shape
        values = np.zeros(self.chessboardinfo.shape)
        for i in range(width):
            for j in range(height):
                if self.chessboardinfo[i,j] !=0:
                    values[i,j] = -1
                else:
                    value = 0
                    for direction in range(8):
                        patternnum = self.__patternrecognizer([i,j],direction)                    
                        # Value Table
                        if patternnum == 0:
                            value += 300000
                        if patternnum == 1:
                            value += 2500
                        if patternnum == 2:
                            value += 3000
                        if patternnum == 3:
                            value += 2600
                        if patternnum == 4:
                            value += 3000
                        if patternnum == 5:
                            value += 500
                        if patternnum == 6:
                            value += 800
                        if patternnum == 7:
                            value += 600
                        if patternnum == 8:
                            value += 550
                        if patternnum == 9:
                            value += 650
                        if patternnum == 10:
                            value += 150
                        if patternnum == 11:
                            value += 250
                        if patternnum == 12:
                            value += 200

                    if value == 0:
                        value = 0.1*np.exp(-((i-width/2)**2/(width/2)**2 + (j-height/2)**2/(height/2)**2))
                    values[i,j] = value
                    
        return values        