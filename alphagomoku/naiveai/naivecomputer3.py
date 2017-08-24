import numpy as np 
import sys
sys.path.append("..")
import Config

HV = Config.ChessInfo['Human']
CV = Config.ChessInfo['Computer']

class Computer:
    '''
    Naive Computer with full cost deep-search strategy
    '''
    def __init__(self):
        self.chessboardinfo = None
        print("Naive Computer with full cost deep-search strategy")
        pass

    def play(self,chessboardinfo):
        self.chessboardinfo = chessboardinfo.copy() # Local version
        point = self.__widesearch()
        return point

    def __widesearch(self,num=4):
        points = self.__generator()
        
        bestvalue = None
        bestpoint = None

        for point in points:
            self.chessboardinfo[point] = CV

            computervalue0 = self.__valueComputer(point)
            humanvalue0 = self.__valueHuman(point)

            computervalue,humanvalue = self.__playbyHuman(num)
            computervalue += computervalue0
            humanvalue += humanvalue0

            # Self-defined principles
            value = computervalue + humanvalue
            if humanvalue > computervalue:
                value = humanvalue
            else:
                value = computervalue
            
            if (bestpoint is None) or (bestvalue < value):
                bestvalue = value
                bestpoint = point

            self.chessboardinfo[point] = 0
        return bestpoint
            
    def __playbyComputer(self,num):
        # Normal process
        values = self.__valuesComputer()
        bestpos = self.__bestposComputer(values)
        computervalue = self.__valueComputer(bestpos) 
        humanvalue = self.__valueHuman(bestpos)
        # End of iteration

        self.chessboardinfo[bestpos] = CV
        if num == 0:
            value = computervalue,humanvalue
        else:
            computervaluenext,humanvaluenext = self.__playbyHuman(num-1)
            value = computervalue + computervaluenext, humanvalue + humanvaluenext
        self.chessboardinfo[bestpos] = 0

        return value

    def __playbyHuman(self,num):
        # Normal process
        values = self.__valuesHuman()
        bestpos = self.__bestposHuman(values)
        computervalue = self.__valueComputer(bestpos) 
        humanvalue = self.__valueHuman(bestpos)
        # End of iteration

        self.chessboardinfo[bestpos] = HV
        if num == 0:
            value = computervalue,humanvalue
        else:
            computervaluenext,humanvaluenext = self.__playbyComputer(num-1)
            value = computervalue + computervaluenext, humanvalue + humanvaluenext
        self.chessboardinfo[bestpos] = 0

        return value

    def __generator(self,num=10):
        (width,height) = self.chessboardinfo.shape

        # The positions which give great value to computer
        values = self.__valuesComputer()
        argpos = np.argsort(-values.flatten())
        choices = argpos[:num]
        #print(choices/width)
        Xs = list(map(int,choices/width))
        Ys = list(choices%height)

        # The positions which give great value to human
        values = self.__valuesHuman()
        argpos = np.argsort(-values.flatten())
        choices = argpos[:num]
        Xs.extend(list(map(int,choices/width)))
        Ys.extend(list(choices%height))

        return list(zip(Xs,Ys))

    def __bestposComputer(self,values):
        return self.__bestposition(values)
    
    def __bestposHuman(self,values):
        return self.__bestposition(values)

    def __bestposition(self,values):
        (width,height) = self.chessboardinfo.shape
        bestpos = np.argmax(values.flatten())
        X = int(bestpos/width)
        Y = bestpos%width
        return (X,Y)

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
            return 1

        # CCCC?
        if (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV) \
           and (self.__getchess([x+3*dx,y+3*dy])==CV) and (self.__getchess([x+4*dx,y+4*dy])==CV) \
           and (self.__getchess([x+5*dx,y+5*dy])==HV):
            return 2
        
        # CCC?C
        if (self.__getchess([x-dx,y-dy])==CV) and (self.__getchess([x+dx,y+dy])==CV) \
            and (self.__getchess([x+2*dx,y+2*dy])==CV) and (self.__getchess([x+3*dx,y+3*dy])==CV):
            return 3

        # CC?CC
        if (self.__getchess([x-2*dx,y-2*dy])==CV) and (self.__getchess([x-dx,y-dy])==CV) \
            and (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV):
            return 4
        
        # ??CCC??
        if (self.__getchess([x-dx,y-dy])==0) and (self.__getchess([x+dx,y+dy])==CV) \
            and (self.__getchess([x+2*dx,y+2*dy])==CV) and (self.__getchess([x+3*dx,y+3*dy])==CV) \
            and (self.__getchess([x+4*dx,y+4*dy])==0) and (self.__getchess([x+5*dx,y+5*dy])==0):
            return 5

        # CCC??
        if (self.__getchess([x-dx,y-dy])==0) and (self.__getchess([x+dx,y+dy])==CV) \
            and (self.__getchess([x+2*dx,y+2*dy])==CV) and (self.__getchess([x+3*dx,y+3*dy])==CV):
            return 6
        
        # ?C?CC?
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==CV) \
            and (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0):
            return 7

        # C??CC
        if (self.__getchess([x-2*dx,y-2*dy])==CV) and (self.__getchess([x-dx,y-dy])==0) \
            and (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV):
            return 8

        # C?C?C
        if (self.__getchess([x-dx,y-dy])==CV) and (self.__getchess([x+dx,y+dy])==CV) \
            and (self.__getchess([x+2*dx,y+2*dy])==0) and (self.__getchess([x+3*dx,y+3*dy])==CV):
            return 9

        # ???CC???
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==0) \
            and (self.__getchess([x+dx,y+dy])==CV) and (self.__getchess([x+2*dx,y+2*dy])==CV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0) and (self.__getchess([x+4*dx,y+4*dy])==0) \
            and (self.__getchess([x+5*dx,y+5*dy])==0):
            return 10
        
        # CC???
        if (self.__getchess([x-2*dx,y-2*dy])==CV) and (self.__getchess([x-dx,y-dy])==CV) \
            and (self.__getchess([x+dx,y+dy])==0) and (self.__getchess([x+2*dx,y+2*dy])==0):
            return 11

        # ??C?C??
        if (self.__getchess([x-3*dx,y-3*dy])==0) and (self.__getchess([x-2*dx,y-2*dy])==0) \
            and (self.__getchess([x-dx,y-dy])==CV) and (self.__getchess([x+dx,y+dy])==CV) \
            and (self.__getchess([x+2*dx,y+2*dy])==0) and (self.__getchess([x+3*dx,y+3*dy])==0):
            return 12
        
        # ?C??C?
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==CV) \
            and (self.__getchess([x+dx,y+dy])==0) and (self.__getchess([x+2*dx,y+2*dy])==CV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0):
            return 13

        # The rules to destroy rivals

         # ?HHHH?
        if (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV) \
            and (self.__getchess([x+3*dx,y+3*dy])==HV) and (self.__getchess([x+4*dx,y+4*dy])==HV) \
            and (self.__getchess([x+5*dx,y+5*dy])==0):
            return -1

        # HHHH?
        if (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV) \
           and (self.__getchess([x+3*dx,y+3*dy])==HV) and (self.__getchess([x+4*dx,y+4*dy])==HV) :
            return -2
        
        # HHH?H
        if (self.__getchess([x-dx,y-dy])==HV) and (self.__getchess([x+dx,y+dy])==HV) \
            and (self.__getchess([x+2*dx,y+2*dy])==HV) and (self.__getchess([x+3*dx,y+3*dy])==HV):
            return -3

        # HH?HH
        if (self.__getchess([x-2*dx,y-2*dy])==HV) and (self.__getchess([x-dx,y-dy])==HV) \
            and (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV):
            return -4
        
        # ??HHH??
        if (self.__getchess([x-dx,y-dy])==0) and (self.__getchess([x+dx,y+dy])==HV) \
            and (self.__getchess([x+2*dx,y+2*dy])==HV) and (self.__getchess([x+3*dx,y+3*dy])==HV) \
            and (self.__getchess([x+4*dx,y+4*dy])==0) and (self.__getchess([x+5*dx,y+5*dy])==0):
            return -5

        # HHH??
        if (self.__getchess([x-dx,y-dy])==0) and (self.__getchess([x+dx,y+dy])==HV) \
            and (self.__getchess([x+2*dx,y+2*dy])==HV) and (self.__getchess([x+3*dx,y+3*dy])==HV):
            return -6
        
        # ?H?HH?
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==HV) \
            and (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0):
            return -7

        # H??HH
        if (self.__getchess([x-2*dx,y-2*dy])==HV) and (self.__getchess([x-dx,y-dy])==0) \
            and (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV):
            return -8

        # H?H?H
        if (self.__getchess([x-dx,y-dy])==HV) and (self.__getchess([x+dx,y+dy])==HV) \
            and (self.__getchess([x+2*dx,y+2*dy])==0) and (self.__getchess([x+3*dx,y+3*dy])==HV):
            return -9

        # ???HH???
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==0) \
            and (self.__getchess([x+dx,y+dy])==HV) and (self.__getchess([x+2*dx,y+2*dy])==HV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0) and (self.__getchess([x+4*dx,y+4*dy])==0) \
            and (self.__getchess([x+5*dx,y+5*dy])==0):
            return -10
        
        # HH???
        if (self.__getchess([x-2*dx,y-2*dy])==HV) and (self.__getchess([x-dx,y-dy])==HV) \
            and (self.__getchess([x+dx,y+dy])==0) and (self.__getchess([x+2*dx,y+2*dy])==0):
            return -11

        # ??H?H??
        if (self.__getchess([x-3*dx,y-3*dy])==0) and (self.__getchess([x-2*dx,y-2*dy])==0) \
            and (self.__getchess([x-dx,y-dy])==HV) and (self.__getchess([x+dx,y+dy])==HV) \
            and (self.__getchess([x+2*dx,y+2*dy])==0) and (self.__getchess([x+3*dx,y+3*dy])==0):
            return -12
        
        # ?H??H?
        if (self.__getchess([x-2*dx,y-2*dy])==0) and (self.__getchess([x-dx,y-dy])==HV) \
            and (self.__getchess([x+dx,y+dy])==0) and (self.__getchess([x+2*dx,y+2*dy])==HV) \
            and (self.__getchess([x+3*dx,y+3*dy])==0):
            return -13

        return 0
        
    def __valueComputer(self,pos):
        (width,height) = self.chessboardinfo.shape
        i,j = pos
        if self.chessboardinfo[i,j] !=0: return -1

        value = 0
        for direction in range(8):
            patternnum = self.__patternrecognizer([i,j],direction)                    
            # Value Table
            if patternnum == 1: value += 300000
            if patternnum == 2: value += 2500
            if patternnum == 3: value += 3000
            if patternnum == 4: value += 2600
            if patternnum == 5: value += 3000
            if patternnum == 6: value += 500
            if patternnum == 7: value += 800
            if patternnum == 8: value += 600
            if patternnum == 9: value += 550
            if patternnum == 10: value += 650
            if patternnum == 11:value += 150
            if patternnum == 12:value += 250
            if patternnum == 13:value += 200

            # Make sure that the initial position has biggest probability around the center
            if value == 0: value = 0.1*np.exp(-((i-width/2)**2/(width/2)**2 + (j-height/2)**2/(height/2)**2))
                    
        return value

    def __valueHuman(self,pos):
        (width,height) = self.chessboardinfo.shape
        i,j = pos
        if self.chessboardinfo[i,j] !=0: return -1

        value = 0
        for direction in range(8):
            patternnum = self.__patternrecognizer([i,j],direction)                    
            # Value Table
            if patternnum == -1: value += 300000
            if patternnum == -2: value += 2500
            if patternnum == -3: value += 3000
            if patternnum == -4: value += 2600
            if patternnum == -5: value += 3000
            if patternnum == -6: value += 500
            if patternnum == -7: value += 800
            if patternnum == -8: value += 600
            if patternnum == -9: value += 550
            if patternnum == -10: value += 650
            if patternnum == -11:value += 150
            if patternnum == -12:value += 250
            if patternnum == -13:value += 200

            # Make sure that the initial position has biggest probability around the center
            if value == 0: value = 0.1*np.exp(-((i-width/2)**2/(width/2)**2 + (j-height/2)**2/(height/2)**2))
                    
        return value

    def __valuesComputer(self):
        (width,height) = self.chessboardinfo.shape
        values = np.zeros(self.chessboardinfo.shape)
        for i in range(width):
            for j in range(height):
                values[i,j] = self.__valueComputer([i,j])
        return values

    def __valuesHuman(self):
        (width,height) = self.chessboardinfo.shape
        values = np.zeros(self.chessboardinfo.shape)
        for i in range(width):
            for j in range(height):
                values[i,j] = self.__valueHuman([i,j])
        return values