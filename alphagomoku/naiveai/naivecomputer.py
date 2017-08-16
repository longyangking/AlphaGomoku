import numpy as np 

class Computer:
    '''
    Naive Computer with minimal cost simplified table-form strategy
    '''
    def __init__(self):
        print("Naive Computer with minimal simplified cost")
        pass

    def play(self,chessboardinfo):
        (width,height) = chessboardinfo.shape
        values = self.__values(chessboardinfo)
        flatpos = np.argmax(values)
        return (int(flatpos/width),flatpos%width)      
    
    def __getchess(self,pos,chessboardinfo):
        x,y = pos
        (width,height) = chessboardinfo.shape
        if (x < 0) or (x >= width) or (y < 0) or (y >= height):
            return -1
        else:
            return chessboardinfo[x,y]

    def __values(self,chessboardinfo):
        (width,height) = chessboardinfo.shape
        values = np.zeros(chessboardinfo.shape)
        for x in range(width):
            for y in range(height):
                if chessboardinfo[x,y] !=0:
                    values[x,y] = -1
                else:
                    value = 0
                    for bias in range(-4,0):
                        valuevertical = 0
                        valuehorizon = 0
                        valuecross = 0
                        for i in range(5):
                            pos = [x + bias + i, y]
                            valuevertical += self.__getchess(pos,chessboardinfo)
                            pos = [x, y + bias +i]
                            valuehorizon += self.__getchess(pos,chessboardinfo)
                            pos = [x + bias + i, y + bias + i]
                            valuecross += self.__getchess(pos,chessboardinfo)
                        value = np.max([value, valuevertical, valuehorizon, valuecross])

                    values[x,y] = value + 0.1*np.exp(-((x-width/2)**2/(width/2)**2 + (y-height/2)**2/(height/2)**2))
                    
        return values        