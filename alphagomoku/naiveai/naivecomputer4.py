import numpy as np 

class Computer:
    '''
    Naive Computer with minimal cost deep-search strategy
    '''
    def __init__(self):
        print("Naive Computer with minimal cost deep-search strategy")
        pass

    def play(self,chessboardinfo):
        (width,height) = chessboardinfo.shape
        values = self.__values(chessboardinfo)
        flatpos = np.argmax(values)
        return (int(flatpos/width),flatpos%width)

    def __deepsearch(self,chesses,num=5):
        # Case: oooo?
        if np.sum(chesses) == 4:
            return 300000
        

    def __values(self,chessboardinfo):
        (width,height) = chessboardinfo.shape
        values = np.zeros(chessboardinfo.shape)
        for i in range(width):
            for j in range(height):
                if chessboardinfo[i,j] !=0:
                    values[i,j] = -1
                else:
                    
        return values        