import numpy as np 

class Computer:
    '''
    Naive Computer with Monte-Carlo Method
    '''
    def __init__(self):
        print("Naive Computer with Monte-Carlo Method")
        pass

    def play(self,chessboardinfo):
        (width,height) = chessboardinfo.shape
        values = self.__values(chessboardinfo)
        flatpos = np.argmax(values)
        return (int(flatpos/width),flatpos%width)

    def __montecarlo(self,chessboardinfo,num=100):
        # Pick the position with highest value from the random samplings
        (width,height) = chessboardinfo.shape
        for i in range(num):
            x = np.random.randint(width)
            y = np.random.randint(height)

            # Calculate the value of position (x,y)
        

    def __values(self,chessboardinfo):
        pass     