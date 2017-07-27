import numpy as np 

class Computer:
    def __init__(self):
        pass

    def play(self,chessboardinfo):
        (width,height) = chessboardinfo.shape
        values = self.__values(chessboardinfo)
        flatpos = np.argmax(values)
        return (int(flatpos/width),flatpos%width)

    def __values(self,chessboardinfo):
        (width,height) = chessboardinfo.shape
        values = np.zeros(chessboardinfo.shape)
        for i in range(width):
            for j in range(height):
                value = 0
                if ((i-4 >= 0) and (j-4 >= 0)):
                    value += 4*chessboardinfo[i-1,j-1] \
                        + 3*chessboardinfo[i-2,j-2] \
                        + 2*chessboardinfo[i-3,j-3] \
                        + chessboardinfo[i-4,j-4]
                if (j-4 >= 0):
                    value += 4*chessboardinfo[i,j-1] \
                        + 3*chessboardinfo[i,j-2] \
                        + 2*chessboardinfo[i,j-3] \
                        + chessboardinfo[i,j-4]
                if ((i+4 < width) and (j-4 >= 0)):
                    value += 4*chessboardinfo[i+1,j-1] \
                        + 3*chessboardinfo[i+2,j-2] \
                        + 2*chessboardinfo[i+3,j-3] \
                        + chessboardinfo[i+4,j-4]
                if (i+4 < width):
                    value += 4*chessboardinfo[i+1,j] \
                        + 3*chessboardinfo[i+2,j] \
                        + 2*chessboardinfo[i+3,j] \
                        + chessboardinfo[i+4,j]
                if ((i+4 < width) and (j+4 <height)):
                    value += 4*chessboardinfo[i+1,j+1] \
                        + 3*chessboardinfo[i+2,j+2] \
                        + 2*chessboardinfo[i+3,j+3] \
                        + chessboardinfo[i+4,j+4]
                if (j+4 < height):
                    value += 4*chessboardinfo[i,j+1] \
                        + 3*chessboardinfo[i,j+2] \
                        + 2*chessboardinfo[i,j+3] \
                        + chessboardinfo[i,j+4]
                if ((i-4 >= 0) and (j+4 < height)):
                    value += 4*chessboardinfo[i-1,j+1] \
                        + 3*chessboardinfo[i-2,j+2] \
                        + 2*chessboardinfo[i-3,j+3] \
                        + chessboardinfo[i-4,j+4]
                if (i-4 >= 0):
                    value += 4*chessboardinfo[i-1,j] \
                        + 3*chessboardinfo[i-2,j] \
                        + 2*chessboardinfo[i-3,j] \
                        + chessboardinfo[i-4,j]
                values[i,j] = value
        return values        