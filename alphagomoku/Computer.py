import numpy as np 
import deepai

class Computer:
    def __init__(self):
        self.AI = self.deepai.AI()

    def play(self,chessboardinfo):
        chesspos = self.AI.play(chessboardinfo)
        return chesspos
