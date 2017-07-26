import numpy as np 
import ai

class Computer:
    def __init__(self):
        self.AI = self.ai.AI()

    def play(self,chessboardinfo):
        chesspos = self.AI.play(chessboardinfo)
        return chesspos