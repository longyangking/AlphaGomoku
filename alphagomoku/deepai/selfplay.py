#
# Utils for self-play
#

import numpy as np 
from mcts import MCTS 

class Selfplay:
    def __init__(self,ai,verbose=False):
        self.ai = ai
        self.dataset = list()
        
        computerA = MCTS()
        computerB = MCTS()
        self.gameenigne = GameEnigne(playA=computerA,playB=computerB)

    def init(self):
        self.dataset = list()

    def get_data(self):
        gamedata = self.gameenigne.run()
        winner, train_data = gamedata.getdata()
        return winner, train_data

class Evaluator:
    def __init__(self,ai,verbose=False):
        self.ai = ai
        self.evaluate_player = 

    def evaluate(self):



        return win_ratio