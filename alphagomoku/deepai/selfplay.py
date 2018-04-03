#
# Utils for self-play
#

import numpy as np 
from ..GameEnigne import GameEnigne
from mcts import MCTSPlayer

class Selfplay:
    def __init__(self,ai, c_puct, n_playout, is_selfplay, verbose=False):
        self.ai = ai
        self.dataset = list()
        self.gameenigne = None
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.is_selfplay = is_selfplay
        self.verbose = verbose

    def init(self):
        self.dataset = list()
        computerA = MCTSPlayer(func=ai.value_function, 
                        c_puct=self.c_puct, 
                        n_playout=self.n_playout, 
                        is_selfplay=is_selfplay,
                        role='Self_A',
                        verbose=self.verbose)
        computerB = MCTSPlayer(func=ai.value_function, 
                        c_puct=self.c_puct, 
                        n_playout=self.n_playout, 
                        is_selfplay=is_selfplay,
                        role='Self_B',
                        verbose=self.verbose)
        self.gameenigne = GameEnigne(playA=computerA,playB=computerB)

    def get_data(self):
        gamedata = self.gameenigne.run()
        winner, train_data = gamedata.getdata()
        return winner, train_data

class Evaluator:
    def __init__(self,ai,verbose=False):
        self.ai = ai
        self.evaluate_player = None

    def evaluate(self):
        # TODO play with naive MCTS player to evaluate

        win_ratio = 0

        return win_ratio

if __name__== "__main__":
