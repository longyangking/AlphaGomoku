#
# Utils for self-play
#
from __future__ import absolute_import

import numpy as np 
from mcts import MCTSPlayer
import Config
import Computer
from Gameengine import GameEngine

class Selfplay:
    def __init__(self, ai, c_puct, n_playout, is_selfplay, verbose=False):
        self.ai = ai
        self.dataset = list()
        self.gameenigne = None
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.is_selfplay = is_selfplay
        self.verbose = verbose

    def init(self):
        self.dataset = list()
        computerA = MCTSPlayer(value_function=self.ai.value_function, 
                        c_puct=self.c_puct, 
                        n_playout=self.n_playout, 
                        is_selfplay=self.is_selfplay,
                        role='Self_A',
                        verbose=self.verbose)
        computerB = MCTSPlayer(value_function=self.ai.value_function, 
                        c_puct=self.c_puct, 
                        n_playout=self.n_playout, 
                        is_selfplay=self.is_selfplay,
                        role='Self_B',
                        verbose=self.verbose)
        self.gameenigne = GameEngine(playerA=computerA,playerB=computerB)

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
