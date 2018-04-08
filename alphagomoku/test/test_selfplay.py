import sys
sys.path.append("..")

from deepai.test import TestAI
from selfplay import Selfplay

if __name__== "__main__":
    ai = TestAI()
    c_puct, n_playout, is_selfplay = 1, 10, 1
    selfplay = Selfplay(ai=ai, c_puct=c_puct, n_playout=n_playout, is_selfplay=is_selfplay, verbose=False)
    selfplay.init()