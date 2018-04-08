import sys
sys.path.append("..")

from deepai.test import TestAI
from selfplay import Selfplay
from viewer import Dataviewer

if __name__== "__main__":
    ai = TestAI()
    c_puct, n_playout, is_selfplay = 1, 10, 1
    selfplay = Selfplay(ai=ai, c_puct=c_puct, n_playout=n_playout, is_selfplay=is_selfplay, verbose=True)
    selfplay.init()

    gamedata = selfplay.get_data()
    winner, steps = gamedata.getinfo()
    print(winner)
    print(steps)
    dataviewer = Dataviewer(gamedata)
    dataviewer.start()
    