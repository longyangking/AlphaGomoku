import sys
sys.path.append("..")

import Config
from ai import AI
from selfplay import Selfplay
from viewer import Dataviewer

if __name__== "__main__":
    input_size = Config.ChessBoardHeight*Config.ChessBoardWidth
    output_size = (1,input_size)
    hidden_layers = [(100,100),(100,50)]
    ai = AI(input_size, output_size, hidden_layers)
    c_puct, n_playout, is_selfplay = 1, 10, 1
    selfplay = Selfplay(ai=ai, c_puct=c_puct, n_playout=n_playout, is_selfplay=is_selfplay, verbose=True)
    selfplay.init()

    # gamedata = selfplay.get_data()
    # winner, steps = gamedata.getinfo()
    # print(winner)
    # print(steps)

    gamedata = selfplay.get_data()
    
    dataviewer = Dataviewer(gamedata)
    dataviewer.start()
    