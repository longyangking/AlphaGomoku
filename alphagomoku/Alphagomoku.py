import numpy as np 
import argparse
from train import TrainAI
import Config

board_size = (Config.ChessBoardHeight, Config.ChessBoardWidth)

__version__ = "0.0.1"
__author__ = "Yang Long"
__info__ = "Play Gomoku with AI"

class Alphagomoku:
    def __init__(self, borad_size=board_size):
        pass

    def restart_train(self):
        pass

    def continue_train(self,model):
        pass

    def loadai(self):
        pass

    def run(self):
        pass

if __name__=='__main__':

    parser = argparse.ArgumentParser(description="Play with AI or Train AI")
    #parser.add_argument("--retrain", type=bool, action='store_true', default=False, help="Re-Train AI")
    #parser.add_argument("--train", type=bool, action='store_true', default=False, help="Train AI")
    #parser.add_argument("--verbose", type=bool, action='store_true', default=True, help="Show the process information")

    parser.add_argument("--retrain", action='store_true', default=False, help="Re-Train AI")
    parser.add_argument("--train",  action='store_true', default=False, help="Train AI")
    parser.add_argument("--verbose", action='store_true', default=False, help="Show the process information")
    parser.add_argument("--play", action='store_true', default=False, help="Play with AI")

    args = parser.parse_args()
    if args.train:
        print("Train AI")

    if args.retrain:
        print("Re-train AI")
        trainai = TrainAI(board_size=board_size)

    if args.verbose:
        info = """
            {name}: {version}
            Author: {author}
            Info: {info}
        """.format(
            name='AlphaZero',
            version=__version__,
            author=__author__,
            info=__info__
        )
        print(info)

        help_info = """
            --retrain:  Re-Train AI
            --train:    Train AI
            --verbose:  Show the process information
            --play:     Play with AI
        """
        print("Arguments:")
        print(help_info)

    if args.play:
        print("Play with AI!")