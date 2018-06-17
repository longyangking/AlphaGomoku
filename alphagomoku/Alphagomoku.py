import numpy as np 
import argparse
from train import TrainAI
import Config

board_size = (Config.ChessBoardHeight, Config.ChessBoardWidth)

__version__ = "0.0.1"
__author__ = "Yang Long"

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
    info = """
            {name} 
            [version:{version},
            Author: {author}]
        """.format(
            name='AlphaGomoku',
            version=__version__,
            author=__author__
        )

    parser = argparse.ArgumentParser(description=info)

    parser.add_argument("--retrain", action='store_true', default=False, help="Re-Train AI")
    parser.add_argument("--train",  action='store_true', default=False, help="Train AI")
    parser.add_argument("--verbose", action='store_true', default=False, help="Verbose")
    parser.add_argument("--info", action='store_true', default=False, help="Show the process information")
    parser.add_argument("--play", action='store_true', default=False, help="Play with AI")

    args = parser.parse_args()
    verbose = args.verbose

    if args.train:
        print("Train AI")

    if args.retrain:
        print("Re-train AI")
        trainai = TrainAI(
            board_size=board_size,
            verbose=verbose)
        trainai.start()

    if args.play:
        print("Play with AI!")