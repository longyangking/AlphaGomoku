import numpy as np 
import argparse
from trainutils import TrainAI
from gomoku import GameEngine
import Config

__board_width__, __board_height__ = 10,10
__channel_depth__ = 7
__state_shape__ = __board_width__, __board_height__, __channel_depth__

__version__ = "0.0.1"
__author__ = "Yang Long"
__info__ = """
            {name} 
            [version:{version},
            Author: {author}]
        """.format(
            name='AlphaGomoku',
            version=__version__,
            author=__author__
        )

if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__info__)

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
        trainprocess = TrainAI(state_shape=__state_shape__,verbose=verbose)
        trainprocess.start()

    if args.info:
        print("Info")

    if args.play:
        print("Play with AI!")
        engine = GameEngine(state_shape=__state_shape__, verbose=verbose)
        engine.start()
