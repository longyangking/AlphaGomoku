import numpy as np 
import argparse

class Alphagomoku:
    def __init__(self):
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
    parser.add_argument("--verbose", action='store_true', default=True, help="Show the process information")

    args = parser.parse_args()
    if args.train:
        print("Train AI")

    if args.retrain:
        print("Re-train AI")

    if args.verbose:
        print("Verbose is on")