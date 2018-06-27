import numpy as np 
import Config
import Chessboard

class GameEngine:
    def __init__(self,playerA,playerB,verbose=False):
        self.playerA = playerA
        self.playerB = playerB
        self.chessboard = Chessboard.Chessboard(players=[playerA,playerB])
        self.verbose = verbose

    def init(self):
        self.chessboard.init()

    def run(self,op='A'):
        # Offensive Position, default: Player A
        if op == 'A':
            players = [self.playerA,self.playerB]
        else:
            players = [self.playerB,self.playerA]

        status,winner = self.chessboard.get_status()
        # The game will continue to battle it out
        while not status:
            for i in range(len(players)):
                chesspos = players[i].play(self.chessboard)
                while not self.chessboard.playchess(chesspos,role=players[i].role):
                    chesspos = players[i].play(self.chessboard)
                # Save chessboard info
                status,winner = self.chessboard.victoryjudge(role=players[i].role)
                if status:
                    break
                if self.verbose:
                    print("{step}th Self-play step ...".format(step=self.chessboard.get_stepinfo()))
        return self.chessboard.get_game_data()
