import numpy as np 
import Config
import Chessboard

class Gameengine:
    def __init__(self,playerA,playerB):
        self.playerA = playerA
        self.playerB = playerB
        self.chessboard = Chessboard.Chessboard()

    def run(self,op='A'):
        # Offensive Position, default: Player A
        if op == 'A':
            players = [self.playerA,self.playerB]
        else:
            players = [self.playerB,self.playerA]

        status,winner = self.chessboard.victoryjudge()
        while not status:
            for i in range(len(players)):
                chesspos = self.players[i].play(self.chessboard.chessboardinfo())
                while not self.chessboard.putchess(chesspos,role=self.players[i].role):
                    chesspos = self.players[i].play(self.chessboard.chessboardinfo())
                status,winner = self.chessboard.victoryjudge(self.players[i].role)
                if status:
                    break
        return winner
