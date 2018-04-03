import numpy as np 
import Config
import Chessboard

class Gameengine:
    def __init__(self,playerA,playerB):
        self.playerA = playerA
        self.playerB = playerB
        self.chessboard = Chessboard.Chessboard(players=[playerA,playerB])

    def init(self):
        self.chessboard.init()

    def run(self,op='A'):
        # Offensive Position, default: Player A
        if op == 'A':
            players = [self.playerA,self.playerB]
        else:
            players = [self.playerB,self.playerA]

        status,winner = self.chessboard.victoryjudge()
        # The game will continue to battle it out
        while not status:
            for i in range(len(players)):
                chesspos = self.players[i].play(self.chessboard)
                while not self.chessboard.playchess(chesspos,role=self.players[i].role):
                    chesspos = self.players[i].play(self.chessboard)
                # Save chessboard info
                status,winner = self.chessboard.victoryjudge(self.players[i].role)
                if status:
                    break
        return self.chessboard.get_game_data()