import numpy as np 
import Config
import Chessboard
import Gamedata

class Gameengine:
    def __init__(self,playerA,playerB):
        self.playerA = playerA
        self.playerB = playerB
        self.chessboard = Chessboard.Chessboard()

    def init(self):
        self.chessboard.init()

    def run(self,op='A'):
        # Offensive Position, default: Player A
        if op == 'A':
            players = [self.playerA,self.playerB]
        else:
            players = [self.playerB,self.playerA]

        gamedata = Gamedata(board_shape=self.chessboard.chessboardinfo().shape)
        status,winner = self.chessboard.victoryjudge()
        # The game will continue to battle it out
        while not status:
            for i in range(len(players)):
                chesspos = self.players[i].play(self.gamedata.getstate())
                while not self.chessboard.putchess(chesspos,role=self.players[i].role):
                    chesspos = self.players[i].play(self.gamedata.getstate())
                # Save chessboard info
                status,winner = self.chessboard.victoryjudge(self.players[i].role)
                gamedata.append(self.gamedata.getstate())
                if status:
                    break
        # Have winner
        gamedata.gameend(winner)
        return gamedata

    def getchessboard(self):
        return self.chessboard