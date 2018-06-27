import numpy as np
from Config import Role
import Computer
import Human

class Player:
    '''
    Essential class for playing
    '''
    def __init__(self,role):
        self.role = role
        if self.role == Role.Human:
            self.player = self.Human.Human()
        if self.role == Role.Computer:
            self.player = self.Computer.Computer()

    def play(self,chessboardinfo):
        chesspos = self.player.play(chessboardinfo)
        return chesspos