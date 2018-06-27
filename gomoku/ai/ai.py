import numpy as np 
from . import models
import os
import Config

HV = Config.ChessInfo['Human']
CV = Config.ChessInfo['Computer']

class AI:
    def __init__(self, input_size, output_size, hidden_layers, model="neural_network", **args):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        if model == "neural_network":
            self.model = models.NeuralNetworks(input_size, output_size, hidden_layers, **args)

        self.model.init()
        
    def value_function(self,chessboard,role,verbose=False):
        chessboardinfo = chessboard.get_chessboardinfo(role)
        X = chessboardinfo.reshape(1,self.input_size)
        value, probs = self.model.predict(X)

        positions, rec_pos = chessboard.get_availables()
        probs = probs[rec_pos]
        action_prob = list()
        for action,prob in zip(rec_pos,probs):
            action_prob.append((action,prob))
        return action_prob, value

    def update(self,gamedata):
        