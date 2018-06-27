import numpy as np
from model import Model
from collections import defaultdict, deque

class TrainAI:
    def __init__(self, input_size):
        self.input_size = input_size
        
        self.n_round
        self.game
        self.model = Model()

    def selfplay(self):

    def update_network(self, train_data):

    def evaluate_network(self):

    def start(self):
        for i in range(self.n_round):
            train_data = self.selfplay()
            self.update_network(train_data)

            if (i+1)%self.checkpoint == 0:
                win_ratio = self.evaluate_network()
                
            
