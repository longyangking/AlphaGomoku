from __future__ import absolute_import

import numpy as np 
import Config
from gameengine import GameEngine
from ai import NeuralNetwork, AI

class SelfplayEngine:
    '''
    Self-play game engine to get the training data
    '''
    def __init__(self, state_shape, model, verbose=False):
        self.state_shape = state_shape
        self.model = model
        self.verbose = verbose

    def start(self):
        '''
        Start a self-play game
        '''
        # Self-player A
        roleA = 'Self_A'
        playerA = AI(
            state_shape=self.state_shape, 
            role=roleA,  # The customed string with specific meaning 
            is_selfplay=True, 
            model=self.model, 
            verbose=self.verbose
        )
        # Self-player B
        roleB = 'Self_B'
        playerB = AI(
            state_shape=self.state_shape, 
            role=roleB,  # The customed string with specific meaning 
            is_selfplay=True, 
            model=self.model, 
            verbose=self.verbose
        )

        engine = GameEngine(
            players=[playerA, playerB], 
            state_shape=self.state_shape, 
            is_selfplay=True,
            verbose=verbose)

        if self.verbose:
            print("Start to self-play...")
        winner = engine.start()
        if self.verbose:
            print("End of self-play")

        data = engine.get_data()
        data_A, data_B = zip(*data)
        _, ys_A = zip(*data_A)
        values_A, _ = zip(*ys_A)
        _, ys_B = zip(*data_B)
        values_B, _ = zip(*ys_B)

        # Not to copy, operate on the reference
        if winner == roleA:
            values_A, values_B = 1, -1
        else:
            values_A, values_B = -1, 1

        return data

class EvaluationEngine:
    '''
    Evaluation Engine to evaluate the performance of AI model
    '''
    def __init__(self):
        pass

class TrainAI:
    def __init__(self,
        state_shape=(10,10,3),
        verbose=False):

        self.state_shape = state_shape
        self.verbose = verbose

        hidden_layers = list()
        hidden_layers.append({'nb_filter':20, 'kernel_size':3})
        hidden_layers.append({'nb_filter':20, 'kernel_size':3})
        hidden_layers.append({'nb_filter':20, 'kernel_size':3})
        hidden_layers.append({'nb_filter':20, 'kernel_size':3})

        self.model = NeuralNetwork(
                input_shape=state_shape, 
                hidden_layers=hidden_layers
                )
            
        if self.verbose:
            print("Initiating AI model...",end="")
        self.model.init()
        if self.verbose:
            print("End")

    def get_model(self):
        return self.model   

    def get_selfplay_data(self):
        '''
        Get MCTS self-play data for training
        '''
        selfplayengine = SelfplayEngine(
                state_shape=self.state_shape, 
                model=self.model, 
                verbose=False)
        data = selfplayengine.start()
        return data

    def network_update(self, dataset):
        '''
        update AlphaZero policy-value network
        '''
        loss, entropy = 0, 0
        
        # TODO update Neural network model

        return loss, entropy

    def network_evaluate(self,n_games=10):
        '''
        get win ratio 
        '''
        win_ratio = 0

        # TODO evaluate Neural network model

        return win_ratio

    def start(self):
        '''
        start to train in pipline
            1. MCTS self-play and then update network
            2. Competition among serveral candidates to obtain best
            hyper-parameters for MCTS
            3. Return 1 until convergence or satisfying the end conditions
        '''
        epochs = 1000
        check_freq = 50

        for i in range(epochs):
            if self.verbose:
                print("Train Batch {0}: ".format(i+1))
            play_data = self.get_selfplay_data()
            
            if self.verbose:
                print("Updating network...",end="")
            loss, entropy = self.network_update(play_data)
            if self.verbose:
                print("End with loss = {loss}, entropy = {entropy}".format(
                    loss=loss,
                    entropy=entropy
                ))

            if (i+1) % check_freq == 0:
                if self.verbose:
                    print("Checkpoint: Evaluating...",end="")
                win_ratio = self.network_evaluate()
                if self.verbose:
                    print("Saving model...",end="")
                self.ai.save("./model.h5")
                if win_ratio > self.best_win_ratio:
                    print("New best AI...",end="")
                    self.best_win_ratio = win_ratio
                    self.ai.save("./best_model.h5")
                    if (self.best_win_ratio == 1.0 and self.n_playout_checkpoint < 5000):
                        self.n_playout_checkpoint += 1000
                        self.best_win_ratio = 0.0
