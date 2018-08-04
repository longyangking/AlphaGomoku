from __future__ import absolute_import

import numpy as np 
import Config
from gameutils import ChessBoard
from ai import AI

class SelfplayEngine:
    '''
    Self-play game engine to get the training data
    '''
    def __init__(self, ai, verbose=False):
        self.ai = ai
        self.state_shape = ai.get_state_shape()
        self.verbose = verbose

        self.channel = self.state_shape[2] - 1 # Even number

        # state_shape = [lenght, height, channel]
        self.chessboard = ChessBoard(board_shape=self.state_shape[:2])
        self.boards = list()

        # Train data
        self.states = list()
        self.pis = list()

        # player information
        self.current_player = 0
        self.chesses = [1, -1]

        # Control for MCTS
        self.dataset_mcts_index = 0
        self.chessboard_mcts = None
        self.current_player_mcts = 0

    def get_availables(self):
        actions = self.chessboard.get_availables()
        return actions

    def update_current_player(self):
        self.current_player += 1
        self.current_player %= 2

    def update_states(self):
        '''
        Update states

        state: [X_{t-channel+1}, Y_{t-channel+1}, ..., X{t}, Y{t}, C]
        '''
        player = self.current_player
        opposite_player = (self.current_player + 1)%2

        state = np.zeros(self.state_shape)
        # feature plane
        state[:,:,-1] = player*np.ones(self.state_shape[:-1])

        # state planes
        time_steps = int(self.channel/2)
        len_boards = len(self.boards)
        for i in range(time_steps,):
            if len_boards-1-i >= 0:
                index = 2*i
                board = self.boards[len_boards-1-i]
                state[:,:, self.channel-1-index] = 1*(board == self.chesses[player])
                state[:,:, self.channel-1-(index+1)] = 1*(board == self.chesses[opposite_player])

        # Append it into the state list
        self.states.append(state)        

    def get_traindata(self):
        states = self.states
        pis = self.pis
        values = self.values

        return states, pis, values

    def get_state(self):
        '''
        Get state vector of the current player
        '''
        if len(self.states) == 0:
            state = np.zeros(self.state_shape)
        else:
            state = np.array(self.states[-1])

        return state

    def init_mcts(self):
        '''
        Record the status before MCTS
        '''
        self.current_player_mcts = self.current_player
        self.states_mcts_index = len(self.states)
        self.chessboard_mcts = self.chessboard.clone()

    def exit_mcts(self):
        '''
        Recover the status after MCTS
        '''
        self.current_player = self.current_player_mcts
        self.states = self.states[:self.states_mcts_index]
        self.chessboard = self.chessboard_mcts

    def play_mcts(self, action):
        '''
        Play game in MCTS
        '''
        flag = self.chessboard.play(action)
        
        # Update data
        board = self.chessboard.get_board()
        self.boards.append(board)
        self.update_current_player()
        self.update_states()

        return flag

    def play(self, action):
        '''
        Play game formally
        '''
        flag = self.chessboard.play(action)
        
        # Update data
        board = self.chessboard.get_board()
        self.boards.append(board)
        self.update_current_player()
        self.update_states()

        return flag

    def start(self, temperature=1.0):
        '''
        Start a self-play game
        '''

        # Initiate states
        state = np.zeros(self.state_shape)
        self.states.append(state)

        while True:
            action, pi = self.ai.play(self, is_selfplay=True, temperature=temperature)
            self.pis.append(pi)

            flag = self.play(action)
            if flag:
                break

        # Just for debugging
        if flag:
            if len(self.pis) < 5:
                raise Exception("Error in logic about chessing")

        n_values = len(self.pis)
        values = np.ones(n_values)
        if flag == 1:
            values[range(1,n_values,2)] = -1
        else:
            values[range(0,n_values,2)] = -1

        return self.states, self.pis, values

class EvaluationEngine:
    '''
    Evaluation Engine to evaluate the performance of AI model
    '''
    def __init__(self, ai, verbose=False):
        self.ai = ai
        self.verbose = verbose

        # player information
        self.states = list()
        self.current_player = 0
        self.chesses = [1, -1]

        self.state_shape = ai.get_state_shape()
        self.channel = self.state_shape[2] - 1 # Even number

        self.state_shape = self.ai.get_state_shape()
        self.chessboard = None
        self.boards = list()
    
    def get_availables(self):
        actions = self.chessboard.get_availables()
        return actions

    def update_states(self):
        '''
        Update states

        state: [X_{t-channel+1}, Y_{t-channel+1}, ..., X{t}, Y{t}, C]
        '''
        player = self.current_player
        opposite_player = (self.current_player + 1)%2

        state = np.zeros(self.state_shape)
        # feature plane
        state[:,:,-1] = player*np.ones(self.state_shape[:-1])

        # state planes
        time_steps = int(self.channel/2)
        len_boards = len(self.boards)
        for i in range(time_steps,):
            if len_boards-1-i >= 0:
                index = 2*i
                board = self.boards[len_boards-1-i]
                state[:,:, self.channel-1-index] = 1*(board == self.chesses[player])
                state[:,:, self.channel-1-(index+1)] = 1*(board == self.chesses[opposite_player])

        # Append it into the state list
        self.states.append(state) 

    def get_state(self):
        '''
        Get state vector of the current player
        '''
        if len(self.states) == 0:
            state = np.zeros(self.state_shape)
        else:
            state = np.array(self.states[-1])

        return state

    def update_current_player(self):
        self.current_player += 1
        self.current_player %= 2

    def play(self, action):
        flag = self.chessboard.play(action)
        
        # Update data
        board = self.chessboard.get_board()
        self.boards.append(board)
        self.update_current_player()
        self.update_states()

        return flag

    def start(self, test_ai, n_playout=100, temperature=1.0):
        '''
        Start evaluation
        '''
        if self.verbose:
            print("Start to evaluate...")

        win_count = 0
        for i in range(n_playout):
            self.chessboard = ChessBoard(board_shape=self.state_shape[:2])

            while True:
                action = self.ai.play(self, temperature=temperature)
                flag = self.play(action)

                if flag:
                    break

                action = test_ai.play(self, temperature=temperature)
                flag = self.play(action)

                if flag:
                    win_count += 1
                    break
                
        if self.verbose:
            print("End of evaluation.")

        win_ratio = win_count/n_playout
        return win_ratio

class TrainAI:
    def __init__(self,
        state_shape,
        verbose=False):

        self.state_shape = state_shape
        self.verbose = verbose

        self.ai = AI(state_shape=state_shape, verbose=verbose)
        self.best_ai = AI(state_shape=state_shape, verbose=verbose)

    def get_selfplay_data(self, n_round):
        '''
        Get MCTS self-play data for training. The training data is obtained by "best" AI model.
        '''
        all_states, all_pis, all_values = list(), list(), list()

        if self.verbose:
            print("Starting to self-play to get dataset with size: {0}".format(n_round))

        for i in range(n_round):
            print("{0}th self-playing...".format(i+1))

            selfplayengine = SelfplayEngine(ai=self.best_ai, verbose=self.verbose)
            states, pis, values = selfplayengine.start()

            for i in range(len(values)):
                all_states.append(states[i])
                all_pis.append(pis[i])
                all_values.append(values[i])
        
        all_states = np.array(all_states)
        all_pis = np.array(all_pis)
        all_values = np.array(all_values).reshape(-1,1)

        if self.verbose:
            print("End of self-play with shape [state: {0}; pi: {1}; value: {2}]".format(
                all_states.shape,
                all_pis.shape,
                all_values.shape
            ))

        return [all_states, all_values, all_pis]

    def network_update(self, dataset, verbose=False):
        '''
        update AlphaZero policy-value network
        '''
        loss = self.ai.update_network(dataset, epochs=100, batch_size=64, verbose=verbose)
        return loss

    def network_evaluate(self):
        '''
        Get win ratio 
        '''
        engine = EvaluationEngine(ai=self.best_ai, verbose=self.verbose)
        win_ratio = engine.start(test_ai=self.ai)
        return win_ratio

    def start(self):
        '''
        start to train in pipline
            1. MCTS self-play and then update network
            2. Competition among serveral candidates to obtain best
            hyper-parameters for MCTS
            3. Return 1 until convergence or satisfying the end conditions
        '''
        epochs = 50000
        check_freq = 100
        n_round = 30

        for i in range(epochs):
            if self.verbose:
                print("Train Batch {0}: ".format(i+1))
            selfplaydata = self.get_selfplay_data(n_round=n_round)
            
            if self.verbose:
                print("Updating network...")
            loss = self.network_update(selfplaydata, verbose=self.verbose)
            if self.verbose:
                print("End of updating with loss = {loss:.4f}.".format(loss=loss.history['loss'][-1]))

            if (i+1) % check_freq == 0:
                if self.verbose:
                    print("Checkpoint: Evaluating...")
                win_ratio = self.network_evaluate()
                if self.verbose:
                    print("End of Evaluation with win ratio: {0}".format(win_ratio))

                if self.verbose:
                    print("Saving model...")
                self.ai.save_nnet("./model.h5")

                if win_ratio > 0.55:
                    print("Get new best AI")
                    self.ai.save_nnet("./best_model.h5")
                    self.best_ai = self.ai.clone()

