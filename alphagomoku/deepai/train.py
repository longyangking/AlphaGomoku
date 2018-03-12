import numpy as np
from alphazero import AlphaZero 
from mcts import MCTS 

class TrainAI:
    def __init__(self,
        game,
        board_size=(10,10),
        learning_rate=2e-3,
        lr_multiplier=1.0,
        temperature=1.0,
        n_playout=400,
        n_playout_mcts=1000,
        c_puct=5,
        buffer_size=10000,
        batch_size=512,
        selfplay_batch_size=1,
        epochs=5,
        kl_targ=0.02,
        check_freq=50,
        game_batch_size=1500,
        best_win_ratio=0.0,
        parallelize=False):

        self.game = game # instance of game engine
        self.board_size = board_size    # size of board
        self.learning_rate = learning_rate  # learning rate for training network
        self.lr_multiplier = lr_multiplier  # adjust learning rate adaptively based on KL
        self.temperature = temperature  # temperature to control anneal
        self.n_playout = n_playout  # number of simulations for each move
        self.n_playout_mcts = n_playout_mcts    # num of simulations to evaluate the trained network
        self.c_puct = c_puct    # constant determining the level of exploration
        self.buffer_size = buffer_size 
        self.batch_size = batch_size    # batch size for training network
        self.selfplay_batch_size = selfplay_batch_size
        self.epochs = epochs
        self.kl_targ = kl_targ
        self.check_freq = check_freq
        self.game_batch_size = game_batch_size
        self.best_win_ratio = best_win_ratio
        self.parallelize = parallelize

        # TODO initiate alphazero model
        self.ai = AlphaZero()

        # TODO initiate mcts self-play model
        self.mcts_player = MCTSPlaye()

    def get_data_extended(self,play_data):
        '''
        Augment the data set by rotation and flipping according to the symmetry
        '''
        extend_data = []

        # TODO achieve rotation and flipping operations

        return extend_data

    def get_selfplay_data(self, n_games):
        '''
        Get MCTS self-play data for training
        '''

    def network_update(self):
        '''
        update AlphaZero policy-value network
        '''

        return loss, entropy

    def network_evaluate(self,n_games=10)
        '''
        get win ratio 
        '''

        return win_ratio

    def start(self):
        '''
        start to train in pipline
            1. MCTS self-play and then update network
            2. Competition among serveral candidates to obtain best
            hyper-parameters for MCTS
            3. Return 1 until convergence or satisfying the end conditions
        '''
        try:
            for i in range(self.game_batch_size):
                self.get_selfplay_data()
        except KeyboardInterrupt:
            print('\n\r Train Quit by user manually')
