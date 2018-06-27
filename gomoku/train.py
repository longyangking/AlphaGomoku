import numpy as np
from deepai.alphazero import AlphaZero 
from collections import defaultdict, deque
from selfplay import SelfplayEngine

class TrainAI:
    def __init__(self,
        board_size=(10,10),
        learning_rate=2e-3,
        lr_multiplier=1.0,
        temperature=1.0,
        n_playout=400,
        n_playout_mcts=1000,
        n_playout_checkpoint=1000,
        c_puct=5,
        buffer_size=10000,
        batch_size=512,
        selfplay_batch_size=1,
        epochs=5,
        kl_targ=0.02,
        check_freq=50,
        game_batch_size=1500,
        best_win_ratio=0.0,
        parallelize=1,
        vectorlength=3,
        verbose=False):

        self.game = None # instance of game engine
        self.board_size = board_size    # size of board
        self.learning_rate = learning_rate  # learning rate for training network
        self.lr_multiplier = lr_multiplier  # adjust learning rate adaptively based on KL
        self.temperature = temperature  # temperature to control anneal
        self.n_playout = n_playout  # number of simulations for each move
        self.n_playout_mcts = n_playout_mcts    # num of simulations to evaluate the trained network
        self.n_playout_checkpoint = n_playout_checkpoint
        self.c_puct = c_puct    # constant determining the level of exploration
        self.buffer_size = buffer_size 
        self.batch_size = batch_size    # batch size for training network
        self.selfplay_batch_size = selfplay_batch_size # number of self-play simulations
        self.epochs = epochs    # number of train steps for each update
        self.kl_targ = kl_targ
        self.check_freq = check_freq
        self.game_batch_size = game_batch_size
        self.best_win_ratio = best_win_ratio
        self.parallelize = parallelize
        self.verbose = verbose

        # TODO initiate alphazero model
        hidden_layers = list()
        hidden_layers.append({'nb_filter':20, 'kernel_size':3})
        hidden_layers.append({'nb_filter':20, 'kernel_size':3})
        hidden_layers.append({'nb_filter':20, 'kernel_size':3})
        hidden_layers.append({'nb_filter':20, 'kernel_size':3})


        input_size = (*board_size, vectorlength)
        self.ai = AlphaZero(
            input_size=input_size,
            hidden_layers=hidden_layers,
            learning_rate=1e-4,
            momentum=0.9,
            l2_const=1e-4,
            verbose=False
        )

        self.ai.init()

    def get_ai(self):
        return self.ai

    def get_data_extended(self,play_data):
        '''
        Augment the data set by rotation and flipping according to the symmetry
        '''
        extend_data = []

        # TODO achieve rotation and flipping operations

        return extend_data

    def get_selfplay_data(self):
        '''
        Get MCTS self-play data for training
        '''
        dataset = list()
        for i in range(self.selfplay_batch_size):
            selfplayengine = SelfplayEngine(
                ai=self.ai, 
                c_puct=self.c_puct, 
                n_playout=self.n_playout, 
                verbose=self.verbose)
            selfplayengine.init()
            dataset.append(selfplayengine.get_data())
        return dataset

    def network_update(self, dataset):
        '''
        update AlphaZero policy-value network
        '''
        return 0, 0

        n_dataset = len(dataset)
        Xs_train, values_train, policy_train = list(), list(), list()
        for i in range(n_dataset):
            Xs, values, policy = dataset.get_traindata()
            Xs_train.append(Xs)
            values_train.append(values)
            policy_train.append(policy)

        self.ai.train(Xs_train, [values_train, policy_train])

        return loss, entropy

    def network_evaluate(self,n_games=10):
        '''
        get win ratio 
        '''

        return win_ratio

    def savemodel(self):
        pass

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

                if (i+1) % self.check_freq == 0:
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

        except KeyboardInterrupt:
            print('\n\r Train Quit by user manually')
