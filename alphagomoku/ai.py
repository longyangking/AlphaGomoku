from __future__ import absolute_import

import numpy as np 
import copy
import Config
from operator import itemgetter

def softmax(x):
    probs = np.exp(x-np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode:
    def __init__(self,parent,prior_p):
        self._parent = parent
        self._childern = {} # Save childre nodes in Hash data structure
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self,action_priors):
        #print(action_priors)
        for action, prob in action_priors:
            if action not in self._childern:
                self._childern[action] = TreeNode(self,prob)

    def select(self,c_puct):
        return max(self._childern.items(),
            key=lambda action_node: action_node[1].get_value(c_puct)
            )

    def update(self,leaf_value):
        self._n_visits += 1
        self._Q += 1.0*(leaf_value - self._Q) / self._n_visits

    def update_recursive(self, leaf_value):
        if self._parent:
            self._parent.update_recursive(leaf_value)
        self.update(leaf_value)

    def get_value(self,c_puct):
        self._u = (c_puct * self._P * np.sqrt(self._parent._n_visits) / (1 + self._n_visits))
        return self._Q + self._u

    def is_leaf(self):
        return self._childern == {}

    def is_root(self):
        return self._parent is None

class MCTS:
    def __init__(self,value_function, c_puct, n_playout, role, verbose=False):
        self._root = TreeNode(None, 1.0)
        self._value_function = value_function
        self._c_puct = c_puct
        self._n_playout = n_playout
        self.role = role
        self.verbose = verbose

    def _playout(self, chessboard):
        node = self._root
        roles = chessboard.get_roles()
        role_index = [index for (index,role) in zip(range(len(roles)),roles) if role==self.role][0]

        while 1:
            if node.is_leaf():
                break
            action, node = node.select(self._c_puct)
            chessboard.playchess_rec(pos_rec=action,role=roles[role_index])
            role_index = (role_index + 1)%len(roles)

        leaf_value, action_probs = self._value_function(chessboard=chessboard, role=roles[role_index], verbose=self.verbose)
        end, winner =  chessboard.get_status()

        if not end:
            node.expand(action_probs)
        else:
            # for end state，return the "true" leaf_value
            if winner == None:  # tie
                leaf_value = 0.0
            else:
                leaf_value = (
                    1.0 if winner == self.role else -1.0
                )
        
        # Update the whole MC tree recursively
        node.update_recursive(leaf_value)

    def get_move_probs(self, chessboard, temperature, eps=1e-10):
        for i in range(self._n_playout):
            _chessboard = copy.deepcopy(chessboard)
            self._playout(_chessboard)

        action_visits = [(action, node._n_visits) 
                        for action, node in self._root._childern.items()]
        print(action_visits)
        actions, visits = zip(*action_visits)
        action_probs = softmax(1.0/temperature*np.log(np.array(visits) + eps))
        return actions, action_probs

    def get_move(self, chessboard):
        for i in range(self._n_playout):
            _chessboard = copy.deepcopy(chessboard)
            self._playout(_chessboard)
        return max(self._root._childern.items(),
            key=lambda action_node: action_node[1]._n_visits)[0]

    def _evaluate_rollout(self, chessboard, n_round):
        # TODO Pure playout
        #for i in range(n_round):
        #    end, winner = 
        #    if end:
        #        break
        #    action_probs = self._rollout_value_function(chessboard)
        #    _action = max(action_probs, key=itemgetter(1))[0]
        #    chessboard.play(_action)
        #else:
        #    if self.verbose:
        #        print("Warning: Rollout reaches round limit")
        pass
        # TODO default player

    def _rollout_value_function(self, chessboard):
        pass

    def update_with_move(self, move):
        if move in self._root._childern:
            self._root = self._root._childern[move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def __str__(self):
        return "Monte Carlo Tree Search"

class MCTSPlayer:
    def __init__(self, value_function, c_puct, n_playout, is_selfplay, role, verbose=False):
        self.mcts = MCTS(value_function, c_puct, n_playout, role, verbose)
        self._is_selfplay = is_selfplay
        self.role = role
        self.verbose = verbose

    def set_player_index(self, player):
        self.player = player

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def play(self, chessboard):
        move, action_prob = self.get_action(chessboard=chessboard, temperature=1.0)
        x = int(move/board.shape[0])
        y = move%board.shape[1]
        return (x,y)

    def get_action(self, chessboard, temperature, return_prob=0):
        move_probs = np.zeros(chessboard.get_shape())
        is_available = chessboard.is_available()

        if is_available:
            actions, probs = self.mcts.get_move_probs(chessboard, temperature)
            positions = chessboard.rec2pos(actions)
            move_probs[positions] = probs
            if self._is_selfplay:
                move = np.random.choice(
                    actions,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(actions,p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            else:
                return move
        else:
            if self.verbose:
                # TODO need modification
                print("The board is full")

    def play(self,chessboard):
        temperature = 1.0
        action = self.get_action(chessboard=chessboard,temperature=temperature,return_prob=False)
        return chessboard.rec2pos([action])

    def __str__(self):
        return "Monte Carlo Tree Search: {player}".format(player=player)

def value_function_check(state):
    # equivalent probability to rollout
    pass

class Checkplayer:
    def __init__(self, c_puct, n_playout):
        self.mcts = MCTS(value_function_check, c_puct, n_playout)

    def set_player_index(self, player):
        self.player = player
    
    def reset_player(self):
        self.mcts.update_with_move(-1)

    def get_action(self, state):
        pass

from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Flatten, BatchNormalization, Activation, LeakyReLU, add
from keras.optimizers import SGD
from keras import regularizers
import keras.backend as K
import tensorflow as tf

import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # Only error will be shown

def softmax_cross_entropy_with_logits(y_true, y_pred):
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
	return loss

class AlphaZero:
    def __init__(self,
        input_size,
        hidden_layers,
        learning_rate=1e-4,
        momentum=0.9,
        l2_const=1e-4,
        verbose=False
        ):

        self.input_size = input_size
        self.hidden_layers = hidden_layers

        self.learning_rate = learning_rate
        self.l2_const = l2_const
        self.momentum = momentum

        self.output_size = self.input_size[0]*self.input_size[1]
        self.model = None

        self.verbose = verbose

    def init(self):
        main_input = Input(shape = self.input_size, name = 'main_input')

        x = self._conv_block(main_input, self.hidden_layers[0]['nb_filter'], self.hidden_layers[0]['kernel_size'])
        if len(self.hidden_layers) > 1:
            for h in self.hidden_layers[1:]:
                x = self._res_block(x, h['nb_filter'], h['kernel_size'])

        value = self._policy_value_block(x)
        action_prob = self._action_prob_block(x)

        self.model = Model(inputs=[main_input], outputs=[value,action_prob])
        self.model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
			optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),	
			loss_weights={'value_head': 0.5, 'policy_head': 0.5}	
			)
        
    def value_function(self, chessboard, role, verbose=False):
        state = chessboard.get_state()
        state = np.array(state)
        #print(state.shape)
        state = state.reshape(1, *self.input_size)

        value, prob_act = self.model.predict(state)
        _, acts = chessboard.get_availables()
        action_probs = zip(acts, prob_act[0][acts])
        return value[0][0], action_probs

    def train(self, X, y, batchsize=128, epochs=30, validation_split=0.1, verbose=False):
        self.model.fit(X, y, epochs=epochs, batchsize=batchsize, validation_split=validation_split, verbose=verbose)

    def update(self, dataset):
        states, values, policys  = dataset.get_traindata()
        self.model.train_on_batch(states, [values, policys])
    
    def _policy_value_block(self,input_tensor):
        out = Conv2D(
            filters = 1,
            kernel_size = (1,1),
            data_format="channels_first",
            padding = 'same',
            use_bias=False,
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
		)(input_tensor)

        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)
        out = Flatten()(out)

        out = Dense(
			20,
            use_bias=False,
            activation='linear',
            kernel_regularizer= regularizers.l2(self.l2_const)
		)(out)

        out = LeakyReLU()(out)

        value = Dense(
			1, 
            use_bias=False,
            activation='tanh',
            kernel_regularizer=regularizers.l2(self.l2_const),
            name = 'value_head'
			)(out)

        return value

    def _action_prob_block(self, input_tensor):
        out = Conv2D(
            filters = 2,
            kernel_size = (1,1),
            data_format="channels_first"
            , padding = 'same'
            , use_bias=False
            , activation='linear'
            , kernel_regularizer = regularizers.l2(self.l2_const)
            )(input_tensor)

        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)
        out = Flatten()(out)

        out = Dense(
			self.output_size,
            use_bias=False,
            activation='linear',
            kernel_regularizer=regularizers.l2(self.l2_const),
            name = 'policy_head'
			)(out)

        return out

    def _conv_block(self, input_tensor, nb_filter, kernel_size=3):
        out = Conv2D(
            filters = nb_filter,
            kernel_size = kernel_size,
            data_format="channels_first",
            padding = 'same',
            use_bias=False,
            activation='linear',
            kernel_regularizer = regularizers.l2(self.l2_const)
		)(input_tensor)

        out = BatchNormalization(axis=1)(out)
        out = LeakyReLU()(out)

        return out

    def _res_block(self, input_tensor, nb_filter, kernel_size=3):
        out = self._conv_block(input_tensor, nb_filter, kernel_size)

        out = Conv2D(
                filters = nb_filter,
                kernel_size = kernel_size,
                data_format="channels_first",
                padding = 'same',
                use_bias=False,
                activation='linear',
                kernel_regularizer = regularizers.l2(self.l2_const)
		    )(out)

        out = BatchNormalization(axis=1)(out)
        out = add([input_tensor, out])
        out = LeakyReLU()(out)

        return out



if __name__=='__main__':
    input_size = (3,5,5)
    hidden_layers = list()
    hidden_layers.append({'nb_filter':20, 'kernel_size':3})
    hidden_layers.append({'nb_filter':20, 'kernel_size':3})
    hidden_layers.append({'nb_filter':20, 'kernel_size':3})
    hidden_layers.append({'nb_filter':20, 'kernel_size':3})
    alphazero = AlphaZero(input_size, hidden_layers)
    alphazero.init()

    from keras.utils import plot_model
    plot_model(alphazero.model,show_shapes=True,show_layer_names=False, to_file='model.png')
    print(alphazero.model.summary())

    # input_data = np.random.random((1,*input_size))
    # value, action_prob = alphazero.model.predict(input_data)
    # print(value)
    # print(action_prob)
        