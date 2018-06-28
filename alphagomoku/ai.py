from __future__ import absolute_import
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

import copy
import Config
from operator import itemgetter

def softmax(x):
    probs = np.exp(x-np.max(x))
    probs /= np.sum(probs)
    return probs

class TreeNode:
    '''
    Monte Carlo Tree Node
    '''
    def __init__(self,parent,prior_p):
        self._parent = parent
        self._childern = {} 
        self._n_visits = 0
        self._Q = 0
        self._u = 0
        self._P = prior_p

    def expand(self,action_priors):
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
    def __init__(self, evaluate_function, c_puct, n_playout, role, verbose=False):
        self.root = TreeNode(None, 1.0)
        self.evaluate_function = evaluate_function
        self.c_puct = c_puct
        self.n_playout = n_playout
        self.role = role
        self.verbose = verbose

    def search(self, engine):
        '''
        Simulate MCTS to search best policy for current status
        '''
        eps = 1e-12
        s_reversal = 0

        for i in range(self.n_playout):
            if self.root.is_leaf():
                # Expand and Evaluate
                value, action_probs = self.evaluate_function(
                    engine, 
                    role=self.role, 
                    s_reversal=s_reversal
                    )
                s_reversal = (s_reversal + 1)%2

                self.root.expand(zip(*action_probs))

                # Backup
                self.root.update(value)
            else:
                # Select 
                action, node = self.root.select(self.c_puct)
                engine.play(action)
                while not node.is_leaf():
                    action, node = node.select(self.c_puct)
                    engine.play(action)

                # Expand and Evaluate
                value, action_probs = self.evaluate_function(
                    engine, 
                    role=self.role, 
                    s_reversal=s_reversal
                    )
                s_reversal = (s_reversal + 1)%2

                node.expand(zip(*action_probs))
                # Backup
                node.update_recursive(value)

        # Play: Return simulation results
        actions_visits = [(action, node._n_visits) for action, node in self.root._childern.items()]
        actions, visits = zip(*actions_visits)
        probs = softmax(1.0/temperature*np.log(np.array(visits) + eps))
        return actions, probs

def softmax_cross_entropy_with_logits(y_true, y_pred):
	loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true, logits=y_pred)
	return loss

class NeuralNetwork:
    def __init__(self,
        input_shape,
        hidden_layers,
        learning_rate=1e-4,
        momentum=0.9,
        l2_const=1e-4,
        verbose=False
        ):

        self.input_shape = input_shape
        self.hidden_layers = hidden_layers

        self.learning_rate = learning_rate
        self.l2_const = l2_const
        self.momentum = momentum

        self.output_size = self.input_shape[0]*self.input_shape[1]
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

        self.model = Model(inputs=[main_input], outputs=[value, action_prob])
        self.model.compile(loss={'value_head': 'mean_squared_error', 'policy_head': softmax_cross_entropy_with_logits},
			optimizer=SGD(lr=self.learning_rate, momentum=self.momentum),	
			loss_weights={'value_head': 0.5, 'policy_head': 0.5}	
			)
    
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

class AI:
    def __init__(self, state_shape, role, is_selfplay, model=None, verbose=False):
        self.role = role
        self.is_selfplay = is_selfplay
        self.verbose = verbose

        if model is not None:
            self.model = model
        else:
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

    def load_model(self, filename):
        '''
        Load Model with file name
        '''
        from keras.models import load_model as LOAD_MODEL
        self.model = LOAD_MODEL(filename)

    def save_model(self, filename):
        '''
        Save model with file name
        '''
        self.model.save(filename)

    def plot_model(self, filename='model.png'):
        from keras.utils import plot_model
        plot_model(self.model, show_shapes=True, to_file=filename)

    def evaluate_function(self, engine, role=None, s_reversal=False):
        '''
        Evaluate function
        '''
        if role is None:
            role = self.role
        state = engine.get_state(role, s_reversal)
        value, probs = self.model.predict(state)

        actions = engine.get_availables()
        probs = probs[0][actions]
        probs = probs/np.sum(probs)
        action_probs = zip(actions, probs) # actions with their probabilities
        return value[0][0], action_probs

    def train(self, data, batchsize=128, epochs=30, verbose=False):
        '''
        Train AI
        '''
        Xs, ys = zip(*data)
        states = Xs
        values, policys = zip(*ys) 
        self.model.fit(states, [values, policys] 
            epochs=epochs, 
            batchsize=batchsize, 
            verbose=verbose)

    def update(self, data):
        '''
        Update AI
        '''
        Xs, ys = zip(*data)
        states = Xs
        values, policys = zip(*ys) 
        self.model.train_on_batch(states, [values, policys])

    def play(self, engine, temperature=1.0):
        '''
        Play game (or self-play)
        '''
        n_search = 10
        c_puct = 0.95
        n_playout = 10

        if self.is_selfplay:
            actions = list()
            for i in range(n_search):  
                _engine = engine.clone()
                mcts = MCTS(
                        evaluate_function=self.evaluate_function, 
                        c_puct=c_puct, 
                        n_playout=n_playout, 
                        verbose=self.verbose
                    )
                # MCTS_actions should be 1D vector
                mcts_actions, probs = self.mcts.search(engine=_engine) 
                action = np.random.choice(
                    mcts_actions,
                    p=0.75*probs + 0.25*np.random.dirichlet(0.3*np.ones(len(probs)))
                )
                actions.append(action)
            action = np.random.choice(action)
        else:
            value, action_probs = self.evaluate_function(engine)
            actions, probs = zip(*action_probs)
            index = np.argmax(probs)
            action = actions[index]

        return action


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
        