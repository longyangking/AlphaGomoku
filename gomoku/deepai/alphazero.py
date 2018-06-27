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