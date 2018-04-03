from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Merge
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os

class AlphaZero:
    def __init__(self,
        input_size,
        learning_rate,
        l2_const=1e-4,
        ):
        self.input_size = input_size

        self.input_size = input_size
        self.learning_rate = learning_rate
        self.l2_const = l2_const

        self.model = None

    def init(self):
        self.model = None

    def evaluate(self,state):
        # TODO return policy_value, action_prob
        if len(state) == 0:
            raise ValueError("Void state!")
        action_prob = np.ones(self.input_size[1:])
        action_prob = action_prob/np.sum(action_prob)
        return 0.0, action_prob
        
    def value_function(self,state):
        return 0.0

    def update(self,data):
        # TODO update policy
        pass
    
    def _policy_value_block(self,input_tensor,nb_filter,kernel_size=3):
        # TODO policy value network
        out = Conv2D(nb_filter[0], 1, 1)(input_tensor)

        return out

    def _action_prob_block(self):
        out = Conv2D(nb_filter[0], kernel_size, kernel_size)(out)

        return out

    def _conv_block(self, input_tensor, nb_filter, kernel_size=3):
        out = Conv2D(nb_filter[0], 1, 1)(input_tensor)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv2D(nb_filter[1], kernel_size, kernel_size, border_mode='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv2D(nb_filter[2], 1, 1)(out)
        out = BatchNormalization()(out)

        input_tensor = Conv2D(nb_filter[2], 1, 1)(input_tensor)
        input_tensor = BatchNormalization()(input_tensor)

        out = Merge([out, input_tensor], mode='sum')
        out = Activation('relu')(out)

        return out

    def _identity_block(self, input_tensor, nb_filter, kernel_size=3):
        out = Conv2D(nb_filter[0], 1, 1)(input_tensor)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv2D(nb_filter[1], kernel_size, kernel_size, border_mode='same')(out)
        out = BatchNormalization()(out)
        out = Activation('relu')(out)

        out = Conv2D(nb_filter[2], 1, 1)(out)
        out = BatchNormalization()(out)

        out = Merge([out, x], mode='sum')
        out = Activation('relu')(out)

        return out