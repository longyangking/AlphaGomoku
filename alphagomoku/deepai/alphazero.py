import numpy as np 
import keras

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