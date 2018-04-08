import numpy as np
import os

class TestAI:
    def value_function(self,chessboard):
        state = chessboard.get_state()
        if len(state) == 0:
            raise ValueError("Void state!")
        action_prob = np.ones(self.input_size[1:])
        action_prob = action_prob/np.sum(action_prob)
        return 0.0, action_prob
