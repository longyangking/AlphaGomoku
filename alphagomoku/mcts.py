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
        