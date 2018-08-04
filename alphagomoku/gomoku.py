import numpy as np 
from ai import AI
from gameutils import ChessBoard
import sys
import viewer

class GameEngine:
    def __init__(self, state_shape, filename='best_model.h5',verbose=False):
        self.verbose = verbose
        self.ai = AI(state_shape=state_shape, verbose=self.verbose)
        self.ai.load_nnet(filename)

        self.state_shape = state_shape
        self.channel = self.state_shape[2] - 1 # Even number

        # state_shape = [lenght, height, channel]
        self.chessboard = ChessBoard(board_shape=self.state_shape[:2])
        self.boards = list()

        # Train data
        self.states = list()

        self.chesses = [1, -1]
        self.current_player = 0

        # Control game
        self.waitingforplay = True
        self.human_flag = False

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

    def get_state(self):
        '''
        Get state vector of the current player
        '''
        if len(self.states) == 0:
            state = np.zeros(self.state_shape)
        else:
            state = np.array(self.states[-1])

        return state

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

    def get_availables(self):
        actions = self.chessboard.get_availables()
        return actions

    def play_human(self, pos):
        Nx, Ny = self.chessboard.get_shape()
        action = pos[0] + Nx*pos[1]

        if action in self.get_availables():
            self.human_flag = self.play(action)
            self.waitingforplay = False
    
    def start(self):
        self.ui = viewer.UI(pressaction=self.play_human,chessboardinfo=self.chessboard.get_board())
        self.ui.start()

        while True:
            while self.waitingforplay:
                pass 
            if self.human_flag:
                self.endgame(role='Human')
                break

            action = self.ai.play(self)
            computer_flag = self.play(action)
            if computer_flag:
                self.endgame(role='Computer')
                break
                
            self.ui.setchessboard(self.chessboard.get_board())
            self.waitingforplay = True

    def endgame(self, role):
        print(role + " Win")
        self.ui.gameend(role)
        sys.exit()