import random
import numpy as np
from pyludo.utils import token_vulnerability, is_stacked, is_globe_pos, is_on_opponent_globe

class LudoPlayerQLearning:
    ####### Class variables ########
    name = 'QLearning'
    # Token states
    homed = 0
    normal = 1
    stacked = 2 # safe if player is stacked
    globe = 3 # safe on globe except other players home globe
    vulnerable = 4 # other players token can hit your token
    globe_home = 5 # other players globe at home
    goal = 6 # when token is inside goal

    # Rewards
    win = 10
    
    # Actions
    actions = [0, 1, 2, 3] # chooses the token number

    def __init__(self):
        self.__tokenState = 8 # Describes best choices for tokens based on board state
        self.__QTable = dict()

    # STATE REPRESENTATION #
    def __getTokenState(self, state):
        currentPlayerState = np.copy(state.state[0])

        # Set all to normal if no criteria is true
        currentPlayerState[:] = LudoPlayerQLearning.normal

        # Homed
        currentPlayerState[state.state[0] == -1] = LudoPlayerQLearning.homed
        
        # Vulnerable
        tmp = [token_vulnerability(state, token_id) > 0 for token_id in range(4)]
        currentPlayerState[tmp] = LudoPlayerQLearning.vulnerable

        # Stacked
        currentPlayerState[is_stacked(state.state)] = LudoPlayerQLearning.stacked

        # On globe
        tmp = [is_globe_pos(token) for token in state.state[0]]
        currentPlayerState[tmp] = LudoPlayerQLearning.globe

        # On other players home globe
        currentPlayerState[is_on_opponent_globe(state.state)] = LudoPlayerQLearning.globe_home

        # Token inside goal
        currentPlayerState[state.state[0] == 99] = LudoPlayerQLearning.goal

        print(currentPlayerState) # State is all 4 players with current player as index = 0
        return currentPlayerState


    # MAKE Q-TABLE #

    def __update_Q_table(self, state, q_value):

        self.__QTable[state] = [q_value] # Update dictionary
        


    def play(self, state, dice_roll, next_states):
        self.__getTokenState(state)
        return random.choice(np.argwhere(next_states != False))
        pass
        #return # return number token want to move