import random
import csv
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

    def __updateQTable(self, tokenState, qValue):
        # Update dictionary
        strTokenState = str(tokenState)

        if (strTokenState in self.__QTable):
            tmpQValue = self.__QTable[strTokenState]
            self.__QTable[strTokenState] = np.add(qValue, tmpQValue)  
        # Make new entry
        else:
            self.__QTable[strTokenState] = qValue

    def saveQTable(self):
        w = csv.writer(open("QTable.csv", "w"))
        for key, val in self.__QTable.items():
            w.writerow([key, val])


    def printQTable(self):
        print(self.__QTable)
        print("argmax", np.argmax(self.__QTable[str(np.array([0, 0, 0, 0]))]))


    # MAKE POLICiES #
    
    def epsilonGreedyPolicy(self, QTable, epsilon): 
        """ 
        Creates an epsilon-greedy policy based 
        on a given Q-function and epsilon. 
        
        Returns a function that takes the state 
        as an input and returns the probabilities 
        for each action in the form of a numpy array  
        of length of the action space(set of possible actions). 
        """
        def policyFunction(tokenState): 
            num_actions = 4
            Action_probabilities = np.ones(num_actions, 
                    dtype = float) * epsilon / num_actions 
            print(Action_probabilities)
            best_action = np.argmax(QTable[tokenState]) 
            Action_probabilities[best_action] += (1.0 - epsilon) 
            return Action_probabilities 
    
        return policyFunction 
    




    def play(self, state, dice_roll, next_states):
        tokenState = self.__getTokenState(state)
        self.__updateQTable(tokenState, np.array([0, 1, 0, 0]))
        return random.choice(np.argwhere(next_states != False))
        pass
        #return # return number token want to move