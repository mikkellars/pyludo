import random
import csv
import os
import numpy as np
import pandas as pd
from pyludo.utils import token_vulnerability, is_stacked, is_globe_pos, is_on_opponent_globe
from pyludo.LudoGame import LudoState

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

    def __init__(self, chosenPolicy):
        self.__QTable = self.readQTable() # if not an existing file returning empty dictionary
        
        self.__chosenPolicy = chosenPolicy

        # Parameters
        self.__epsilon = 0.1
        self.__discount_factor = 1.0 
        self.__alpha = 0.6

    # STATE REPRESENTATION #
    def __getTokenState(self, state):
        #print("bob",state.state[0])
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

        #print(currentPlayerState) # State is all 4 players with current player as index = 0
        return currentPlayerState


    # MAKE Q-TABLE #

    def __updateQTable(self, tokenState, qValue):
        # Update dictionary
        strTokenState = str(tokenState)
        #print(self.__QTable[strTokenState])
        if (strTokenState in self.__QTable):
            tmpQValue = self.__QTable[strTokenState]
            tmpQValue.astype(float)
            self.__QTable[strTokenState] = np.add(qValue, tmpQValue)  
        # Make new entry
        else:
            self.__QTable[strTokenState] = qValue

    def saveQTable(self):
        np.save("QTable.npy",self.__QTable)
        print("QTable saved succefully")

    def readQTable(self):
        tmpQTable = dict()
        if os.path.isfile("QTable.npy"):
            tmpQTable = np.load('QTable.npy',allow_pickle=True).item()
            print("QTable read succefully with " + str(len(tmpQTable)) + " number of states")
        else:
            print ("QTable file not found, making a new")
        
        return tmpQTable

        

    def printQTable(self):
        print(self.__QTable)
      #  print("argmax", np.argmax(self.__QTable[str(np.array([0, 0, 0, 0]))]))
        pass


    # REWARD FUNCTION #
    def __calc_reward(self, next_token_state):
        reward = 0
        if (next_token_state == np.array([6, 6, 6, 6])).all():
            reward = LudoPlayerQLearning.win

        return reward

    # MAKE POLICIES #
    

    def __valid_actions(self, next_states):
        """
        Based all the next_states it finds valid actions (token that can move) and 
        sets 1 if valid, 0 if invalid and returns it. 
        """
        valid_actions = []
        for token_id, next_state in enumerate(next_states):
            if next_state is not False:
                valid_actions.append(1)
            else:
                valid_actions.append(0)
        return np.array(valid_actions)



    def policies(self, QTable, epsilon, state, next_states): # Inspiration from https://www.geeksforgeeks.org/q-learning-in-python/?fbclid=IwAR1UXR88IuJBhhTakjxNq_gcf3nCmJB0puuoA46J8mZnEan_qx9hhoFzhK8
        """ 
        Creates an epsilon-greedy policy based 
        on a given Q-function and epsilon. 
        
        Returns a function that takes the state 
        as an input and returns the probabilities 
        for each action in the form of a numpy array  
        of length of the action space(set of possible actions). 
        """
        num_actions = 4
        def epsilonGreedyPolicy(tokenState): 
            tmpTokenState = str(tokenState)
                
            valid_actions = self.__valid_actions(next_states)
            valid_act_len = len(np.where(valid_actions==True)[0])

            Action_probabilities = np.ones(num_actions, dtype = float) * epsilon / valid_act_len  # divides probability based on number of valid actions and epsilon (each 0.025 if 4 actions)       
            Action_probabilities *= valid_actions   

            # If same values in QTable choose random valid action 
            if( np.equal.reduce( QTable[tmpTokenState] == QTable[tmpTokenState]) ):
                best_action = random.choice(np.argwhere(valid_actions == True))
            else:
                best_action = np.argmax(QTable[tmpTokenState]) # Find index of action which gives highest QValue

            Action_probabilities[best_action] += (1.0 - epsilon) # Assigns rest probability to best action so probability sums to 1

            return Action_probabilities 

        def greedyPolicy(tokenState):
            tmpTokenState = str(tokenState)

            valid_actions = self.__valid_actions(next_states)

            Action_probabilities = np.zeros(num_actions, dtype = float)

            # If same values in QTable choose random valid action 
            if( np.equal.reduce( QTable[tmpTokenState] == QTable[tmpTokenState]) ):
                best_action = random.choice(np.argwhere(valid_actions == True))
            else:
                best_action = np.argmax(QTable[tmpTokenState]) # Find index of action which gives highest QValue

            Action_probabilities[best_action] += 1.0
            return Action_probabilities


        if(self.__chosenPolicy == "epsilon greedy"):
            return epsilonGreedyPolicy 
        if(self.__chosenPolicy == "greedy"):
            return greedyPolicy


    # Q LEARNING #
    def QLearning(self, state, next_states):  # Inspiration from https://www.geeksforgeeks.org/q-learning-in-python/?fbclid=IwAR1UXR88IuJBhhTakjxNq_gcf3nCmJB0puuoA46J8mZnEan_qx9hhoFzhK8
        
        # Convert statespace representation
        tokenState = self.__getTokenState(state)

        # Creates entry if current state does not exists
        self.__updateQTable(tokenState, np.array([0, 0, 0, 0]))

        # Get probabilites based on initialized policy (chosenPolicy)
        policy = self.policies(self.__QTable, self.__epsilon, state, next_states) # returns a policy function
        actionProbability = policy(tokenState)

        # Choose action based on the probability distribution
        action = np.random.choice( LudoPlayerQLearning.actions, p=actionProbability )

        # Find next state based on action and updates Q-table. 
        # DEFINE A FUNCTION WHICH GIVES REWARDS BASED ON THE NEXT STATE #
        
        next_state = next_states[action] # Current player = 0
        nextTokenState = self.__getTokenState(next_state)
        reward = self.__calc_reward(nextTokenState)

        # Creates entry if nextTokenState does not exists
        self.__updateQTable(nextTokenState, np.array([0, 0, 0, 0]))

        # Update based on TD Update
        # If same values in QTable choose random valid action 
        
        if( np.equal.reduce( self.__QTable[str(nextTokenState)] == self.__QTable[str(nextTokenState)]) ):
            best_next_action = np.random.choice(LudoPlayerQLearning.actions)
        else:
            best_next_action = np.argmax(self.__QTable[str(nextTokenState)])     

        #print(self.__QTable[str(nextTokenState)])
        #print("best_next_action", best_next_action)

        td_target = reward + self.__discount_factor * self.__QTable[str(nextTokenState)][best_next_action] 
        td_delta = td_target - self.__QTable[str(tokenState)][action] 
        update_val = self.__alpha * td_delta 
        Q_val_update = np.array([0, 0, 0, 0])
        np.add.at(Q_val_update, action, update_val)

        self.__updateQTable(tokenState, Q_val_update)

        # print("n",nextTokenState)
        #print("r",reward)

        return action






    def play(self, state, dice_roll, next_states):
        action = self.QLearning(state, next_states)

        return action # return number token want to move
