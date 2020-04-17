import random
import csv
import os
import numpy as np
import pandas as pd
from perf.ludo.utils import token_vulnerability, is_stacked, is_globe_pos, is_on_opponent_globe
from perf.ludo.LudoGame import LudoState

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
    safe = 10
    got_vulnerable = -5
    knock_home = 10
    suicide = -10

    one_token_win = 20
    win = 20    
    
    # Actions
    actions = [0, 1, 2, 3] # chooses the token number

    def __init__(self, chosenPolicy, QtableName, RewardName, epsilon, discount_factor, learning_rate):
        self.__chosenPolicy = chosenPolicy
        
        # Parameters
        self.__epsilon = epsilon
        self.__discount_factor = discount_factor
        self.__alpha = learning_rate

        # Save and Read parameters
        self.Qtable_save_name = QtableName + f'_e-{epsilon}_d-{discount_factor}_a-{learning_rate}.csv'
        self.Reward_save_name = RewardName + f'_e-{epsilon}_d-{discount_factor}_a-{learning_rate}.csv'

        self.__QTable = self.readQTable() # if not an existing file returning empty dictionary
        self.total_reward = 0.0


    # STATE REPRESENTATION #
    def __getTokenState(self, state, player_num):
        #print("bob",state.state[0])
        tokenState = np.copy(state.state[player_num])

        # Set all to normal if no criteria is true
        tokenState[:] = LudoPlayerQLearning.normal

        # Homed
        tokenState[state.state[player_num] == -1] = LudoPlayerQLearning.homed
        
        # Vulnerable
        tmp = [token_vulnerability(state, token_id, player_num) > 0 for token_id in range(4)]
        tokenState[tmp] = LudoPlayerQLearning.vulnerable

        # Stacked
        tokenState[is_stacked(state.state, player_num)] = LudoPlayerQLearning.stacked

        # On globe
        tmp = [is_globe_pos(token) for token in state.state[player_num]]
        tokenState[tmp] = LudoPlayerQLearning.globe

        # On other players home globe
        tokenState[is_on_opponent_globe(state.state, player_num)] = LudoPlayerQLearning.globe_home

        # Token inside goal
        tokenState[state.state[player_num] == 99] = LudoPlayerQLearning.goal

        #print(currentPlayerState) # State is all 4 players with current player as index = 0
        return tokenState


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
        csv_writer = csv.writer(open(self.Qtable_save_name, "w", newline=''))
        for key, val in self.__QTable.items():
            csv_writer.writerow((key, val))
        print("QTable saved succefully")

    def readQTable(self):

        tmpQTable = dict()
        if os.path.isfile(self.Qtable_save_name):
            read = csv.reader(open(self.Qtable_save_name))
            i = 0
            for row in read:
                i = i + 1
                state, QVal = row
                QVal = np.fromstring(QVal[1:-1], sep=' ')
                tmpQTable[state] = QVal
            print("QTable read succefully. Found " + str(i) + " states")
        else:
            print ("QTable file not found, making a new")
        
        return tmpQTable

    def saveReward(self):
        with open(self.Reward_save_name, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([(self.total_reward)])
        # Resets total_reward for next game
        self.total_reward = 0

    def printQTable(self):
        print(self.__QTable)
      #  print("argmax", np.argmax(self.__QTable[str(np.array([0, 0, 0, 0]))]))
        pass


    # REWARD FUNCTION #
    def __get_diff_next_states(self, tokenStatePlayers, nextTokenStatePlayers):
        diff_next_states = []
        for player_num, player_state in enumerate(tokenStatePlayers):
            tokenState = player_state
            nextTokenState = nextTokenStatePlayers[player_num]
            diff = tokenState != nextTokenState
            diff_next_states.append(nextTokenState[diff]) # Is empty if there are no diff between state and nextstate

        return diff_next_states

    def __did_knock_home(self, diff_next_states):
        did_knock_home = False
        # Delete current players entry
        diff_next_states.remove(diff_next_states[0])
        for diff_next_state in diff_next_states:
            if len(diff_next_state[diff_next_state == LudoPlayerQLearning.homed]) >= 1:
                did_knock_home = True

        return did_knock_home

    def __calc_reward(self, state, next_states_based_action):

        reward = 0

        tokenStatePlayers = [self.__getTokenState(state, player_id) for player_id in range(0,4)]
        nextTokenStatePlayers = [self.__getTokenState(next_states_based_action, player_id) for player_id in range(0,4)]

        diff_next_states = self.__get_diff_next_states(tokenStatePlayers, nextTokenStatePlayers)
        diff_next_state = diff_next_states[0] # current player

        # Can sometimes be more than one elemtents, but only when transitioning to stacked state or normal or both or hitting more six in a row. Thus setting to size 1
        if (len(diff_next_state) != 0):
            if (np.equal.reduce(diff_next_state == LudoPlayerQLearning.stacked)): # more than one stacked
                diff_next_state = np.array([LudoPlayerQLearning.stacked])
            elif( np.equal.reduce(diff_next_state == LudoPlayerQLearning.normal)): # more than one normal
                diff_next_state = np.array([LudoPlayerQLearning.normal])
            elif( len(diff_next_state[diff_next_state == LudoPlayerQLearning.stacked]) >= 1): # one normal and one stacked
                diff_next_state = np.array([LudoPlayerQLearning.stacked])
            elif( len(diff_next_state[diff_next_state == LudoPlayerQLearning.globe]) >= 1): # one/two normal and one globe
                diff_next_state = np.array([LudoPlayerQLearning.globe])
            elif( len(diff_next_state[diff_next_state == LudoPlayerQLearning.vulnerable]) >= 1): # one/two normal and one globe (hitting six more than one time)
                diff_next_state = np.array([LudoPlayerQLearning.vulnerable])
            elif( len(diff_next_state[diff_next_state == LudoPlayerQLearning.globe_home]) >= 1): # one/two normal and one globe home (hitting six more than one time)
                diff_next_state = np.array([LudoPlayerQLearning.globe_home])
            elif( len(diff_next_state[diff_next_state == LudoPlayerQLearning.goal]) >= 1): # one/two normal and one goal (hitting six more than one time)
                diff_next_state = np.array([LudoPlayerQLearning.goal])


        if (nextTokenStatePlayers[0] == np.array([LudoPlayerQLearning.goal, LudoPlayerQLearning.goal, LudoPlayerQLearning.goal, LudoPlayerQLearning.goal])).all():
            reward = LudoPlayerQLearning.win
        elif(diff_next_state == LudoPlayerQLearning.goal):
            reward = LudoPlayerQLearning.one_token_win
        elif(diff_next_state == LudoPlayerQLearning.homed):
            reward = LudoPlayerQLearning.suicide
        elif(diff_next_state == LudoPlayerQLearning.globe or diff_next_state == LudoPlayerQLearning.stacked):
            reward = LudoPlayerQLearning.safe
        elif(diff_next_state == LudoPlayerQLearning.vulnerable or diff_next_state == LudoPlayerQLearning.globe_home):
            reward = LudoPlayerQLearning.got_vulnerable

        # # Checking for knocking home an opponent player
        if self.__did_knock_home(diff_next_states):
            reward += LudoPlayerQLearning.knock_home

        return reward

    # MAKE POLICIES #
    

    def __valid_actions(self, next_states):
        """
        Based on all the next_states it finds valid actions (token that can move) and 
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
            Action_probabilities = np.multiply(Action_probabilities, valid_actions)

            # If same values in QTable choose random valid action 
            best_action = np.argmax(QTable[tmpTokenState]) # Find index of action which gives highest QValue
            i = 3
            while not valid_actions[best_action]:
                best_action = np.argsort(QTable[tmpTokenState])[i]
                i -= 1

            Action_probabilities[best_action] += (1.0 - epsilon) # Assigns rest probability to best action so probability sums to 1

            return Action_probabilities 

        def greedyPolicy(tokenState):
            tmpTokenState = str(tokenState)

            valid_actions = self.__valid_actions(next_states)

            Action_probabilities = np.zeros(num_actions, dtype = float)

           
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
        tokenState = self.__getTokenState(state, 0)
        
        # Creates entry if current state does not exists
        self.__updateQTable(tokenState, np.array([0.0, 0.0, 0.0, 0.0]))

        # Get probabilites based on initialized policy (chosenPolicy)
        policy = self.policies(self.__QTable, self.__epsilon, state, next_states) # returns a policy function
        actionProbability = policy(tokenState)

        # Choose action based on the probability distribution
        action = np.random.choice( LudoPlayerQLearning.actions, p=actionProbability )

        # Find next state based on action and updates Q-table. 
        next_states_based_action = next_states[action] 
        nextTokenState = self.__getTokenState(next_states_based_action, 0) # getTokenState handles that it is state[0] that is current player

        reward = self.__calc_reward(state, next_states_based_action)
        self.total_reward += reward

        # Creates entry if nextTokenState does not exists
        self.__updateQTable(nextTokenState, np.array([0.0, 0.0, 0.0, 0.0]))

        # Update based on TD Update
        td_target = reward + self.__discount_factor * np.max(self.__QTable[str(nextTokenState)])
        td_delta = td_target - self.__QTable[str(tokenState)][action] 
        update_val = self.__alpha * td_delta 

        self.__QTable[str(tokenState)][action] += update_val

        # print("n",nextTokenState)
        #print("r",reward)

        return action

    def play(self, state, dice_roll, next_states):
        action = self.QLearning(state, next_states)

        return action # return number token want to move
