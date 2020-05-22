import random
import csv
import os
import numpy as np
import pandas as pd
from perf.pyludo.utils import token_vulnerability, is_stacked, is_globe_pos, is_on_opponent_globe, star_jump
from perf.pyludo.LudoGame import LudoState
import time
import cProfile


class LudoPlayerQLearningToken:
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
    star = 7 # when on a star 
    end_game_area = 8 # Last 5 spaces into goal

    # Rewards
    r_normal = 0
    r_safe = 0
    r_got_vulnerable = -1
    r_knock_home = 1
    r_suicide = 0
    r_star_jump = 0
    r_move_onto_board = 1

    r_moved_end_game_token = 0
    r_end_game_safe = 1
    r_one_token_win = 1
    r_win = 1
    
    # Actions
    actions = [0, 1, 2, 3] # chooses the token number
    name = 'Q-learning Token'
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
        self.rewards = []


    # STATE REPRESENTATION #
    def __getTokenState(self, state, player_num):
        """
        Converts whole state representation of one player,
        to a simpler one
        """
        playerState = np.array(state.state[player_num])
        tokenState = np.copy(playerState)

        # Set all to normal if no criteria is true
        tokenState[:] = LudoPlayerQLearningToken.normal

        # Homed
        tokenState[playerState == -1] = LudoPlayerQLearningToken.homed
        
        # Vulnerable
        tmp = [token_vulnerability(state, token_id, player_num) > 0 for token_id in range(4)]
        tokenState[tmp] = LudoPlayerQLearningToken.vulnerable

        # Stacked
        tmp = is_stacked(np.array(state.state), player_num)
        tokenState[tmp] = LudoPlayerQLearningToken.stacked

        # On globe
        tmp = [is_globe_pos(token) for token in playerState]
        tokenState[tmp] = LudoPlayerQLearningToken.globe

        # On star
        tmp = [star_jump(token) > 0 for token in playerState]
        tokenState[tmp] = LudoPlayerQLearningToken.star

        # On other players home globe
        tmp = is_on_opponent_globe(np.array(state.state), player_num)
        tokenState[tmp] = LudoPlayerQLearningToken.globe_home

        # Token end game last 5 space into goal
        tokenState[playerState >= 52] = LudoPlayerQLearningToken.end_game_area

        # Token inside goal
        tokenState[playerState == 99] = LudoPlayerQLearningToken.goal

        return tokenState

    
    def __calc_cum_reward(self, token_to_move, next_states):
        min_val = -1*4
        max_val = 99*4

        next_state_sum = np.sum(next_states[token_to_move][0])
        
        # Get the opponent who is most ahead by finding sum of the state
        opponents_next_state_sum = np.sum(next_states[token_to_move][1:])
        oppenent_ahead = np.max(opponents_next_state_sum)

        diff_state_sum = next_state_sum - oppenent_ahead

        return (diff_state_sum - min_val)/(max_val - min_val)

    def __calc_cum_mean_reward(self, token_to_move, next_states):
        """
        Calculates normalized cumulative reward based on
        all the opponents tokens meaned
        """
        min_val = -1
        max_val = 99

        next_state_sum = np.mean(np.sum(next_states[token_to_move][0]))
        
        # Get the opponent who is most ahead by finding sum of the state
        opponents_next_state_sum = np.sum(next_states[token_to_move][1:])
        opponents_mean = np.mean(np.mean(opponents_next_state_sum))

        diff_state_sum = next_state_sum - opponents_mean

        return (diff_state_sum - min_val)/(max_val - min_val)

    def __win_only_reward(self, token_to_move, next_states):
        next_state = next_states[token_to_move].state[0]
        if np.all(np.array(next_state) == 99):
            return 1
        else:
            return 0

    def append_reward(self):
        self.rewards.append(self.total_reward)
        self.total_reward = 0

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
        if self.__chosenPolicy is not 'greedy':
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
                tmpQTable[state] = np.array(QVal)
            print("QTable read succefully. Found " + str(i) + " states")
        else:
            print ("QTable file not found, making a new")
        
        return tmpQTable

    def saveReward(self):
        if self.__chosenPolicy is not 'greedy':
            with open(self.Reward_save_name, 'a', newline='') as csv_file:
                csv_writer = csv.writer(csv_file)
                for reward in self.rewards:
                    csv_writer.writerow([reward])

    def printQTable(self):
        print(self.__QTable)
      #  print("argmax", np.argmax(self.__QTable[str(np.array([0, 0, 0, 0]))]))
        pass

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
            if not valid_actions[best_action]:
                actions = np.argsort(-QTable[tmpTokenState]) # descending order of action values
                for i in range(len(valid_actions)):
                    if valid_actions[actions[i]]:
                        best_action = actions[i]
                        break

            Action_probabilities[best_action] += (1.0 - epsilon) # Assigns rest probability to best action so probability sums to 1

            return Action_probabilities 

        def greedyPolicy(tokenState):
            tmpTokenState = str(tokenState)

            valid_actions = self.__valid_actions(next_states)

            Action_probabilities = np.zeros(num_actions, dtype = float)

            best_action = np.argmax(QTable[tmpTokenState]) # Find index of action which gives highest QValue
            if not valid_actions[best_action]:
                actions = np.argsort(-QTable[tmpTokenState]) # descending order of action values
                for i in range(len(valid_actions)):
                    if valid_actions[actions[i]]:
                        best_action = actions[i]
                        break


            Action_probabilities[best_action] += 1.0
            return Action_probabilities


        if(self.__chosenPolicy == "epsilon greedy"):
            return epsilonGreedyPolicy 
        if(self.__chosenPolicy == "greedy"):
            return greedyPolicy


    # Q LEARNING #
    def QLearning(self, state, next_states):  # Inspiration from https://www.geeksforgeeks.org/q-learning-in-python/?fbclid=IwAR1UXR88IuJBhhTakjxNq_gcf3nCmJB0puuoA46J8mZnEan_qx9hhoFzhK8
        # Convert statespace representation for current states
        tokenStates = self.__getTokenState(state, 0)

        # Creates entry if current state does not exists
        self.__updateQTable(tokenStates, np.array([0.0, 0.0, 0.0, 0.0]))

        # Get probabilites based on initialized policy (chosenPolicy)
        policy = self.policies(self.__QTable, self.__epsilon, state, next_states) # returns a policy function
        actionProbability = policy(tokenStates)

        # Choose action based on the probability distribution
        action = np.random.choice( LudoPlayerQLearningToken.actions, p=actionProbability )

        # Find next state based on action and updates Q-table. 
        next_states_based_action = next_states[action] 
        nextTokenStates = self.__getTokenState(next_states_based_action, 0) 

        # reward = self.__win_only_reward(action, next_states)
        # self.total_reward += reward

        # Cummulative reward
        reward = self.__calc_cum_reward(action, next_states)
        #reward = self.__calc_cum_mean_reward(action, next_states)
        self.total_reward += reward


        if self.__chosenPolicy is not 'greedy':
            # Creates entry if nextTokenState does not exists
            self.__updateQTable(nextTokenStates, np.array([0.0, 0.0, 0.0, 0.0]))

            # Update based on TD Update
            td_target = reward + self.__discount_factor * np.max(self.__QTable[str(nextTokenStates)])
            td_delta = td_target - self.__QTable[str(tokenStates)][action] 
            update_val = self.__alpha * td_delta 

            self.__QTable[str(tokenStates)][action] += update_val

        return action

    def play(self, state, dice_roll, next_states):

       # cProfile.runctx('self.QLearning(state, next_states)', globals(), locals()) # Time profile for functions

        action = self.QLearning(state, next_states)

        return action # return number token want to move
