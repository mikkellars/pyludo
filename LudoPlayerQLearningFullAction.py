import random
import csv
import os
import numpy as np
import pandas as pd
from perf.pyludo.utils import token_vulnerability, is_stacked, is_globe_pos, is_on_opponent_globe, star_jump
from perf.pyludo.LudoGame import LudoState
import time
import cProfile


class LudoPlayerQLearningFullAction:
    ####### Class variables ########
   
    # Actions
    actions = np.array([0,1,2,3,4]) # 5 actions, [moved_out, into_goal, send_opp_home, send_self_home, move_token] 

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

    def __will_send_opponent_home(self, opponent_states, opponent_next_states):
        """
        Function for evaluating if possible
        to knock opponent home
        """
        for player_idx, opponent_next_state in enumerate(opponent_next_states):
            if opponent_next_state is False:
                continue
            opponent_state = opponent_states[player_idx]
            if np.sum( opponent_state[:] == -1 ) < np.sum( opponent_next_state[:] == -1 ):
                return True
        return False


    def __valid_token_moves(self, state, next_state, token_id):
        """
        Finds valid moves for a token and uses it as actions
        """
        if next_state == False:
            return [False, False, False, False]

        current_pos_token = state.state[0][token_id]
        next_pos_token = next_state.state[0][token_id]

        current_opponent_states = state.state[1:]
        next_opponent_states = next_state.state[1:]

        moved_out = (current_pos_token == -1) and (next_pos_token != -1)
        into_goal = (current_pos_token != 99) and (next_pos_token == 99)
        send_opp_home = self.__will_send_opponent_home(np.array(current_opponent_states), np.array(next_opponent_states))
        send_self_home = (current_pos_token != -1) and (next_pos_token == -1)
       

        token_actions = [moved_out, into_goal, send_opp_home, send_self_home] # True if action is valid

        return token_actions

    def __get_actions(self, state, next_states):
        """
        Converts whole state representation of one player,
        to a simpler one. State is a representation of what
        the player can do. E.g. move_out 
        """
        val_tok_mov = np.zeros((4, 4))
        for token_id in range(4):
            val_tok_mov[token_id] = self.__valid_token_moves(state, next_states[token_id], token_id)

        actions = np.logical_or.reduce((val_tok_mov[0,:], val_tok_mov[1,:], val_tok_mov[2,:], val_tok_mov[3,:]))

        return actions, val_tok_mov

    def __action_to_token(self, action, next_states, val_tok_mov):
        """
        Maps from combined action chosen based on
        the minimal state representation to which token
        to move. Returns empty if no token can do action
        If move than one token can do the same move, 
        return a random token.
        """
        if (int)(action) == 4: #  move_token action returns first valid token to move
            for token_id, next_state in enumerate(next_states):
                if next_state is not False:
                    return token_id

        valid_tokens = np.argwhere(val_tok_mov[:,int(action)] == True).squeeze()
        if valid_tokens.size > 1:
            return np.random.choice(valid_tokens)
        else:
            return valid_tokens


    def append_reward(self):
        self.rewards.append(self.total_reward)
        self.total_reward = 0

    # MAKE Q-TABLE #

    def __updateQTable(self, state, qValue):
        # Update dictionary
        tmp_state = str(state)
        if (tmp_state in self.__QTable):
            tmpQValue = self.__QTable[tmp_state]
            self.__QTable[tmp_state] = np.add(qValue, tmpQValue)  
        # Make new entry
        else:
            self.__QTable[tmp_state] = qValue

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
                tmpQTable[state] = np.array(QVal)
            print("QTable read succefully. Found " + str(i) + " states")
        else:
            print ("QTable file not found, making a new")
        
        return tmpQTable

    def saveReward(self):
        with open(self.Reward_save_name, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for reward in self.rewards:
                csv_writer.writerow([reward])

    def printQTable(self):
        print(self.__QTable)
      #  print("argmax", np.argmax(self.__QTable[str(np.array([0, 0, 0, 0]))]))
        pass

    def policies(self, QTable, epsilon, state, next_states, action_to_do): # Inspiration from https://www.geeksforgeeks.org/q-learning-in-python/?fbclid=IwAR1UXR88IuJBhhTakjxNq_gcf3nCmJB0puuoA46J8mZnEan_qx9hhoFzhK8
        """ 
        Creates an epsilon-greedy policy based 
        on a given Q-function and epsilon. 
        
        Returns a function that takes the state 
        as an input and returns the probabilities 
        for each action in the form of a numpy array  
        of length of the action space(set of possible actions). 
        """
        num_actions = 5 # 5 actions-value, [moved_out, into_goal, send_opp_home, send_self_home, move_token] 
        def epsilonGreedyPolicy(): 
            tmp_state = str(state.state[0])
            valid_actions = np.append(action_to_do, True) # the True appended is move_token
            valid_act_len = len(np.where(valid_actions==True)[0])

            Action_probabilities = np.ones(num_actions, dtype = float) * epsilon / valid_act_len  # divides probability based on number of valid actions and epsilon (each 0.025 if 4 actions)       
            Action_probabilities = np.multiply(Action_probabilities, valid_actions)

            # If same values in QTable choose random valid action 
            best_action = np.argmax(QTable[tmp_state]) # Find index of action which gives highest QValue
            # Check if valid action else find new best action
            if not valid_actions[best_action]:
                actions = np.argsort(-QTable[tmp_state]) # descending order of action values
                for i in range(len(valid_actions)):
                    if valid_actions[actions[i]]:
                        best_action = actions[i]
                        break

            Action_probabilities[best_action] += (1.0 - epsilon) # Assigns rest probability to best action so probability sums to 1

            return Action_probabilities 

        def greedyPolicy():
            tmp_state = str(state.state[0])
            valid_actions = np.append(action_to_do, True) # the True appended is move_token

            Action_probabilities = np.zeros(num_actions, dtype = float)

            best_action = np.argmax(QTable[tmp_state]) # Find index of action which gives highest QValue
            # Check if valid action else find new best action
            if not valid_actions[best_action]:
                actions = np.argsort(-QTable[tmp_state]) # descending order of action values
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
     
        # Creates entry if current state does not exists
        self.__updateQTable(state.state[0], np.array([0.0, 0.0, 0.0, 0.0, 0.0]))
        actions, val_token_moves = self.__get_actions(state, next_states)

        # Get probabilites based on initialized policy (chosenPolicy)
        policy = self.policies(self.__QTable, self.__epsilon, state, next_states, actions) # returns a policy function
        actionProbability = policy()

        # Choose action based on the probability distribution
        action = np.random.choice( LudoPlayerQLearningFullAction.actions, p=actionProbability )
        token_to_move = self.__action_to_token(action, next_states, val_token_moves)

        # Find next state based on action and updates Q-table. 
        next_states_based_action = str(next_states[token_to_move].state[0]) 

        # Cummulative reward
        reward = self.__calc_cum_reward(token_to_move, next_states)
        # reward = self.__calc_cum_mean_reward(token_to_move, next_states)
        self.total_reward += reward

        # win only reward
        # reward = self.__win_only_reward(token_to_move, next_states)
        # self.total_reward += reward

        # Creates entry if nextTokenState does not exists
        self.__updateQTable(next_states_based_action, np.array([0.0, 0.0, 0.0, 0.0, 0.0]))

        # Update based on TD Update
        td_target = reward + self.__discount_factor * np.max(self.__QTable[str(next_states_based_action)])
        td_delta = td_target - self.__QTable[str(state.state[0])][token_to_move] 
        update_val = self.__alpha * td_delta 

        self.__QTable[str(state.state[0])][token_to_move] += update_val

        return token_to_move

    def play(self, state, dice_roll, next_states):

       # cProfile.runctx('self.QLearning(state, next_states)', globals(), locals()) # Time profile for functions

        action = self.QLearning(state, next_states)

        return action # return number token want to move
