import random
import csv
import os
import numpy as np
import pandas as pd
from perf.pyludo.utils import token_vulnerability, is_stacked, is_globe_pos, is_on_opponent_globe, star_jump
from perf.pyludo.LudoGame import LudoState
import time
import cProfile


class LudoPlayerQLearningSimple:

    actions = np.array([0,1,2,3,4]) # 5 actions, [moved_out, into_goal, send_opp_home, send_self_home, move_token] 

    # Rewards   
    r_moved_out = 1.49247334 #1
    r_into_goal = 4.88517378 #1
    r_send_opp_home = 0.45690021 #1
    r_send_self_home = -5.2411656 # -1
    r_move_token = 0.5

    def __init__(self, Parameters, chosenPolicy="epsilon greedy", QtableName=None, RewardName=None):
        self.__chosenPolicy = chosenPolicy
        
        # Only used for GA
        self.chromosome = Parameters

        # Parameters
        self.__epsilon = Parameters[0]
        self.__discount_factor = Parameters[1]
        self.__alpha = Parameters[2]

        # Data logging rewards and QTable
        if RewardName is not None:
            self.Reward_save_name = RewardName + f'_e-{self.__epsilon}_d-{self.__discount_factor}_a-{self.__alpha}.csv'
        if QtableName is not None:
            self.Qtable_save_name = QtableName + f'_e-{self.__epsilon}_d-{self.__discount_factor}_a-{self.__alpha}.csv'

        self.__total_reward = 0.0
        self.__all_rewards = [] 

        if(QtableName is None):
            self.__QTable = np.zeros((2**4, 5)) # State space is 2^4 = 16 with 5 actions-value, [moved_out, into_goal, send_opp_home, send_self_home, move_token] 
        else:
            self.__QTable = self.readQTable()#np.zeros((2**4, 5)) # State space is 2^4 = 16 with 5 actions-value, [moved_out, into_goal, send_opp_home, send_self_home, move_token] 

        # Core rewards
        self.__rewards = np.array([LudoPlayerQLearningSimple.r_moved_out, LudoPlayerQLearningSimple.r_into_goal, LudoPlayerQLearningSimple.r_send_opp_home, LudoPlayerQLearningSimple.r_send_self_home, LudoPlayerQLearningSimple.r_move_token])

        ##### TESTING PARAMETERS #####
        self.iterations = 0
            
    def append_reward(self):
        self.__all_rewards.append(self.__total_reward)
        self.__total_reward = 0.0
        
    def saveQTable(self):
        csv_writer = csv.writer(open(self.Qtable_save_name, "w", newline=''))
       
        for state, qval in enumerate(self.__QTable):
            state = bin(state)
            csv_writer.writerow((state, qval))
        print("QTable saved succefully")

    def readQTable(self):
        tmpQTable = np.zeros((2**4, 5))
        if os.path.isfile(self.Qtable_save_name):
            read = csv.reader(open(self.Qtable_save_name))
            i = 0
            for state, row in enumerate(read):
                i = i + 1
                state, QVal = row
                QVal = np.fromstring(QVal[1:-1], sep=' ')
                state = int(state,2)
                tmpQTable[state] = np.array(QVal)
            print("QTable read succefully. Found " + str(i) + " states")
        else:
            print ("QTable file not found, making a new")
        
        return tmpQTable

    def saveReward(self):
        with open(self.Reward_save_name, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for reward in self.__all_rewards:
                csv_writer.writerow([reward])

    def __from_np_arr_2_binary(self, np_arr):
        np_arr = np_arr.astype(int)
        np_arr = np.array2string(np_arr)
        np_arr = np_arr.replace(' ','')
        np_arr = np_arr.replace('[','')
        np_arr = np_arr.replace(']','')

        return int(np_arr,2)


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
        Finds valid moves for a token
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
       

        reduced_state = [moved_out, into_goal, send_opp_home, send_self_home] # True if action is valid

        return reduced_state

    def __get_token_state(self, state, next_states):
        """
        Converts whole state representation of one player,
        to a simpler one. State is a representation of what
        the player can do. E.g. move_out 
        """
        val_tok_mov = np.zeros((4, 4))
        for token_id in range(4):
            val_tok_mov[token_id] = self.__valid_token_moves(state, next_states[token_id], token_id)

        reduced_state = np.logical_or.reduce((val_tok_mov[0,:], val_tok_mov[1,:], val_tok_mov[2,:], val_tok_mov[3,:]))

        return reduced_state, val_tok_mov

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

    def __calc_cum_reward(self, token_to_move, next_states):
        min_val = -1*4
        max_val = 99*4

        next_state_sum = np.sum(next_states[token_to_move][0])
        
        # Get the opponent who is most ahead by finding sum of the state
        opponents_next_state_sum = np.sum(next_states[token_to_move][1:])
        oppenent_ahead = np.max(opponents_next_state_sum)

        diff_state_sum = next_state_sum - oppenent_ahead

        return (diff_state_sum - min_val)/(max_val - min_val)


    def __win_only_reward(self, token_to_move, next_states):
        next_state = next_states[token_to_move].state[0]
        if np.all(np.array(next_state) == 99):
            return 1
        else:
            return 0


    def policies(self, QTable, epsilon, reduced_state, state_idx): # Inspiration from https://www.geeksforgeeks.org/q-learning-in-python/?fbclid=IwAR1UXR88IuJBhhTakjxNq_gcf3nCmJB0puuoA46J8mZnEan_qx9hhoFzhK8
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
            
            
            valid_actions = np.append(reduced_state, True) # the True appended is move_token
            valid_act_len = len(np.where(valid_actions==True)[0])

            Action_probabilities = np.ones(num_actions, dtype = float) * epsilon / valid_act_len  # divides probability based on number of valid actions and epsilon (each 0.025 if 4 actions)       
            Action_probabilities = np.multiply(Action_probabilities, valid_actions)

            # If same values in QTable choose random valid action 
            best_action = np.argmax(QTable[state_idx]) # Find index of action which gives highest QValue GANG VALID ACTION PÅ HER FOR AT FÅ DEN HØJEST VALID ACTION

            # Check if valid action else find new best action
            if not valid_actions[best_action]:
                actions = np.argsort(-QTable[state_idx]) # descending order of action values
                for i in range(len(valid_actions)):
                    if valid_actions[actions[i]]:
                        best_action = actions[i]
                        break
            
            Action_probabilities[best_action] += (1.0 - epsilon) # Assigns rest probability to best action so probability sums to 1

            return Action_probabilities 

        def greedyPolicy():
            valid_actions = np.append(reduced_state, True) # the True appended is move_token

            Action_probabilities = np.zeros(num_actions, dtype = float)

            best_action = np.argmax(QTable[state_idx]) # Find index of action which gives highest QValue GANG VALID ACTION PÅ HER FOR AT FÅ DEN HØJEST VALID ACTION

            # Check if valid action else find new best action
            if not valid_actions[best_action]:
                actions = np.argsort(-QTable[state_idx]) # descending order of action values
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
        
        if self.iterations > 0:       
            next_next_state, _ = self.__get_token_state(self.prev_next_state, next_states)
            next_state_idx = self.__from_np_arr_2_binary(next_next_state)

            td_target = self.reward + self.__discount_factor * np.max(self.__QTable[next_state_idx])
            td_delta = td_target - self.__QTable[self.state_idx][self.action]
            update_val = self.__alpha * td_delta 
            self.__QTable[self.state_idx][self.action] += update_val
           
       
        # Convert statespace representation
        reduced_state, val_token_mov = self.__get_token_state(state, next_states)

        self.state_idx = self.__from_np_arr_2_binary(reduced_state)

        # Get probabilites based on initialized policy (chosenPolicy)
        policy = self.policies(self.__QTable, self.__epsilon, reduced_state, self.state_idx) # returns a policy function
        actionProbability = policy()

        # Choose action based on the probability distribution
        self.action = np.random.choice( LudoPlayerQLearningSimple.actions, p=actionProbability )
        token_to_move = self.__action_to_token(self.action, next_states, val_token_mov)

        # Gives reward
        self.reward = self.__rewards[int(self.action)]
        self.__total_reward += self.reward

        # Testing cum reward
        # self.reward = self.__calc_cum_reward(token_to_move, next_states)
        # self.__total_reward += self.reward

        # Testing win only reward
        # self.reward = self.__win_only_reward(token_to_move, next_states)
        # self.__total_reward += self.reward

        # Update based on TD Update
        # Because of this reduced state representation can first update state after next round. 
        # Find next state based on action. 
        self.prev_next_state = next_states[token_to_move]
        self.iterations += 1


        return token_to_move

    #### TEST ###
    def reset_upd_val(self):
        self.ite_upd = 0
        self.ite = 0

    def play(self, state, dice_roll, next_states):
        token_to_move = self.QLearning(state, next_states)
        return token_to_move # return number token want to move