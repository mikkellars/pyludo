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
    r_moved_out = 1
    r_into_goal = 1
    r_send_opp_home = 1
    r_send_self_home = -1
    r_move_token = 0.5

    def __init__(self, chosenPolicy, RewardName, epsilon, discount_factor, learning_rate):
        self.__chosenPolicy = chosenPolicy
        
        # Parameters
        self.__epsilon = epsilon
        self.__discount_factor = discount_factor
        self.__alpha = learning_rate

        # Data logging rewards
        self.Reward_save_name = RewardName
        self.total_reward = 0.0
        self.rewards = [] 

        # Core rewards
        self.__rewards = np.array([LudoPlayerQLearningSimple.r_moved_out, LudoPlayerQLearningSimple.r_into_goal, LudoPlayerQLearningSimple.r_send_opp_home, LudoPlayerQLearningSimple.r_send_self_home, LudoPlayerQLearningSimple.r_move_token])
        self.__pre_reward = 0

        self.__val_tok_mov = np.empty((4, 4)) # containts which token can make an valid action
        self.__reduced_state = np.empty(4) # contains valid actions e.g. move out of spawm
        self.__next_state_based_action = None
        self.__prev_state_idx = None
        self.__prev_tok_to_mov = None

        self.__QTable = np.empty((2**4, 5)) # State space is 2^4 = 16 with 5 actions-value, [moved_out, into_goal, send_opp_home, send_self_home, move_token] 
        
    def saveReward(self):
        with open(self.Reward_save_name, 'a', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for reward in self.rewards:
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
            return False

        current_pos_token = state[0][token_id]
        next_pos_token = next_state[0][token_id]

        current_opponent_states = state[1:]
        next_opponent_states = next_state[1:]

        moved_out = (current_pos_token == -1) and (next_pos_token == 1)
        into_goal = (next_pos_token == 99)
        send_opp_home = self.__will_send_opponent_home(np.array(current_opponent_states), np.array(next_opponent_states))
        send_self_home = (next_pos_token == -1)

        reduced_state = [moved_out, into_goal, send_opp_home, send_self_home] # True if action is valid

        return reduced_state

    def __get_token_state(self, state, next_states):
        """
        Converts whole state representation of one player,
        to a simpler one. State is a representation of what
        the player can do. E.g. move_out 
        """
        val_token_mov = np.empty((4, 4))
        for token_id in range(4):
            val_token_mov[token_id] = self.__valid_token_moves(state, next_states[token_id], token_id)

       # print(val_token_mov)
        self.__val_tok_mov = val_token_mov
        self.__reduced_state = np.logical_or.reduce((val_token_mov[0,:],val_token_mov[1,:],val_token_mov[2,:],val_token_mov[3,:]))

        return self.__reduced_state

    def __action_to_token(self, action, next_states):
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

        valid_tokens = np.argwhere(self.__val_tok_mov[:,int(action)] == True).squeeze()
        if valid_tokens.size > 1:
            return np.random.choice(valid_tokens)
        else:
            return valid_tokens

    def policies(self, QTable, epsilon, state_idx): # Inspiration from https://www.geeksforgeeks.org/q-learning-in-python/?fbclid=IwAR1UXR88IuJBhhTakjxNq_gcf3nCmJB0puuoA46J8mZnEan_qx9hhoFzhK8
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
            
            
            valid_actions = np.append(self.__reduced_state, True) # the True appended is move_token
            valid_act_len = len(np.where(valid_actions==True)[0])

            Action_probabilities = np.ones(num_actions, dtype = float) * epsilon / valid_act_len  # divides probability based on number of valid actions and epsilon (each 0.025 if 4 actions)       
            Action_probabilities = np.multiply(Action_probabilities, valid_actions)

            # If same values in QTable choose random valid action 
            best_action = np.argmax(QTable[state_idx]) # Find index of action which gives highest QValue
            i = 3
            while not valid_actions[best_action]:
                best_action = np.argsort(QTable[state_idx])[i]
                i -= 1

            Action_probabilities[best_action] += (1.0 - epsilon) # Assigns rest probability to best action so probability sums to 1

            return Action_probabilities 

    #     def greedyPolicy(tokenState):
    #         tmpTokenState = str(tokenState)

    #         valid_actions = self.__valid_actions(next_states)

    #         Action_probabilities = np.zeros(num_actions, dtype = float)

    #         best_action = np.argmax(QTable[tmpTokenState]) # Find index of action which gives highest QValue

    #         Action_probabilities[best_action] += 1.0
    #         return Action_probabilities


        if(self.__chosenPolicy == "epsilon greedy"):
            return epsilonGreedyPolicy 
    #     if(self.__chosenPolicy == "greedy"):
    #         return greedyPolicy


    # Q LEARNING #
    def QLearning(self, state, next_states):  # Inspiration from https://www.geeksforgeeks.org/q-learning-in-python/?fbclid=IwAR1UXR88IuJBhhTakjxNq_gcf3nCmJB0puuoA46J8mZnEan_qx9hhoFzhK8
        # Convert statespace representation
        self.__get_token_state(state, next_states)

        state_idx = self.__from_np_arr_2_binary(self.__reduced_state)

        # Get probabilites based on initialized policy (chosenPolicy)
        policy = self.policies(self.__QTable, self.__epsilon, state_idx) # returns a policy function
        actionProbability = policy()

        # Choose action based on the probability distribution
        action = np.random.choice( LudoPlayerQLearningSimple.actions, p=actionProbability )
        
        # Gives reward
        reward = self.__rewards[int(action)]
        self.total_reward += reward

        # Update based on TD Update
        # Because of this reduced state representation can first update state after next round. 
        # Find next state based on action. 
        token_to_move = self.__action_to_token(action, next_states)
        if self.__next_state_based_action is None:
            self.__prev_state_idx = state_idx
            self.__prev_tok_to_mov = token_to_move
            self.__pre_reward = reward
            self.__next_state_based_action = next_states[token_to_move][0] 
        elif self.__next_state_based_action == state.state[0]:
            next_next_state = self.__get_token_state(state, next_states)
            state_idx = self.__from_np_arr_2_binary(next_next_state)
            td_target = self.__pre_reward + self.__discount_factor * np.max(self.__QTable[state_idx])
            td_delta = td_target - self.__QTable[self.__prev_state_idx][self.__prev_tok_to_mov]
            update_val = self.__alpha * td_delta 
            self.__QTable[self.__prev_state_idx][self.__prev_tok_to_mov] += update_val
            # Updates to use for next_next_state
            self.__state_idx = state_idx
            self.__prev_tok_to_mov = token_to_move
            self.__pre_reward = reward
            self.__next_state_based_action = next_states[token_to_move][0] 

        return token_to_move

    def play(self, state, dice_roll, next_states):
        action = self.QLearning(state, next_states)
        return action # return number token want to move