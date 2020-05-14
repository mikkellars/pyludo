import random
import csv
import os
import numpy as np
import pandas as pd
from perf.pyludo.utils import token_vulnerability, is_stacked, is_globe_pos, is_on_opponent_globe, star_jump
from perf.pyludo.LudoGame import LudoState
import time
import cProfile


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

    def __init__(self, chosenPolicy, QtableName, RewardName, epsilon, discount_factor, learning_rate):
        self.__chosenPolicy = chosenPolicy
        
        # Parameters
        self.__epsilon = epsilon
        self.__discount_factor = discount_factor
        self.__alpha = learning_rate

        # Rewards dictionary
        self.__reward_dict = self.__make_reward_dict()

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
        tokenState[:] = LudoPlayerQLearning.normal

        # Homed
        tokenState[playerState == -1] = LudoPlayerQLearning.homed
        
        # Vulnerable
        tmp = [token_vulnerability(state, token_id, player_num) > 0 for token_id in range(4)]
        tokenState[tmp] = LudoPlayerQLearning.vulnerable

        # Stacked
        tmp = is_stacked(np.array(state.state), player_num)
        tokenState[tmp] = LudoPlayerQLearning.stacked

        # On globe
        tmp = [is_globe_pos(token) for token in playerState]
        tokenState[tmp] = LudoPlayerQLearning.globe

        # On star
        tmp = [star_jump(token) > 0 for token in playerState]
        tokenState[tmp] = LudoPlayerQLearning.star

        # On other players home globe
        tmp = is_on_opponent_globe(np.array(state.state), player_num)
        tokenState[tmp] = LudoPlayerQLearning.globe_home

        # Token end game last 5 space into goal
        tokenState[playerState >= 52] = LudoPlayerQLearning.end_game_area

        # Token inside goal
        tokenState[playerState == 99] = LudoPlayerQLearning.goal

        return tokenState

    
    def __make_reward_dict(self):
        """
        Dictionary creation where first index
        is current state and second index is next state
        """

        reward_dict = dict()
        states = range(LudoPlayerQLearning.homed, LudoPlayerQLearning.end_game_area + 1)

        # MULTIPLE STATE AND NEXT STATE REWARD ASSIGNMENT
        for curState in states:
            for nextState in states:
                # One token goal
                if (curState is not LudoPlayerQLearning.goal) and (nextState is LudoPlayerQLearning.goal):
                    reward_dict[curState, nextState] = LudoPlayerQLearning.r_one_token_win
                # Token knocked home / suicide
                elif (curState is not LudoPlayerQLearning.homed) and (nextState is LudoPlayerQLearning.homed):
                    reward_dict[curState, nextState] = LudoPlayerQLearning.r_suicide
                # Token safe
                elif(curState is not LudoPlayerQLearning.stacked) and (nextState is LudoPlayerQLearning.stacked):
                    reward_dict[curState, nextState] = LudoPlayerQLearning.r_safe
                # Token safe
                elif (nextState is LudoPlayerQLearning.globe) and (curState is not LudoPlayerQLearning.homed):
                    reward_dict[curState, nextState] = LudoPlayerQLearning.r_safe
                # Token vulnerable
                elif (nextState is LudoPlayerQLearning.vulnerable) or (nextState is LudoPlayerQLearning.globe_home):
                    reward_dict[curState, nextState] = LudoPlayerQLearning.r_got_vulnerable
                # Moved a token into end game area
                elif (nextState is LudoPlayerQLearning.end_game_area) and (curState is not LudoPlayerQLearning.end_game_area):
                    reward_dict[curState, nextState] = LudoPlayerQLearning.r_end_game_safe
                # Moved to normal state
                elif (nextState is LudoPlayerQLearning.normal):
                    reward_dict[curState, nextState] = LudoPlayerQLearning.r_normal
                else:
                    reward_dict[curState, nextState] = 0


        # STATICS ONE STATE AND ONE NEXT STATE
        # Moved onto board
        reward_dict[LudoPlayerQLearning.homed, LudoPlayerQLearning.globe] = LudoPlayerQLearning.r_move_onto_board
        # Moved inside endgame area
        reward_dict[LudoPlayerQLearning.end_game_area, LudoPlayerQLearning.end_game_area] = LudoPlayerQLearning.r_moved_end_game_token

        return reward_dict

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

    def __changeTokenState(self, state, next_state, player_num, tokenStates, nextTokenStates):
        """
        Based on the current state and the next state taken, 
        returns the TokenState and nextTokenState for only
        the token which were moved. 
        """
        playerState = np.array(state.state[player_num])
        nextPlayerState = np.array(next_state.state[player_num])
        
        diff = playerState != nextPlayerState # token that moved

        # all can be false if in end game area and dice roll makes the player go in and out
        if np.all(diff==False):
            idx = np.where(np.logical_and(playerState >= 54, playerState<=57))
            diff[idx] = True
        
        movedTokenIdx = np.argmax(diff)

        return tokenStates[player_num][movedTokenIdx], nextTokenStates[player_num][movedTokenIdx]

    def __calc_reward(self, state, next_states_based_action, tokenStates, nextTokenStates):
        """
        Based on the current state and the next state taken, 
        calculates the reward. 
        """
        reward = 0
       
        # Win whole game
        if (nextTokenStates[0] == np.array([LudoPlayerQLearning.goal, LudoPlayerQLearning.goal, LudoPlayerQLearning.goal, LudoPlayerQLearning.goal])).all():
            return LudoPlayerQLearning.r_win

        # playerTokenTransition = [self.__changeTokenState(state, next_states_based_action, player_id, tokenStates, nextTokenStates) for player_id in range(0,4)]
        # curPlayerState, curPlayerNextState = playerTokenTransition[0]
        #print("r",self.__reward_dict[curPlayerState, curPlayerNextState])
        # reward = self.__reward_dict[curPlayerState, curPlayerNextState]
    

        #### OLD REWARD SYSTEM #####

        # tokenStatePlayers = [self.__getTokenState(state, player_id) for player_id in range(0,4)]
        # nextTokenStatePlayers = [self.__getTokenState(next_states_based_action, player_id) for player_id in range(0,4)]

        # diff_next_states = self.__get_diff_next_states(tokenStatePlayers, nextTokenStatePlayers)
        # diff_next_state = diff_next_states[0] # current player

        # # Can sometimes be more than one elemtents, but only when transitioning to stacked state or normal or both or hitting more six in a row. Thus setting to size 1
        # if (len(diff_next_state) > 1):
        #     if (np.equal.reduce(diff_next_state == LudoPlayerQLearning.stacked)): # more than one stacked
        #         diff_next_state = np.array([LudoPlayerQLearning.stacked])
        #     elif( np.equal.reduce(diff_next_state == LudoPlayerQLearning.normal)): # more than one normal
        #         diff_next_state = np.array([LudoPlayerQLearning.normal])
        #     elif( len(diff_next_state[diff_next_state == LudoPlayerQLearning.stacked]) >= 1): # one normal and one stacked
        #         diff_next_state = np.array([LudoPlayerQLearning.stacked])
        #     elif( len(diff_next_state[diff_next_state == LudoPlayerQLearning.globe]) >= 1): # one/two normal and one globe
        #         diff_next_state = np.array([LudoPlayerQLearning.globe])
        #     elif( len(diff_next_state[diff_next_state == LudoPlayerQLearning.vulnerable]) >= 1): # one/two normal and one globe (hitting six more than one time)
        #         diff_next_state = np.array([LudoPlayerQLearning.vulnerable])
        #     elif( len(diff_next_state[diff_next_state == LudoPlayerQLearning.globe_home]) >= 1): # one/two normal and one globe home (hitting six more than one time)
        #         diff_next_state = np.array([LudoPlayerQLearning.globe_home])
        #     elif( len(diff_next_state[diff_next_state == LudoPlayerQLearning.goal]) >= 1): # one/two normal and one goal (hitting six more than one time)
        #         diff_next_state = np.array([LudoPlayerQLearning.goal])
        # # elif(len(diff_next_state) == 0 and np.array(state.state[0]).any() == LudoPlayerQLearning.end_game_area): # If moving token in end game area but not into goal
        # #     reward += LudoPlayerQLearning.moved_end_game_token
        # # else: # If no new state negative reward
        # #     reward -= 1



        # if (nextTokenStatePlayers[0] == np.array([LudoPlayerQLearning.goal, LudoPlayerQLearning.goal, LudoPlayerQLearning.goal, LudoPlayerQLearning.goal])).all():
        #     reward += LudoPlayerQLearning.r_win
        # elif(diff_next_state == LudoPlayerQLearning.goal):
        #     reward += LudoPlayerQLearning.r_one_token_win
        # elif(diff_next_state == LudoPlayerQLearning.homed):
        #     reward += LudoPlayerQLearning.r_suicide
        # elif(diff_next_state == LudoPlayerQLearning.globe or diff_next_state == LudoPlayerQLearning.stacked):
        #     reward += LudoPlayerQLearning.r_safe
        # # elif(diff_next_state == LudoPlayerQLearning.star):
        # #     reward += LudoPlayerQLearning.star_jump
        # elif(diff_next_state == LudoPlayerQLearning.vulnerable or diff_next_state == LudoPlayerQLearning.globe_home):
        #     reward += LudoPlayerQLearning.r_got_vulnerable
        # # elif(diff_next_state == LudoPlayerQLearning.end_game_area):
        # #     reward += LudoPlayerQLearning.r_end_game_safe

        # # Got one player out of home
        # # if (diff_next_state == LudoPlayerQLearning.normal and )

        # # Checking for knocking home an opponent player
        # if self.__did_knock_home(diff_next_states):
        #     reward += LudoPlayerQLearning.r_knock_home

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
            i = 3
            while not valid_actions[best_action]:
                best_action = np.argsort(QTable[tmpTokenState])[i]
                i -= 1


            Action_probabilities[best_action] += 1.0
            return Action_probabilities


        if(self.__chosenPolicy == "epsilon greedy"):
            return epsilonGreedyPolicy 
        if(self.__chosenPolicy == "greedy"):
            return greedyPolicy


    # Q LEARNING #
    def QLearning(self, state, next_states):  # Inspiration from https://www.geeksforgeeks.org/q-learning-in-python/?fbclid=IwAR1UXR88IuJBhhTakjxNq_gcf3nCmJB0puuoA46J8mZnEan_qx9hhoFzhK8
        # Convert statespace representation for current states
        tokenStates = [self.__getTokenState(state, player_id) for player_id in range(0,4)]
        

        # Creates entry if current state does not exists
        self.__updateQTable(tokenStates[0], np.array([0.0, 0.0, 0.0, 0.0]))

        # Get probabilites based on initialized policy (chosenPolicy)
        policy = self.policies(self.__QTable, self.__epsilon, state, next_states) # returns a policy function
        actionProbability = policy(tokenStates[0])

        # Choose action based on the probability distribution
        action = np.random.choice( LudoPlayerQLearning.actions, p=actionProbability )

        # Find next state based on action and updates Q-table. 
        next_states_based_action = next_states[action] 
        nextTokenStates = [self.__getTokenState(next_states_based_action, player_id) for player_id in range(0,4)]

        # Static reward
        reward = self.__calc_reward(state, next_states_based_action, tokenStates, nextTokenStates)
        self.total_reward += reward

        # Cummulative reward
        # reward = self.__calc_cum_reward(action, next_states)
        #reward = self.__calc_cum_mean_reward(action, next_states)
        # self.total_reward += reward

        # Creates entry if nextTokenState does not exists
        self.__updateQTable(nextTokenStates[0], np.array([0.0, 0.0, 0.0, 0.0]))

        # Update based on TD Update
        td_target = reward + self.__discount_factor * np.max(self.__QTable[str(nextTokenStates[0])])
        td_delta = td_target - self.__QTable[str(tokenStates[0])][action] 
        update_val = self.__alpha * td_delta 

        self.__QTable[str(tokenStates[0])][action] += update_val

        # print("n",nextTokenState)
        #print("r",reward)

        return action

    def play(self, state, dice_roll, next_states):

       # cProfile.runctx('self.QLearning(state, next_states)', globals(), locals()) # Time profile for functions

        action = self.QLearning(state, next_states)

        return action # return number token want to move
