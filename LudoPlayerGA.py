from perf.pyludo.utils import token_vulnerability, is_stacked, is_globe_pos, is_on_opponent_globe, star_jump
from perf.pyludo.LudoGame import LudoState
import numpy as np
import random

class GA_player_base():


    def __init__(self, chromosome):
        self.chromosome = chromosome

    def play(self, state, dice_roll, next_states):
        """
        Based on the action values weighted by the chromosome
        it chooses to move the token which has the heighest 
        valid action value
        """
        action_values = self.evaluate_actions(state, next_states)
        action = np.argsort(-action_values)
        for token_id in action:
            if next_states[token_id] != False:
                return token_id


        action = np.argmax(action_values)



    def evaluate_actions(self, state, next_states):
        """
        Evaluate which actions are valid for a token
        Create this in GA player subclass
        """
        pass

    def will_send_opponent_home(self, opponent_states, opponent_next_states):
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



class simple_GA_player(GA_player_base):
    name = "simple GA"
    def __init__(self, chromosome):
        
        super(simple_GA_player, self).__init__(chromosome)
        
    def evaluate_actions(self, state, next_states):
        """
        Evaluate which actions are valid for a token
        and return an action list weighted with chromosome
        """
        action_value_list = np.zeros(4)
        for i in range(4):
            action_value_list[i] = self.state_representation_for_token(state, next_states[i], i)
        
        return action_value_list

    def state_representation_for_token(self, state, next_state, token_id):
        """
        Finds valid moves for a token and
        weights them with chromosome and return
        a sum of that.
        Inspiration from https://github.com/RasmusHaugaard/pyludo-ai/blob/master/GAPlayers.py
        """
        if next_state == False:
            return False

        current_pos_token = state[0][token_id]
        next_pos_token = next_state[0][token_id]

        current_opponent_states = state[1:]
        next_opponent_states = next_state[1:]

        moved_out = (current_pos_token == -1) and (next_pos_token == 1)
        into_goal = (next_pos_token == 99)
        send_opp_home = self.will_send_opponent_home(np.array(current_opponent_states), np.array(next_opponent_states))
        send_self_home = (next_pos_token == -1)

        reduced_state = [moved_out, into_goal, send_opp_home, send_self_home]

        action_value_token = np.sum(np.array(reduced_state) * np.array(self.chromosome))
        
        return action_value_token

   