import random
import numpy as np

class smart_player:
    """Smart player
    """

    name = "smart"

    @staticmethod
    def play(state, dice_roll, next_states):
        """Always put on token out from home if one can do this
        else chooses to move the token most ahead into goal.
        
        Arguments:
            state {[type]} -- [description]
            dice_roll {[type]} -- [description]
            next_state {[type]} -- [description]
        """
        # Player out from home
        if dice_roll == 6:
            home_players = [i for i, token in enumerate(state[0]) if token == -1]
            if home_players:
                return random.choice(home_players)

        # Chooses token most ahead
        tmp_state = state.state[0]
        token_most_ahead = np.argmax(tmp_state) # Find index of action which token most ahead
        # Check if valid action
        for i in range(len(tmp_state)):
            if not next_states[token_most_ahead]:
                tmp_state[token_most_ahead] = -99 # sets to a high negative number to not pick this token again
                token_most_ahead = np.argmax(tmp_state) # Find index of action which token most ahead
            else:
                break   
            
        return token_most_ahead