import random
import time
import numpy as np
import sys
import math

from tqdm import tqdm
from perf.pyludo import LudoGame, LudoState
from PlotStatistics import PlotStatistics
import multiprocessing

from LudoPlayerQLearningToken import LudoPlayerQLearningToken
from LudoPlayerQLearningAction import LudoPlayerQLearningAction
from perf.pyludo.StandardLudoPlayers import LudoPlayerRandom
from LudoPlayerGA import simple_GA_player
from mathias_player import MathiasPlayer
from smart_player import smart_player

def play_on_thread(num_games, player_type, opponent_type):

    # Player initilization
    if player_type is LudoPlayerRandom:
        players = [player_type() for _ in range(2)]
    elif player_type is LudoPlayerQLearningAction:
        # Exhuastive search parameters
        epsilon = 0.05 
        discount_factor =  0.5
        learning_rate = 0.25
        parameters = [epsilon, discount_factor, learning_rate]
        players = [LudoPlayerQLearningAction(parameters, chosenPolicy="greedy", QtableName='Qlearning_action_logs/QTable_action_r_win', RewardName='Qlearning_action_logs/Reward_action_r_win') for _ in range(2)]
    elif player_type is LudoPlayerQLearningToken:
        # Exhuastive search parameters
        epsilon = 0.05 
        discount_factor =  0.5
        learning_rate = 0.25
        players = [LudoPlayerQLearningToken("greedy", QtableName='Qlearning_token_logs/QTable_token_40000', RewardName='Qlearning_token_logs/Reward_token_40000', epsilon=epsilon, discount_factor=discount_factor, learning_rate=learning_rate) for _ in range(2)]
    elif player_type is simple_GA_player:
        chromosome = [0.800085, 2.05562201, 0.55735083, -0.9978861]
        players = [simple_GA_player(chromosome) for _ in range(2)]
    elif player_type is MathiasPlayer:
        players = [MathiasPlayer() for _ in range(2)]
    elif player_type is smart_player:
        players = [smart_player() for _ in range(2)]
    else:
        print("player not found please check correctly added")
        return

    # Opponent initilization
    if opponent_type is LudoPlayerRandom:
        players += [LudoPlayerRandom() for _ in range(2)]
    elif opponent_type is LudoPlayerQLearningAction:
        # Exhuastive search parameters
        epsilon = 0.05 
        discount_factor =  0.5
        learning_rate = 0.25
        parameters = [epsilon, discount_factor, learning_rate]
        players += [LudoPlayerQLearningAction(parameters, chosenPolicy="greedy", QtableName='Qlearning_action_logs/QTable_action_r_win', RewardName='Qlearning_action_logs/Reward_action_r_win') for _ in range(2)]
    elif opponent_type is LudoPlayerQLearningToken:
        # Exhuastive search parameters
        epsilon = 0.05 
        discount_factor =  0.5
        learning_rate = 0.25
        players += [LudoPlayerQLearningToken("greedy", QtableName='Qlearning_token_logs/QTable_token_40000', RewardName='Qlearning_token_logs/Reward_token_40000', epsilon=epsilon, discount_factor=discount_factor, learning_rate=learning_rate) for _ in range(2)]
    elif opponent_type is simple_GA_player:
        chromosome = [0.800085, 2.05562201, 0.55735083, -0.9978861]
        players += [simple_GA_player(chromosome) for _ in range(2)]
    elif opponent_type is MathiasPlayer:
        players += [MathiasPlayer() for _ in range(2)]
    elif player_type is smart_player:
        players = [smart_player() for _ in range(2)]
    else:
        print("player not found please check correctly added")
        return
    
    for i, player in enumerate(players):
        player.id = i # selv tildele atributter uden defineret i klassen


    score = [0, 0, 0, 0]

    n = num_games


    for i in range(n):
        random.shuffle(players)
        ludoGame = LudoGame(players)
        winner = ludoGame.play_full_game()
        score[players[winner].id] += 1
        # if i%2500 == 0:
        #     print('Game ', i, ' done')

    win_rate_percent_player = np.sum((score/np.sum(score))[0:2])*100
    win_rate_percent_opponent = np.sum((score/np.sum(score))[2:])*100

    significant_player = binominal_test(num_games, np.sum(score[0:2]))
    significant_opponent = binominal_test(num_games, np.sum(score[2:]))

    print(f'Player: {player_type.name}, Won: {win_rate_percent_player:.2f}, Significant: {significant_player} VERSUS. Opponent: {opponent_type.name}, Won: {win_rate_percent_opponent:.2f}, Significant: {significant_opponent}')

def binominal_test(sample_size, num_of_success):
    z = 1.6449 # calculated in matlab with norminv(1-alpha) -> alpha 0.05 critial_value
    pi = 0.5 # probability of success / test value against. We want to test if different from a win-rate of 50%
    n = float(sample_size)
    k = float(num_of_success)
    test_stat = (k-n*pi)/(math.sqrt(n*pi*(1-pi)))
    # print(k)
    # print(z)
    # print(test_stat)
    return (abs(test_stat) > z) # If true it is different from 50% / it is significant

def main():
    player = [LudoPlayerQLearningToken, LudoPlayerQLearningAction, LudoPlayerRandom, simple_GA_player, smart_player, MathiasPlayer]
   # player = [LudoPlayerRandom, simple_GA_player]
    opponent_player = player

    multiprocess = []
    for i, pl in enumerate(player):
        for j, opp_pl in enumerate(player):
            if i > j:
                continue
            p = multiprocessing.Process(target=play_on_thread, args=[10000, pl, opp_pl])
            multiprocess.append(p)
            p.start()

    for index, process in enumerate(multiprocess):
        process.join()
    
    # play_on_thread(1000, LudoPlayerRandom, LudoPlayerRandom)




if __name__ == "__main__":
    main()