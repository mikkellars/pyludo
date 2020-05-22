import random
import time
import numpy as np
import sys

from tqdm import tqdm
from perf.pyludo import LudoGame, LudoState
from PlotStatistics import PlotStatistics
import multiprocessing

from perf.pyludo.StandardLudoPlayers import LudoPlayerRandom
from LudoPlayerGA import simple_GA_player
from LudoPlayerQLearningAction import LudoPlayerQLearningAction

def play_with_on_QLearning_thread(num_games, epsilon, discount_factor, learning_rate):
    players = [LudoPlayerRandom() for _ in range(3)]
    param = [epsilon, discount_factor, learning_rate]
    players.append(LudoPlayerQLearningAction(Parameters=param, chosenPolicy="greedy", QtableName='Qtable', RewardName='Reward'))
    for i, player in enumerate(players):
        player.id = i # selv tildele atributter uden defineret i klassen


    score = [0, 0, 0, 0]

    n = num_games


    for i in range(n):
        random.shuffle(players)
        ludoGame = LudoGame(players)
        
        for player in players: # Saving reward for QLearning player
            if type(player)==LudoPlayerQLearningAction:
                player.append_reward()
                
        winner = ludoGame.play_full_game()
        score[players[winner].id] += 1
        if i%2500 == 0:
            print('Game ', i, ' done')

    for player in players:
        if type(player)==LudoPlayerQLearningAction:
            player.saveQTable() 
            player.saveReward()   
    
    print(f'Player with eps={epsilon}, discountfactor={discount_factor} and learningrate={learning_rate} won {np.around(score/np.sum(score),decimals=2)*100}')


def param_optimization():
    # starttime = time.time()
    # epsilons = [0.1, 0.05]
    # discount_factors = [0.9, 0.5]
    # learning_rate = [0.5, 0.25, 0.1, 0.05]

    # combination = []
    # multiprocess = []

    # for e in epsilons:
    #     for d in discount_factors:
    #         for l in learning_rate:
    #             p = multiprocessing.Process(target=play_with_on_QLearning_thread, args=(10000, e, d, l))
    #             multiprocess.append(p)
    #             p.start()
            
                
    # for index, process in enumerate(multiprocess):
        
    #     process.join()

   # print('That took {} seconds'.format(time.time() - starttime))
    Plot = PlotStatistics()
    Plot.plotMultiple(pathToFolder="Param_optimization", numMovAvg=1000)

def normal_play():
    players = []
    players = [LudoPlayerRandom() for _ in range(3)]
      
    epsilon = 0.05 #0.40463712 #0.05 #
    discount_factor =  0.5 #0.14343606 #0.5 #
    learning_rate = 0.25#0.10783296  #0.25 # 
    parameters = [epsilon, discount_factor, learning_rate]

    t1 = LudoPlayerQLearningAction(parameters, chosenPolicy="epsilon greedy", QtableName='Qlearning_action_logs/1QTable_action_r_win', RewardName='Qlearning_action_logs/1Reward_action_r_win')
    players.append(t1)
    for i, player in enumerate(players):
        player.id = i # selv tildele atributter uden defineret i klassen

    score = [0, 0, 0, 0]    

    # n = 40000
    # start_time = time.time()
    # tqdm_1 = tqdm(range(n), ascii=True)
    # for i in tqdm_1:
    #     tqdm_1.set_description_str(f"win rates {np.around(score/np.sum(score),decimals=2)*100}") 
    #     random.shuffle(players)
    #     ludoGame = LudoGame(players)

    #     winner = ludoGame.play_full_game()
    #     score[players[winner].id] += 1

    #     for player in players: # Saving reward for QLearning player
    #         if type(player)==LudoPlayerQLearningAction:
    #             player.append_reward()

    # for player in players:
    #     if type(player)==LudoPlayerQLearningAction:
    #         player.saveQTable() 
    #         player.saveReward()

    # duration = time.time() - start_time

    # print('win distribution percentage', (score/np.sum(score))*100)
    # print('win distribution:', score)

    Plot = PlotStatistics()
    Plot.plotReward(pathToCSV=f'Qlearning_action_logs/1Reward_action_r_win_e-{epsilon}_d-{discount_factor}_a-{learning_rate}.csv', numMovAvg=1000)

def main():
   #param_optimization()
   normal_play()


if __name__ == "__main__":
    main()