import random
import time
import numpy as np
import sys

from tqdm import tqdm
from perf.pyludo import LudoGame, LudoState
from PlotStatistics import PlotStatistics
import multiprocessing

from LudoPlayerQLearning import LudoPlayerQLearning
from perf.pyludo.StandardLudoPlayers import LudoPlayerRandom
from LudoPlayerGA import simple_GA_player
from LudoPlayerQLearningSimple import LudoPlayerQLearningSimple


def main():
    players = []
    players = [LudoPlayerRandom() for _ in range(3)]
    epsilon = 0.1
    discount_factor = 0.5
    learning_rate = 0.1
    parameters = [epsilon, discount_factor, learning_rate]
    t1 = LudoPlayerQLearningSimple(parameters, chosenPolicy="greedy", QtableName='1_QTable_simple', RewardName='1_Reward_simple')
    players.append(t1)
    for i, player in enumerate(players):
        player.id = i # selv tildele atributter uden defineret i klassen

    score = [0, 0, 0, 0]


    n = 5000
    start_time = time.time()
    tqdm_1 = tqdm(range(n), ascii=True)
    for i in tqdm_1:
        tqdm_1.set_description_str(f"win rates {np.around(score/np.sum(score),decimals=2)*100}") 
        random.shuffle(players)
        ludoGame = LudoGame(players)

        winner = ludoGame.play_full_game()
        score[players[winner].id] += 1

        for player in players: # Saving reward for QLearning player
            if type(player)==LudoPlayerQLearningSimple:
                player.append_reward()
                player.reset_upd_val()
                # player.rewards.append(player.total_reward)
                # player.total_reward = 0

    for player in players:
        if type(player)==LudoPlayerQLearningSimple:
            player.saveQTable() 
            player.saveReward()

    duration = time.time() - start_time

    print('win distribution:', score)

    Plot = PlotStatistics()
    Plot.plotReward(pathToCSV='1_Reward_simple_e-0.1_d-0.5_a-0.1.csv', numMovAvg=1000)


    # starttime = time.time()
    # epsilons = [0.1, 0.2]
    # discount_factors = [0.9, 0.5, 0.2]
    # learning_rate = [0.5, 0.25, 0.1]

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
    
    # Plot = PlotStatistics()
    # Plot.plotMultiple(pathToFolder="Param_optimization", numMovAvg=1000)

if __name__ == "__main__":
    main()