import random
import time
import numpy as np
import sys

from tqdm import tqdm
from perf.pyludo import LudoGame, LudoState
from PlotStatistics import PlotStatistics
import multiprocessing

from LudoPlayerQLearningToken import LudoPlayerQLearningToken
from perf.pyludo.StandardLudoPlayers import LudoPlayerRandom


def play_with_on_QLearning_thread(num_games, epsilon, discount_factor, learning_rate):
    players = [LudoPlayerRandom() for _ in range(3)]
    players.append(LudoPlayerQLearningToken("epsilon greedy", QtableName='Param_optimization/QTable', RewardName='Param_optimization/Reward', epsilon=epsilon, discount_factor=discount_factor, learning_rate=learning_rate))
    for i, player in enumerate(players):
        player.id = i # selv tildele atributter uden defineret i klassen


    score = [0, 0, 0, 0]

    n = num_games


    for i in range(n):
        random.shuffle(players)
        ludoGame = LudoGame(players)
        
        for player in players: # Saving reward for QLearning player
            if type(player)==LudoPlayerQLearningToken:
                player.append_reward()
                
        winner = ludoGame.play_full_game()
        score[players[winner].id] += 1
        if i%2500 == 0:
            print('Game ', i, ' done')

    for player in players:
        if type(player)==LudoPlayerQLearningToken:
            player.saveQTable() 
            player.saveReward()   
    
    print(f'Player with eps={epsilon}, discountfactor={discount_factor} and learningrate={learning_rate} won {np.around(score/np.sum(score),decimals=2)*100}')


def param_optimization():
    starttime = time.time()
    epsilons = [0.1, 0.05]
    discount_factors = [0.9, 0.5]
    learning_rate = [0.5, 0.25, 0.1, 0.05]

    combination = []
    multiprocess = []

    for e in epsilons:
        for d in discount_factors:
            for l in learning_rate:
                p = multiprocessing.Process(target=play_with_on_QLearning_thread, args=(10000, e, d, l))
                multiprocess.append(p)
                p.start()
            
                
    for index, process in enumerate(multiprocess):
        
        process.join()

    print('That took {} seconds'.format(time.time() - starttime))
    Plot = PlotStatistics()
    Plot.plotMultiple(pathToFolder="Param_optimization", numMovAvg=1000)

def normal_training():
    
    players = [LudoPlayerRandom() for _ in range(3)]

    # GA optimized param
    # e = 0.40463712
    # d = 0.14343606
    # a = 0.10783296

    # Exhaustive search param
    e = 0.05
    d = 0.5
    a = 0.25

    # players.append(LudoPlayerQLearningToken("epsilon greedy", QtableName='Qlearning_token_logs/QTable_token_40000', RewardName='Qlearning_token_logs/Reward_token_40000', epsilon=e, discount_factor=d, learning_rate=a))
    # for i, player in enumerate(players):
    #     player.id = i # selv tildele atributter uden defineret i klassen


    # score = [0, 0, 0, 0]

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
    #         if type(player)==LudoPlayerQLearningToken:
    #             player.append_reward()

    # for player in players:
    #     if type(player)==LudoPlayerQLearningToken:
    #         player.saveQTable() # only one player that is Qlearning        
    #         player.saveReward()


    # duration = time.time() - start_time

    # print('win distribution:', score)

    # print('win distribution percentage', (score/np.sum(score))*100)
    # print('games per second:', n / duration)

    Plot = PlotStatistics()
    Plot.plotReward(pathToCSV=f'Qlearning_token_logs/Reward_token_40000_e-{e}_d-{d}_a-{a}.csv', numMovAvg=1000)

def main():
    normal_training()
    #param_optimization()


if __name__ == "__main__":
    main()