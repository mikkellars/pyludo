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


# def play_with_on_QLearning_thread(num_games, epsilon, discount_factor, learning_rate):
#     players = [LudoPlayerRandom() for _ in range(3)]
#     players.append(LudoPlayerQLearning("epsilon greedy", QtableName='Param_optimization/QTable', RewardName='Param_optimization/Reward', epsilon=epsilon, discount_factor=discount_factor, learning_rate=learning_rate))
#     for i, player in enumerate(players):
#         player.id = i # selv tildele atributter uden defineret i klassen


#     score = [0, 0, 0, 0]

#     n = num_games


#     for i in range(n):
#         random.shuffle(players)
#         ludoGame = LudoGame(players)
        
#         for player in players: # Saving reward for QLearning player
#             if player.id == 3:
#                 player.saveReward()

#         winner = ludoGame.play_full_game()
#         score[players[winner].id] += 1
#         if i==5000:
#             print('Game ', i, ' done')

#     for player in players:
#         if player.id == 3:
#             player.saveQTable() # only one player that is Qlearning        


#     print(f'Win distribution percentage with epsilon {epsilon}, discount factor {discount_factor} and learning rate {learning_rate}: ")', (score/np.sum(score))*100)


####################################################################################################################################################
###                                                         PARAMETERS OPTIMIZATION                                                              ###
####################################################################################################################################################
# if __name__ == '__main__':
#     starttime = time.time()
#     epsilons = [0.1, 0.2]
#     discount_factors = [1, 0.9, 0.5]
#     learning_rate = [0.5, 0.25, 0.1]

#     combination = []
#     multiprocess = []

#     for e in epsilons:
#         for d in discount_factors:
#             for l in learning_rate:
#                 p = multiprocessing.Process(target=play_with_on_QLearning_thread, args=(10000, e, d, l))
#                 multiprocess.append(p)
#                 p.start()
            
                
#     for index, process in enumerate(multiprocess):
        
#         process.join()

#     print('That took {} seconds'.format(time.time() - starttime))

####################################################################################################################################################
###                                                              SINGLE GAME TEST                                                                ###
####################################################################################################################################################

# players = [LudoPlayerRandom() for _ in range(3)]
# players.append(LudoPlayerQLearning("epsilon greedy", QtableName='QTable', RewardName='Reward', epsilon=0.1, discount_factor=0.5, learning_rate=0.1))
# for i, player in enumerate(players):
#     player.id = i # selv tildele atributter uden defineret i klassen


# score = [0, 0, 0, 0]

# n = 100
# start_time = time.time()
# tqdm_1 = tqdm(range(n), ascii=True)
# for i in tqdm_1:
#     tqdm_1.set_description_str(f"win rates {np.around(score/np.sum(score),decimals=2)*100}") 
#     random.shuffle(players)
#     ludoGame = LudoGame(players)

#     winner = ludoGame.play_full_game()
#     score[players[winner].id] += 1

#     for player in players: # Saving reward for QLearning player
#         if type(player)==LudoPlayerQLearning:
#             player.rewards.append(player.total_reward)
#             player.total_reward = 0

# for player in players:
#     if type(player)==LudoPlayerQLearning:
#         player.saveQTable() # only one player that is Qlearning        
#         player.saveReward()

#         # ##### TESTING #####
#         # state = LudoState()
#         # state.state = np.array([[46, -1, -1, -1]])
#         # player.testTokenState(state)

# duration = time.time() - start_time

# print('win distribution:', score)

# print('win distribution percentage', (score/np.sum(score))*100)
# print('games per second:', n / duration)

# Plot = PlotStatistics()
# Plot.plotReward(pathToCSV='Reward_e-0.1_d-0.5_a-0.1.csv', numMovAvg=1000)
# Plot.plotMultiple(pathToFolder="Param_optimization", numMovAvg=1000)


######################## GA PLAYER ##################################


# players = [LudoPlayerRandom() for _ in range(3)]
# players.append(simple_GA_player([10,10,1,-100]))
# for i, player in enumerate(players):
#     player.id = i # selv tildele atributter uden defineret i klassen


# score = [0, 0, 0, 0]

# n = 1000
# start_time = time.time()
# tqdm_1 = tqdm(range(n), ascii=True)
# for i in tqdm_1:
#     tqdm_1.set_description_str(f"win rates {np.around(score/np.sum(score),decimals=2)*100}") 
#     random.shuffle(players)
#     ludoGame = LudoGame(players)

#     winner = ludoGame.play_full_game()
#     score[players[winner].id] += 1


# duration = time.time() - start_time

# print('win distribution:', score)

######################## SIMPLE QLEARNING PLAYER ##################################

players = [LudoPlayerRandom() for _ in range(3)]
players.append(LudoPlayerQLearningSimple("epsilon greedy", RewardName='Reward_simple.csv', epsilon=0.1, discount_factor=0.5, learning_rate=0.1))
for i, player in enumerate(players):
    player.id = i # selv tildele atributter uden defineret i klassen


score = [0, 0, 0, 0]

<<<<<<< HEAD
n = 5000
=======
n = 3000
>>>>>>> simple-q-learnig
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
            player.rewards.append(player.total_reward)
            player.total_reward = 0

for player in players:
    if type(player)==LudoPlayerQLearningSimple:
        # player.saveQTable() # Not implemented yet  
        player.saveReward()

duration = time.time() - start_time

print('win distribution:', score)

Plot = PlotStatistics()
Plot.plotReward(pathToCSV='Reward_simple.csv', numMovAvg=1000)