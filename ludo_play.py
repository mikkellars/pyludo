from pyludo import LudoGame, LudoPlayerRandom
import random
import time
import numpy as np
import sys
from LudoPlayerQLearning import LudoPlayerQLearning
from PlotStatistics import PlotStatistics



players = [LudoPlayerRandom() for _ in range(3)]
players.append(LudoPlayerQLearning("epsilon greedy"))
for i, player in enumerate(players):
    player.id = i # selv tildele atributter uden defineret i klassen


score = [0, 0, 0, 0]

# n = 10000

# start_time = time.time()
# for i in range(n):
#     random.shuffle(players)
#     ludoGame = LudoGame(players)
    
#     for player in players: # Saving reward for QLearning player
#         if player.id == 3:
#             player.saveReward()

#     winner = ludoGame.play_full_game()
#     score[players[winner].id] += 1
#     if i%100==0:
#         print('Game ', i, ' done')

# for player in players:
#     if player.id == 3:
#         player.saveQTable() # only one player that is Qlearning        

# duration = time.time() - start_time

# print('win distribution:', score)

# print('win distribution percentage', (score/np.sum(score))*100)
# print('games per second:', n / duration)

Plot = PlotStatistics()
Plot.plotReward(pathToCSV="Reward.csv", numMovAvg=1)