from pyludo import LudoGame, LudoPlayerRandom
import random
import time

import sys
from LudoPlayerQLearning import LudoPlayerQLearning




players = [LudoPlayerRandom() for _ in range(3)]
players.append(LudoPlayerQLearning())
for i, player in enumerate(players):
    player.id = i # selv tildele atributter uden defineret i klassen
    

#random.shuffle(players)
ludoGame = LudoGame(players)

print(ludoGame.play_full_game())

players[3].printQTable()
players[3].saveQTable()

# score = [0, 0, 0, 0]

# n = 10

# start_time = time.time()
# for i in range(n):
#     random.shuffle(players)
#     ludoGame = LudoGame(players)

#     print(ludoGame.state.state)

#     winner = ludoGame.play_full_game()
#     score[players[winner].id] += 1
#     print('Game ', i, ' done')


# duration = time.time() - start_time

# print('win distribution:', score)
# print('games per second:', n / duration)
