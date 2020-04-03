from pyludo import LudoGame, LudoPlayerRandom
import random
import time

import sys
from LudoPlayerQLearning import LudoPlayerQLearning




players = [LudoPlayerRandom() for _ in range(3)]
players.append(LudoPlayerQLearning("epsilon greedy"))
for i, player in enumerate(players):
    player.id = i # selv tildele atributter uden defineret i klassen
    

#random.shuffle(players)
# ludoGame = LudoGame(players)

# print(ludoGame.play_full_game())
# players[3].saveQTable()
#players[3].printQTable()


score = [0, 0, 0, 0]

n = 1

start_time = time.time()
for i in range(n):
    random.shuffle(players)
    ludoGame = LudoGame(players)

    winner = ludoGame.play_full_game()
    score[players[winner].id] += 1
    print('Game ', i, ' done')

for player in players:
    if player.id == 3:
        player.saveQTable() # only one player that is Qlearning        

duration = time.time() - start_time

print('win distribution:', score)
print('games per second:', n / duration)
