import numpy as np
import random
from tqdm import tqdm
from perf.pyludo import LudoGame
from perf.pyludo.StandardLudoPlayers import LudoPlayerRandom
from selection import basetournement
from PlotStatistics import PlotStatistics
from LudoPlayerGA import simple_GA_player
from LudoPlayerQLearningSimple import LudoPlayerQLearningSimple





# basetournement_test = basetournement(simple_GA_player, 32, 'ga-simple') # 32 chromomes max out the thread count of this computer. 

# chromosomes = np.random.uniform(low=-1, high=1, size=(32, 4)) # 32 chromomes max out the thread count of this computer. 
# print(chromosomes)

# best_player = basetournement_test.play_for_generations(chromosomes, tournament_it=10, generations_it=100, validation_it=1) # problem gets the same 5 chromosome as the best
# print(f'The best player chosen has chromosome {best_player.chromosome}')

# # Evaluting the performance of the best player
# players = [LudoPlayerRandom() for _ in range(3)]
# players.append(best_player)

# n_games = 10000

# for i, player in enumerate(players):
#     player.id = i # selv tildele atributter uden defineret i klassen

# score = [0, 0, 0, 0]

# tqdm_1 = tqdm(range(n_games), ascii=True)
# for i in tqdm_1:  
#     random.shuffle(players)
#     ludoGame = LudoGame(players)

#     winner = ludoGame.play_full_game()
#     score[players[winner].id] += 1

#     tqdm_1.set_description_str(f"Validating best player: win rates {np.around(score/np.sum(score),decimals=2)*100}") 

Plot = PlotStatistics()
Plot.plot_chromosome_2D(path_to_csv='chromosomes_plot.csv')

##### TESTING FOR QLEARNING PARAMETERS #####

# basetournement_test = basetournement(LudoPlayerQLearningSimple, 20, type_player='qlearning') # 32 chromomes max out the thread count of this computer. 

# chromosomes = np.random.uniform(low=0, high=1, size=(20, 3)) # 32 chromomes max out the thread count of this computer. 
# print(chromosomes)

# best_player = basetournement_test.play_for_generations(chromosomes, tournament_it=100, generations_it=20, validation_it=1) # problem gets the same 5 chromosome as the best
# print(f'The best player chosen has chromosome {best_player.chromosome}')

# Evaluting the performance of the best player
# players = [LudoPlayerRandom() for _ in range(3)]
# players.append(best_player)

# n_games = 1000

# for i, player in enumerate(players):
#     player.id = i # selv tildele atributter uden defineret i klassen

# score = [0, 0, 0, 0]

# tqdm_1 = tqdm(range(n_games), ascii=True)
# for i in tqdm_1:  
#     random.shuffle(players)
#     ludoGame = LudoGame(players)

#     winner = ludoGame.play_full_game()
#     score[players[winner].id] += 1

#     tqdm_1.set_description_str(f"win rates {np.around(score/np.sum(score),decimals=2)*100}") 


# Plotting 2D and 3D for qlearning parameter optimization
# Plot = PlotStatistics()
# Plot.plot_chromosome_3D(path_to_csv='chromosomes_plot_Qlearning.csv')
# Plot.plot_chromosome_2D(path_to_csv='chromosomes_plot_Qlearning.csv')