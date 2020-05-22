import numpy as np
import random
from tqdm import tqdm
from perf.pyludo import LudoGame
from perf.pyludo.StandardLudoPlayers import LudoPlayerRandom
from selection import basetournement
from PlotStatistics import PlotStatistics
from LudoPlayerQLearningAction import LudoPlayerQLearningAction

def train():
    basetournement_test = basetournement(LudoPlayerQLearningAction, 32, 'qlearning', 'GA_qlearning_param_opti_logs/qlean_param_opti_ga_chromosomes.csv') # 32 chromomes max out the thread count of this computer. 

    chromosomes = np.random.uniform(low=0, high=1, size=(32, 3)) # 32 chromomes max out the thread count of this computer. 
    print(chromosomes)

    best_player = basetournement_test.play_for_generations(chromosomes, tournament_it=1000, generations_it=100, validation_it=10000) # problem gets the same 5 chromosome as the best
    print(f'The best player chosen has chromosome {best_player.chromosome}')

    return best_player

def eval(best_player):
    # Evaluting the performance of the best player
    players = [LudoPlayerRandom() for _ in range(2)]
    players.append(best_player)
    players.append(best_player)

    n_games = 10000

    for i, player in enumerate(players):
        player.id = i # selv tildele atributter uden defineret i klassen

    score = [0, 0, 0, 0]

    tqdm_1 = tqdm(range(n_games), ascii=True)
    for i in tqdm_1:  
        random.shuffle(players)
        ludoGame = LudoGame(players)

        winner = ludoGame.play_full_game()
        score[players[winner].id] += 1

        tqdm_1.set_description_str(f"Validating best player: win rates {np.around(score/np.sum(score),decimals=2)*100}") 

def plot():
    Plot = PlotStatistics()
    labels =  [r'$\epsilon$', r'$\gamma$', r'$\alpha$']
    #Plot.plot_chromosome_2D(path_to_csv='GA_qlearning_param_opti_logs/qlean_param_opti_ga_chromosomes.csv', labels=labels)
    Plot.plot_chromosome_3D(path_to_csv='GA_qlearning_param_opti_logs/qlean_param_opti_ga_chromosomes.csv')
    Plot.plot_chromosome_2D_qlearning(path_to_csv='GA_qlearning_param_opti_logs/qlean_param_opti_ga_chromosomes.csv', labels=labels)

def main():
   # best_player = train()
   # eval(best_player)
    plot()


if __name__ == "__main__":
    main()