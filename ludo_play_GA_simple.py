import numpy as np
import random
from tqdm import tqdm
from perf.pyludo import LudoGame
from perf.pyludo.StandardLudoPlayers import LudoPlayerRandom
from selection import basetournement
from PlotStatistics import PlotStatistics
from LudoPlayerGA import simple_GA_player

def train():
    basetournement_test = basetournement(simple_GA_player, 128, 'ga-simple', 'GA_simple_logs/simple_ga_chromosomes.csv') # 32 chromomes max out the thread count of this computer. 
    chromosomes = np.random.uniform(low=-1, high=1, size=(128, 4)) # 32 chromomes max out the thread count of this computer. 
    print(chromosomes)

    best_player = basetournement_test.play_for_generations(chromosomes, tournament_it=100, generations_it=1000, validation_it=10000) # problem gets the same 5 chromosome as the best
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
    labels =  [r'$\theta_{1}$', r'$\theta_{2}$', r'$\theta_{3}$', r'$\theta_{4}$']
    Plot.plot_chromosome_2D(path_to_csv='GA_simple_logs/simple_ga_chromosomes.csv', labels=labels)


def main():
    best_player = train()
    eval(best_player)
    plot()


if __name__ == "__main__":
    main()