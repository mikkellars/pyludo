import numpy as np
import multiprocessing
import random
from perf.pyludo import LudoGame, LudoState


class basetournement():

    def __init__(self, player_type, population_size:int):
        self.player_type = player_type
        self.population_size = population_size

    def play_tournament(self, chromosomes, play_iterations:int):
        players = [self.player_type(chromosome) for chromosome in chromosomes] # init players with chromosomes
        players = np.array(players)
        np.random.shuffle(players)

        scores = multiprocessing.Array("i", self.population_size, lock=False)

        multiprocess = []

        for player_ids in range(0,self.population_size,4):
            four_players = players[player_ids:player_ids+4]

            p = multiprocessing.Process(target=self.__play, args=[play_iterations,four_players, scores, player_ids])
            multiprocess.append(p)
            p.start()
            
                
        for index, process in enumerate(multiprocess):
            process.join()

        # Reshaping the flatten scores
        scores = np.reshape(scores[:],(self.population_size//4,4))
        
    def __play(self, play_iterations, four_players, scores, player_ids):
        for i, player in enumerate(four_players):
            player.id = i # selv tildele atributter uden defineret i klassen

        score = [0, 0, 0, 0]

        for i in range(play_iterations):
            
            random.shuffle(four_players)
            ludoGame = LudoGame(four_players)

            winner = ludoGame.play_full_game()
            score[four_players[winner].id] += 1

        scores[player_ids:player_ids+4] = score

    def selection(self, players, scores):
        for  in range(0,self.population_size,4):
            scores[]




        

            
              
