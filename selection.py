import numpy as np
import multiprocessing
import random
import copy
import csv
from tqdm import tqdm
from perf.pyludo import LudoGame, LudoState
from perf.pyludo.StandardLudoPlayers import LudoPlayerRandom
from LudoPlayerQLearningSimple import LudoPlayerQLearningSimple


class basetournement():

    def __init__(self, player_type, population_size:int, type_player, chromosomes_save_path):
        self.player_type = player_type
        self.population_size = population_size
        self.type = type_player
        self.save_path = chromosomes_save_path

    def __save_chromosomes(self, file_name, chromosomes):
        with open(file_name, 'w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            for chroms_gen in chromosomes:
                csv_writer.writerow(chroms_gen)

    def play_for_generations(self, chromosomes, tournament_it:int, generations_it:int, validation_it:int):
        players = [self.player_type(chromosome) for chromosome in chromosomes] # init players with chromosomes
        scores = None
        tqdm_1 = tqdm(range(generations_it), ascii=True)
        tqdm_1.set_description_str("Running tournaments") 
        save_chromosomes = []
        for gen_idx in tqdm_1:
            players, scores = self.play_tournament(players, tournament_it)

            # Saves chromosomes 3 times
            if (gen_idx == 0) or (gen_idx == generations_it//2) or (gen_idx == generations_it-1):
                print("Saving chromosomes for generation: ", gen_idx)
                tmp_chromosomes = []
                for p in players:
                    tmp_chromosomes.append(p.chromosome)
                
                save_chromosomes.append(tmp_chromosomes)

        # Saving chromosomes for plotting
        self.__save_chromosomes(self.save_path, save_chromosomes)

        # Getting the best for each tourmenent
        best_pop_id = np.argmax(scores,axis=1)
        best_player_each_pop = []
        for score_id, player_ids in enumerate(range(0,self.population_size,4)):
            best_player_each_pop.append(players[player_ids+best_pop_id[score_id]])
            print(players[player_ids+best_pop_id[score_id]].chromosome)

        # Eval the best players for each tourmenent against random and return the best player
        multiprocess = []
        win_rate = multiprocessing.Array("i", len(best_player_each_pop), lock=False) # win_rate shared varible for process
        for player_id, best_player in enumerate(best_player_each_pop):
            p = multiprocessing.Process(target=self.__play_against_random, args=[best_player, validation_it, player_id, win_rate])
            multiprocess.append(p)
            p.start()
            
        for index, process in enumerate(multiprocess):
            process.join()

        return best_player_each_pop[np.argmax(win_rate)]

    def __play_against_random(self, best_pop_player, game_iterations, p_id, win_rate):
        players = [LudoPlayerRandom() for _ in range(3)]
        players.append(best_pop_player)

        for i, player in enumerate(players):
            player.id = i # selv tildele atributter uden defineret i klassen


        score = [0, 0, 0, 0]

        for i in range(game_iterations):
            random.shuffle(players)
            ludoGame = LudoGame(players)

            winner = ludoGame.play_full_game()
            score[players[winner].id] += 1

        win_rate[p_id] = score[3]

        print(f'Chromosome {best_pop_player.chromosome} scored win percentage against random {np.around(score/np.sum(score),decimals=2)*100}')

    def play_tournament(self, next_gen_players, play_iterations:int):
        players = np.array(next_gen_players)
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

        players = self.selection(players, scores)
        return players, scores
        
    def __play(self, play_iterations, four_players, scores, player_ids):
        
        for i, player in enumerate(four_players):
            player.id = i # selv tildele atributter uden defineret i klassen

        score = [0, 0, 0, 0]

        for i in range(play_iterations):
            # if (i % play_iterations//2) == 0:
            #     print(f"player {player_ids} played {i}")    
            random.shuffle(four_players)
            ludoGame = LudoGame(four_players)

            winner = ludoGame.play_full_game()
            score[four_players[winner].id] += 1

        scores[player_ids:player_ids+4] = score

    def selection(self, players, scores):
        min_score_idx = []
        parent_pairs = []
        new_players = []

        for pop_idx in range(scores.shape[0]):
            #parents = (np.delete(scores[pop_idx,:], scores[pop_idx,:].argsort()[:2])) # deletes the two lowest values to get space for children
            parents = (scores[pop_idx,:]).argsort()[-2:][::-1] # finds idx the 2 highest values and uses them as parents
            parent_pairs.append(parents) 

        # Getting parents players into new players
        par_pair_idx = 0
        
        for player_ids in range(0,self.population_size,4):
            for parent_idx in parent_pairs[par_pair_idx]:
                
                new_players.append(players[player_ids+parent_idx])   
                
            # Mutation
            off_spring_1, off_spring_2 = self.__single_point_crossover(new_players[len(new_players)-1], new_players[len(new_players)-2]) # the last and seconds latest in the array is the newly added parents
            new_players.append(off_spring_1)
            new_players.append(off_spring_2)
            # counting up to get next parent pair from population
            par_pair_idx += 1
            
        return new_players

    def __single_point_crossover(self, parrent_1, parrent_2):
        offspring_1 = copy.deepcopy(parrent_1)
        offspring_2 = copy.deepcopy(parrent_2)

        offspring_1.chromosome = np.concatenate(( np.array(parrent_1.chromosome[0:len(parrent_1.chromosome)//2]), np.array(parrent_2.chromosome[len(parrent_2.chromosome)//2:len(parrent_2.chromosome)]) ))
        offspring_2.chromosome = np.concatenate(( np.array(parrent_2.chromosome[0:len(parrent_2.chromosome)//2]), np.array(parrent_1.chromosome[len(parrent_1.chromosome)//2:len(parrent_1.chromosome)]) ))

        # Mutation
        
        if self.type == 'qlearning':
            offspring_1.chromosome = self.__QLearning_mutation(offspring_1.chromosome, probability_for_mutation=0.1)
            offspring_2.chromosome = self.__QLearning_mutation(offspring_2.chromosome, probability_for_mutation=0.1)
        else:
            offspring_1.chromosome = self.__simple_mutation(offspring_1.chromosome, probability_for_mutation=0.1)
            offspring_2.chromosome = self.__simple_mutation(offspring_2.chromosome, probability_for_mutation=0.1)

        return offspring_1, offspring_2

    def __simple_mutation(self, chromosome, probability_for_mutation):
        mutated_chromosome = np.zeros(chromosome.shape)
        for idx, gene in enumerate(chromosome):
            rand_num = np.random.randn()
            if rand_num > probability_for_mutation: # mutation
                mutated_chromosome[idx] += np.random.randn()
            else: # non-mutation
                mutated_chromosome[idx] = gene

        return mutated_chromosome
        
    def __QLearning_mutation(self, chromosome, probability_for_mutation):
        mutated_chromosome = np.zeros(chromosome.shape)
        for idx, gene in enumerate(chromosome):
            rand_num = np.random.randn()
            if rand_num > probability_for_mutation: # mutation
                mut_val = np.random.randn()
                if mutated_chromosome[idx] + mut_val > 1:
                    mutated_chromosome[idx] = 1
                elif mutated_chromosome[idx] + mut_val < 0:
                    mutated_chromosome[idx] = 0
                else:
                    mutated_chromosome[idx] += mut_val
            else: # non-mutation
                mutated_chromosome[idx] = gene

        return mutated_chromosome

            
              
