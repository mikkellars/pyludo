import matplotlib.pyplot as plt
import matplotlib.style
import csv
import numpy as np
import pandas as pd
import glob
import math
import re


matplotlib.style.use('ggplot') 

class PlotStatistics:
    def __init__(self):
        pass

    def plotReward(self, pathToCSV, numMovAvg):
        data = pd.read_csv(pathToCSV)
        data = data.rolling(window=numMovAvg).mean()

        plt.plot(data)

        plt.title('Game Reward over Time')
        plt.xlabel('Number of Games')
        plt.ylabel('Game Reward')
        plt.show()

    def plotMultiple(self, pathToFolder, numMovAvg):
        reward_files = glob.glob(pathToFolder + "/Reward*.csv")

        if len(reward_files) == 0:
            print("No reward.csv files found")


        for n in range(math.ceil(len(reward_files)/4)): # Plot 4 rewards in one figure
            row = 0
            col = 0
            temp_files = reward_files[n*4:n*4+4]
            fig, axs = plt.subplots(2, 2)
            plt.figure(n+1)

            for i, path in enumerate(temp_files):
                if i%2 == 0 and i > 0:
                    row = row + 1

                data = pd.read_csv(path)
                data = data.rolling(window=numMovAvg).mean()
                axs[row, col].plot(data)
                # Regex for getting title
                title = re.findall("(?<=Reward_)(.*?)(?=.csv)", path) 
                axs[row, col].set_title(title)
    
                col = col + 1
                if col > 1:
                    col = 0
            
            fig.tight_layout()
        plt.show()

    def plot_chromosome_3D(self, path_to_csv):
        # chromosomes_generations = pd.read_csv(path_to_csv)
        # print(np.array(chromosomes_generations)[:,0])
       

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        gen_chromosomes = []

        with open(path_to_csv, mode='r') as csv_file:        
            for i, row in enumerate(csv_file):
                tmp_chromosomes = []
                if i > 3:
                    print("Error more than 3 generations check csv file")

                for j in range(len(row)): # all chromosome from one generations
                    if row[j] == '[':
                        r = np.fromstring(row[1+j:-1], sep=' ')
                        tmp_chromosomes.append(r)

                gen_chromosomes.append(tmp_chromosomes)
        gen_chromosomes = np.array(gen_chromosomes)
        print(len(gen_chromosomes[2][:,0]))
        ax.scatter(gen_chromosomes[0][:,0], gen_chromosomes[0][:,1], gen_chromosomes[0][:,2], marker='o', color=['red']) # generation 1 (x,y,z)
        ax.scatter(gen_chromosomes[1][:,0], gen_chromosomes[1][:,1], gen_chromosomes[1][:,2], marker='^', color=['blue']) # generation 2 (x,y,z)
        ax.scatter(gen_chromosomes[2][:,0], gen_chromosomes[2][:,1], gen_chromosomes[2][:,2], marker='*', color=['green']) # generation 3 (x,y,z)

        #for gen_chrom in gen_chromosomes:1, 1.5, 2.5, 3, 3.5, 6.5, 5, 6, 7, 8, 7.5 
           # ax.scatter(gen_chrom[:,0], gen_chrom[:,1], gen_chrom[:,2])

        ax.set_xlabel(r'$\epsilon$')
        ax.set_ylabel(r'$\gamma$')
        ax.set_zlabel(r'$\alpha$')

        plt.show()

    def plot_chromosome_2D(self, path_to_csv):
        gen_chromosomes = []

        with open(path_to_csv, mode='r') as csv_file:        
            for i, row in enumerate(csv_file):
                tmp_chromosomes = []
                for j in range(len(row)): # all chromosome from one generations
                    if row[j] == '[':
                        r = np.fromstring(row[1+j:-1], sep=' ')
                        tmp_chromosomes.append(r)

                gen_chromosomes.append(tmp_chromosomes)
        
        gen_chromosomes = np.array(gen_chromosomes)

        #fig, ax = plt.subplots(4, 2)

        plot_idx = 0
        prev = 0

        for chrom_i in range((len(gen_chromosomes[0][0]))):
            for chrom_j in range((len(gen_chromosomes[0][0]))):
                fig = plt.figure(plot_idx)
                ax = fig.add_subplot()
                if chrom_i >= chrom_j:
                    continue
                # ax[plot_idx].axis('equal')
                ax.scatter(gen_chromosomes[0][:,chrom_i], gen_chromosomes[0][:,chrom_j], marker='o', color=['red']) # generation 1
                ax.scatter(gen_chromosomes[1][:,chrom_i], gen_chromosomes[1][:,chrom_j], marker='^', color=['blue']) # generation 2
                ax.scatter(gen_chromosomes[2][:,chrom_i], gen_chromosomes[2][:,chrom_j], marker='*', color=['green']) # generation 3
                ax.set_xlabel(r'$\theta$%s' %(chrom_i+1))
                ax.set_ylabel(r'$\theta$%s' %(chrom_j+1))

                if prev == chrom_i:
                    plot_idx += 1

        plt.show()