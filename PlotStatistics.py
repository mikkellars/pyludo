import matplotlib.pyplot as plt
import matplotlib.style
import csv
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


