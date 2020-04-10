import matplotlib.pyplot as plt
import matplotlib.style
import csv
import pandas as pd

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

        # OLD #
        # x = []
        # y = []
        # with open(pathToCSV, 'r') as csvfile:
        #     plots = csv.reader(csvfile)
        #     for idx, reward in enumerate(plots):
        #         x.append(idx)
        #         y.append(int(reward[0]))

        # plt.plot(x,y)

        # plt.title('Game Reward over Time')
        # plt.xlabel('Number of Games')
        # plt.ylabel('Game Reward')

        # plt.show()

