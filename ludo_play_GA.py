import numpy as np
from selection import basetournement
from LudoPlayerGA import simple_GA_player


basetournement_test = basetournement(simple_GA_player, 20)

chromosomes = np.ones((20, 4))



basetournement_test.play_tournament(chromosomes, 100)