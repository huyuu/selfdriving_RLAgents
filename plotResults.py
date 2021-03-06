import numpy as np
from matplotlib import pyplot as pl
import pickle


modelDirPath = './savedModels/ReinforcementKeras'
with open(f'{modelDirPath}/record.pickle', 'rb') as file:
    results = pickle.load(file)

results = np.array(results)
pl.plot(range(results.shape[0]), results[:, 1], label='rewards')
pl.plot(range(results.shape[0]), results[:, 2], label='averaged rewards')
pl.xlabel('episode')
pl.ylabel('rewards')
pl.legend()
pl.show()

pl.plot(range(results.shape[0]), results[:, 3], label='losses')
pl.xlabel('episode')
pl.ylabel('losses')
pl.legend()
pl.show()
