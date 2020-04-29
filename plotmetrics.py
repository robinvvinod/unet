# yapf: disable
import matplotlib.pyplot as plt
import numpy as np
from hyperparameters import log_path

epoch, metric, loss = np.loadtxt(log_path, delimiter=',', unpack=True, skiprows=1)
fig = plt.figure(figsize=(20, 10))
ax = fig.add_axes([1, 1, 1, 1])
# ax.set_yscale('log')
ax.plot(epoch, metric)
