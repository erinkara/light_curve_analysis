import numpy as np
from scipy.stats import binned_statistic

cross = np.random.rand(100)
f = np.arange(100)
bins = np.logspace(0., 2., 10)
bin_means = binned_statistic(f, cross, bins=bins)[0]

print bin_means
