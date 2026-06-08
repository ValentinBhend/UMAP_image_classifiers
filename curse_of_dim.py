import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

rng = np.random.default_rng(0)
n_points = 500
dims = [2, 3, 5, 10, 20, 50, 100, 200, 500, 1000]

nn_mean, all_mean, ratio = [], [], []
for d in dims:
    X = rng.random((n_points, d))          # uniform random points in the N-dim cube
    D = cdist(X, X)
    np.fill_diagonal(D, np.inf)
    nearest = D.min(axis=1)
    D[D == np.inf] = np.nan
    avg = np.nanmean(D, axis=1)
    nn_mean.append(nearest.mean())
    all_mean.append(avg.mean())
    ratio.append((nearest / avg).mean())

# Plot 1: distances grow
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.plot(dims, nn_mean, 'o-', label='nearest neighbor')
ax.plot(dims, all_mean, 's-', label='average over all points')
ax.set(xscale='log', xlabel='dimension N', ylabel='distance',
       title='Distances of random points in a N-dimensional cube')
ax.legend()
fig.tight_layout(); fig.savefig('curse_distances.png', dpi=130)

# Plot 2: distances concentrate
fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.plot(dims, ratio, 'o-', color='crimson')
ax.axhline(1, color='gray', ls='--', lw=1)
ax.set(xscale='log', xlabel='dimension N',
       ylabel='nearest distance / average distance', ylim=(0, 1.05),
       title='Distance ratio of nearest neighbors to average distance')
fig.tight_layout(); fig.savefig('curse_ratio.png', dpi=130)
print("done")