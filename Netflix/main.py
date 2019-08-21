import numpy as np
import kmeans
import common
import naive_em
import em

# Parameters tuning K and seed
X = np.loadtxt("toy_data.txt")

#---------------------------
#  Part 1 : Using kmeans
# --------------------------

costs = np.zeros((4,5))
for K in range(1,5):
    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, costs[K-1,seed] = kmeans.run(X, mixture, post)
        common.plot(X, mixture, post, f"mixtures when K = {K}, seed = {seed}")
# print minimum cost for different K = [1,2,3,4]
print(np.min(costs, axis=1))

#---------------------------
#  Part 2 : Using naive_em
# --------------------------

log_lh = np.zeros((4, 5))
for K in range(1, 5):
    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, log_lh[K-1, seed] = naive_em.run(X, mixture, post)
        common.plot(X, mixture, post, f"mixtures when K = {K}, seed = {seed}")
# print maximum log likelihood for different K = [1,2,3,4]
print(np.max(log_lh, axis=1))
