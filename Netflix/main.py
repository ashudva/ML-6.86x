import numpy as np
import kmeans
import common
import naive_em
import em

# Parameters tuning K and seed
X = np.loadtxt("toy_data.txt")
costs = np.zeros((4,5))
for K in range(1,5):
    for seed in range(5):
        mixture, post = common.init(X, K, seed)
        mixture, post, costs[K-1,seed] = kmeans.run(X, mixture, post)
        # uncomment below code for plots
        # common.plot(X, mixture, post, f"mixtures when K = {K}, seed = {seed}")
# print minimum cost for different K = [1,2,3,4]
print(np.min(costs, axis=1))
