import pandas as pd
import numpy as np
from pso_clustering import PSOClusteringSwarm, quantization_error


plot = True
data_points = pd.read_csv('iris.csv', sep=',', header=None)
clusters = data_points[4].values
data_points = data_points.drop([4], axis=1)
if plot:
    data_points = data_points[[0, 1]]
data_points = data_points.values

n_clusters = 3
n_particles = 10
inertia_weight = 0.72
cognitive_coeff = 1.49
social_coeff = 1.49

pso = PSOClusteringSwarm()
pso.n_clusters = n_clusters
pso.n_particles = n_particles
pso.data = data_points
pso._generate_particles(inertia_weight, cognitive_coeff, social_coeff)

best_clustering, best_value = pso.start(iteration=1000, plot=plot)

# For showing the actual clusters
mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
clusters = np.array([mapping[x] for x in clusters])
print('Actual classes = ', clusters)

# Calculate and print the quantization error
qe = quantization_error(data_points, pso.global_best_position)
print("Quantization Error:", qe)

# Calculate and print the silhouette score
silhouette_score = pso.calculate_silhouette_score(data_points)
print("Silhouette Score:", silhouette_score)