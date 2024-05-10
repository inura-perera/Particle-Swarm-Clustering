from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from particle import Particle
from sklearn.metrics import silhouette_score

class PSOClusteringSwarm:
    n_clusters = None
    n_particles = None
    data = None
    particles = []
    global_best_position = None
    global_best_value = np.inf
    global_best_clustering = None

    def _initialize_swarm(self, iteration, plot):
        print('*** Initializing swarm with', self.n_particles, 'PARTICLES,', self.n_clusters, 'CLUSTERS with', iteration,
              'MAX ITERATIONS and with PLOT =', plot, '***')
        print('Data=', self.data.shape[0], 'points in', self.data.shape[1], 'dimensions')

    def _generate_particles(self, inertia_weight: float, cognitive_coeff: float, social_coeff: float):
        for _ in range(self.n_particles):
            particle = Particle()
            particle.initialize(num_clusters=self.n_clusters, data=self.data, inertia_weight=inertia_weight, cognitive_coeff=cognitive_coeff, social_coeff=social_coeff)
            self.particles.append(particle)

    def update_global_best(self, particle):
        if particle.best_value < self.global_best_value:
            self.global_best_value = particle.best_value
            self.global_best_position = particle.best_position.copy()
            self.global_best_clustering = particle.best_clustering.copy()

    def start(self, iteration=1000, plot=False) -> Tuple[np.ndarray, float]:
        self._initialize_swarm(iteration, plot)
        progress = []

        for i in range(iteration):
            if i % 200 == 0:
                clusters = self.global_best_clustering
                print('iteration', i, 'GB =', self.global_best_value)
                print('best clusters so far = ', clusters)
                if plot:
                    centroids = self.global_best_position
                    if clusters is not None:
                        plt.scatter(self.data[:, 0], self.data[:, 1], c=clusters, cmap='viridis', marker='D')
                        plt.scatter(centroids[:, 0], centroids[:, 1], c='black', s=200, alpha=0.5)
                        plt.show()
                    else:
                        plt.scatter(self.data[:, 0], self.data[:, 1])
                        plt.show()

            for particle in self.particles:
                particle.update_best(data=self.data)
                self.update_global_best(particle=particle)

            for particle in self.particles:
                particle.move_centroids(global_best_position=self.global_best_position)
            progress.append([self.global_best_position, self.global_best_clustering, self.global_best_value])

        print('Finished!')
        return self.global_best_clustering, self.global_best_value
    
    def calculate_silhouette_score(self, data):
        clustering = np.array(self.global_best_clustering)
        silhouette_avg = silhouette_score(data, clustering)
        return silhouette_avg

def quantization_error(data, centroids):
    total_distance = 0
    for point in data:
        # Find the nearest centroid to the data point
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        nearest_centroid_index = np.argmin(distances)
        nearest_centroid = centroids[nearest_centroid_index]
        # Calculate the distance between the data point and its nearest centroid
        distance_to_nearest_centroid = np.linalg.norm(point - nearest_centroid)
        total_distance += distance_to_nearest_centroid
    # Calculate the average distance (quantization error)
    quantization_error = total_distance / len(data)
    return quantization_error


