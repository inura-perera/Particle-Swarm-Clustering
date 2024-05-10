import numpy as np

class Particle:
    def __init__(self):
        self.num_clusters = None
        self.centroids = None
        self.best_value = np.inf
        self.best_position = None
        self.velocity = None
        self.best_clustering = None
        self.inertia_weight = None
        self.cognitive_coeff = None
        self.social_coeff = None

    def initialize(self, num_clusters, data, inertia_weight, cognitive_coeff, social_coeff):
        self.num_clusters = num_clusters
        self.centroids = data[np.random.choice(list(range(len(data))), self.num_clusters)]
        self.best_position = self.centroids.copy()
        self.velocity = np.zeros_like(self.centroids)
        self.best_clustering = None
        self.inertia_weight = inertia_weight
        self.cognitive_coeff = cognitive_coeff
        self.social_coeff = social_coeff

    def update_best(self, data: np.ndarray):
        distances = self._calculate_distances(data=data)
        clusters = np.argmin(distances, axis=0)

        new_value = self._evaluate_fitness(clusters=clusters, distances=distances)
        if new_value < self.best_value:
            self.best_value = new_value
            self.best_position = self.centroids.copy()
            self.best_clustering = clusters.copy()

    def move_centroids(self, global_best_position):
        self.update_velocity(global_best_position=global_best_position)
        new_position = self.centroids + self.velocity
        self.centroids = new_position.copy()

    def update_velocity(self, global_best_position):
        self.velocity = self.inertia_weight * self.velocity + \
                        self.cognitive_coeff * np.random.random() * (self.best_position - self.centroids) + \
                        self.social_coeff * np.random.random() * (global_best_position - self.centroids)

    def _calculate_distances(self, data: np.ndarray) -> np.ndarray:
        distances = []
        for centroid in self.centroids:
            d = np.linalg.norm(data - centroid, axis=1)
            distances.append(d)
        distances = np.array(distances)
        return distances

    def _evaluate_fitness(self, clusters: np.ndarray, distances: np.ndarray) -> float:
        total_distance = 0.0
        for i in range(self.num_clusters):
            points_indices = np.where(clusters == i)[0]
            if len(points_indices):
                total_distance += sum(distances[i][points_indices]) / len(points_indices)
        return total_distance / self.num_clusters
