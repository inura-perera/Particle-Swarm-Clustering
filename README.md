# Particle Swarm Clustering on the Iris Dataset

## Overview

This project implements a clustering algorithm using Particle Swarm Optimization (PSO) to analyze the Iris dataset. The project was developed as part of the "Swarm Intelligence" module at university, providing hands-on experience with swarm-based optimization techniques.

## Key Features

- **Particle Swarm Optimization (PSO)**: Utilizes a swarm of particles to find the optimal clustering of the Iris dataset.
- **Clustering Evaluation**: Measures clustering performance using quantization error and silhouette score.
- **Data Visualization**: Visualizes clustering progress and results for better analysis and insights.

## Technologies Used

- Python
- Numpy
- Pandas
- Matplotlib
- Scikit-learn

## Project Structure

- `main.py`: The main script to run the PSO clustering algorithm on the Iris dataset.
- `pso_clustering.py`: Contains the `PSOClusteringSwarm` class which implements the PSO algorithm.
- `particle.py`: Defines the `Particle` class representing individual particles in the swarm.

## How to Run

1. **Clone the repository**:
    ```bash
    git clone https://github.com/inura-perera/Particle-Swarm-Clustering.git
    cd Particle-Swarm-Clustering
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the Iris dataset**:
    Ensure you have the `iris.csv` file in the project directory. You can download it from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/iris).

4. **Run the script**:
    ```bash
    python main.py
    ```

## Results

- **Quantization Error**: The average distance of data points to their nearest centroids.
- **Silhouette Score**: Measures the cohesion and separation of clusters.


## Learnings and Insights

Gained hands-on experience with swarm intelligence and its applications in clustering. Learned to balance exploration and exploitation using inertia weight, cognitive, and social coefficients to enhance algorithm performance. Improved skills in visualizing clustering results for better data interpretation.

## Contributing

Feel free to fork this repository and contribute by submitting a pull request. For major changes, please open an issue first to discuss what you would like to change.

