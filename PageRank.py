
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define the adjacency matrix for the graph
adj_matrix = np.array([
	[0, 1, 0, 0],
	[0, 0, 1, 0],
	[1, 0, 0, 1],
	[0, 1, 0, 0]
])

# Number of nodes
n = len(adj_matrix)

# Damping factor
damping_factor = 0.85

# Initialize the PageRank values
pagerank = np.ones(n) / n
diff=1
# Transition matrix
transition_matrix= (1 - damping_factor) / n + damping_factor * (adj_matrix / adj_matrix.sum(axis=1, keepdims=True))
# Iterative computation of PageRank
i=0
while diff > 1e-6: 
	i+=1
	pagerank1 = np.dot(transition_matrix.T, pagerank)
	diff = np.linalg.norm(pagerank1 - pagerank)
	pagerank = pagerank1

# Output the PageRank values
print("PageRank values:")
print(pagerank)
# Output the number of iterations
print(f"Number of iterations: {i}")
