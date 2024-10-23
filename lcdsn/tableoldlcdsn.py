import networkx as nx
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
from scipy.io import mmread
import os

# Function to calculate the importance of nodes
def calculate_imp(graph, alpha=0.7, beta=0.3, max_iter=6):
    importance = {node: 1.0 for node in graph.nodes()}
    for _ in range(max_iter):
        new_importance = {}
        for node in graph.nodes():
            first_degree_neighbors = set(graph.neighbors(node))
            second_degree_neighbors = set()
            for neighbor in first_degree_neighbors:
                second_degree_neighbors.update(set(graph.neighbors(neighbor)))
            second_degree_neighbors -= first_degree_neighbors
            second_degree_neighbors.discard(node)

            imp_score = sum(alpha * (importance[neighbor] / len(list(graph.neighbors(neighbor)))) 
                            for neighbor in first_degree_neighbors)
            imp_score += sum(beta * (importance[neighbor] / len(list(graph.neighbors(neighbor)))) 
                             for neighbor in second_degree_neighbors)

            new_importance[node] = imp_score
        importance = new_importance
    return importance

# First Phase: Formation of Initial Communities
def form_initial_communities(graph, importance):
    communities = []
    sorted_nodes = sorted(importance, key=importance.get, reverse=True)
    visited = set()
    
    for node in sorted_nodes:
        if node not in visited:
            community = set([node]) | set(graph.neighbors(node))
            communities.append(community)
            visited.update(community)
    
    return communities

# Calculate similarity of node to a community
def calculate_similarity(node, community, graph):
    neighbors = set(graph.neighbors(node))
    community_nodes = community
    common_neighbors = neighbors & community_nodes
    return len(common_neighbors) / len(neighbors)  # Example similarity measure

# Second Phase: Determining Status of Overlapping Nodes
def assign_overlapping_nodes(graph, communities):
    node_to_communities = {}
    for community in communities:
        for node in community:
            if node in node_to_communities:
                node_to_communities[node].append(community)
            else:
                node_to_communities[node] = [community]

    final_communities = []
    assigned_nodes = set()

    for node, comms in node_to_communities.items():
        if node not in assigned_nodes:
            if len(comms) > 1:
                best_community = max(comms, key=lambda c: calculate_similarity(node, c, graph))
                for community in comms:
                    if community != best_community:
                        community.discard(node)
            assigned_nodes.add(node)
    
    final_communities = [community for community in communities if community]
    
    return final_communities

# Third Phase: Integration of Communities
def merge_communities(graph, communities, mc=1.5):
    # Merge small communities
    L = [c for c in communities if len(c) < 3]
    while L:
        node = next(iter(L))  # Get one community from L
        L.remove(node)
        if not communities:
            break
        best_community = max(communities, key=lambda c: calculate_similarity(next(iter(node)), c, graph))
        best_community.update(node)
    
    # Merge weak communities
    final_communities = []
    for community in communities:
        E_in = sum(1 for u in community for v in community if graph.has_edge(u, v))
        E_out = sum(1 for u in community for v in graph.nodes() if v not in community and graph.has_edge(u, v))
        if E_in <= mc * E_out:
            best_community = max(communities, key=lambda c: calculate_similarity(next(iter(community)), c, graph))
            best_community.update(community)
        else:
            final_communities.append(community)
    
    return final_communities

# Main function to run the LCD-SN algorithm
def lcd_sn_algorithm(graph, alpha=0.7, beta=0.3, max_iter=6, mc=1.5):
    importance = calculate_imp(graph, alpha, beta, max_iter)
    initial_communities = form_initial_communities(graph, importance)
    communities = assign_overlapping_nodes(graph, initial_communities)
    final_communities = merge_communities(graph, communities, mc)
    
    all_nodes = set(graph.nodes())
    covered_nodes = set().union(*final_communities)
    missing_nodes = all_nodes - covered_nodes

    if missing_nodes:
        for node in missing_nodes:
            final_communities.append({node})
    
    return final_communities

# Calculate Modularity
def calculate_modularity(graph, communities):
    return nx.algorithms.community.quality.modularity(graph, communities)

# Load the graph from .mtx or .txt file
def load_graph(file_path):
    if file_path.endswith('.mtx'):
        matrix = mmread(file_path)
        G = nx.from_scipy_sparse_array(matrix)
    elif file_path.endswith('.txt'):
        G = nx.read_edgelist(file_path)
    else:
        raise ValueError("Unsupported file format")
    return G

# Main script to compute modularity for all datasets
def main():
    dataset_files = [
        'karate.mtx', 'netscience.mtx', 'polbooks.mtx', 'football.mtx',
        'Email-Enron.txt', 'dolphins.mtx', 'CA-HepTh.txt', 'CA-HepPh.txt', 'CA-AstroPh.txt'
    ]

    for file_path in dataset_files:
        if os.path.exists(file_path):
            print(f"Processing {file_path}...")
            try:
                G = load_graph(file_path)
                communities = lcd_sn_algorithm(G)
                mod = calculate_modularity(G, communities)
                print(f"File: {file_path}, Modularity: {mod}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"File {file_path} does not exist.")

if __name__ == "__main__":
    main()
