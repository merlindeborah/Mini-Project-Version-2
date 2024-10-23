import networkx as nx
import numpy as np
from sklearn.metrics.cluster import normalized_mutual_info_score
import matplotlib.pyplot as plt
from scipy.io import mmread

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
    
    print(f"Initial communities formed: {communities}")
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
    
    print(f"Communities after resolving overlaps: {final_communities}")
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
    
    print(f"Communities after merging: {final_communities}")
    return final_communities

# Main function to run the LCD-SN algorithm
def lcd_sn_algorithm(graph, alpha=0.7, beta=0.3, max_iter=6, mc=1.5):
    importance = calculate_imp(graph, alpha, beta, max_iter)
    print(f"Node importance: {importance}")
    
    initial_communities = form_initial_communities(graph, importance)
    communities = assign_overlapping_nodes(graph, initial_communities)
    final_communities = merge_communities(graph, communities, mc)
    
    all_nodes = set(graph.nodes())
    covered_nodes = set().union(*final_communities)
    missing_nodes = all_nodes - covered_nodes

    if missing_nodes:
        print(f"Missing nodes detected: {missing_nodes}")
        for node in missing_nodes:
            final_communities.append({node})
    
    return final_communities

# Calculate Modularity
def calculate_modularity(graph, communities):
    return nx.algorithms.community.quality.modularity(graph, communities)

# Visualize the final community graph
def visualize_communities(graph, communities):
    pos = nx.spring_layout(graph)
    colors = ['r', 'g', 'b', 'y', 'c', 'm']
    for i, community in enumerate(communities):
        nx.draw_networkx_nodes(graph, pos, nodelist=list(community), node_color=colors[i % len(colors)], label=f"Community {i+1}")
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_labels(graph, pos)
    plt.legend()
    plt.show()

# Load the Karate Club graph from the .mtx file
def load_graph_from_mtx(file_path):
    matrix = mmread(file_path)
    G = nx.from_scipy_sparse_array(matrix)
    return G

# Validate the communities detected
def validate_communities(graph, communities):
    all_nodes = set(graph.nodes())
    community_nodes = set().union(*communities)
    
    if all_nodes != community_nodes:
        print("Warning: Not all nodes are covered in the detected communities.")
        print(f"Graph nodes: {all_nodes}")
        print(f"Covered nodes: {community_nodes}")
        print(f"Missing nodes: {all_nodes - community_nodes}")

    non_empty_communities = [community for community in communities if community]
    if not non_empty_communities:
        print("Error: No valid communities found.")
        return False
    
    return True

# Example Usage
if __name__ == "__main__":
    # Load the graph from the karate.mtx file
    file_path = 'dolphins.mtx'  # Update this path if needed
    G = load_graph_from_mtx(file_path)
    
    # Run the LCD-SN algorithm
    communities = lcd_sn_algorithm(G)
    
    # Validate communities
    if validate_communities(G, communities):
        # Calculate modularity
        mod = calculate_modularity(G, communities)
        print(f"Modularity: {mod}")

        # Visualize the communities
        visualize_communities(G, communities)
    else:
        print("Community detection failed. Please check the input data and algorithm.")


