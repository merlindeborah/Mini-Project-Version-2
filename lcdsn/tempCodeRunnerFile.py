import networkx as nx
import numpy as np
from community import community_louvain
from scipy.io import mmread
from scipy.sparse import coo_matrix

# Load the dataset
def load_graph_from_mtx(filename):
    # Read the matrix
    matrix = mmread(filename).tocoo()  # Read the .mtx file into a COO matrix
    
    # Create a NetworkX graph from the COO matrix
    graph = nx.Graph()
    
    # Add edges to the graph
    for i, j, value in zip(matrix.row, matrix.col, matrix.data):
        graph.add_edge(i, j, weight=value)
    
    return graph

def calculate_imp(graph, alpha=0.7, beta=0.3, gamma=6):
    imp = {node: 1.0 for node in graph.nodes}  # Initialize importance
    for _ in range(gamma):  # Iterate to compute importance values
        new_imp = {}
        for node in graph.nodes:
            first_neighbors = list(graph.neighbors(node))
            second_neighbors = set()
            for neighbor in first_neighbors:
                second_neighbors.update(graph.neighbors(neighbor))
            
            imp_value = 0
            for neighbor in first_neighbors:
                if graph.has_edge(node, neighbor):
                    weight = graph[node][neighbor].get('weight', 1)
                    neighbor_weight_sum = sum(
                        graph[neighbor][n].get('weight', 1) for n in graph.neighbors(neighbor)
                        if graph.has_edge(neighbor, n)
                    )
                    if neighbor_weight_sum > 0:
                        imp_value += alpha * weight * imp[neighbor] / neighbor_weight_sum
            
            for second_neighbor in second_neighbors:
                if second_neighbor != node and not graph.has_edge(node, second_neighbor):
                    if graph.has_edge(node, second_neighbor):
                        weight = graph[node][second_neighbor].get('weight', 1)
                        second_neighbor_weight_sum = sum(
                            graph[second_neighbor][n].get('weight', 1) for n in graph.neighbors(second_neighbor)
                            if graph.has_edge(second_neighbor, n)
                        )
                        if second_neighbor_weight_sum > 0:
                            imp_value += beta * weight * imp[second_neighbor] / second_neighbor_weight_sum
            
            new_imp[node] = imp_value
        
        imp = new_imp
    return imp

def form_initial_communities(graph, imp):
    sorted_nodes = sorted(imp, key=imp.get, reverse=True)
    communities = []
    assigned = set()
    
    for node in sorted_nodes:
        if node not in assigned:
            community = {node}
            community.update(graph.neighbors(node))
            communities.append(community)
            assigned.update(community)
    
    return communities

def glhn_similarity(graph, node1, node2):
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    shared_neighbors = neighbors1.intersection(neighbors2)
    return len(shared_neighbors) / (len(neighbors1) * len(neighbors2)) ** 0.5

def calculate_similarity(graph, node, community):
    return sum(glhn_similarity(graph, node, member) for member in community)

def resolve_overlaps(graph, communities):
    final_communities = []
    for community in communities:
        overlaps = set()
        for node in community:
            for other_community in communities:
                if node in other_community and community != other_community:
                    overlaps.add(node)
        
        for overlap_node in overlaps:
            best_community = max(communities, key=lambda c: calculate_similarity(graph, overlap_node, c))
            if best_community != community:
                community.remove(overlap_node)
        
        final_communities.append(community)
    
    return final_communities

def merge_small_weak_communities(graph, communities, mc=4):
    merged_communities = []
    for community in communities:
        internal_edges = sum(1 for node in community for neighbor in graph.neighbors(node) if neighbor in community)
        external_edges = sum(1 for node in community for neighbor in graph.neighbors(node) if neighbor not in community)
        
        if internal_edges <= mc * external_edges:
            best_community = max(communities, key=lambda c: calculate_similarity(graph, list(community)[0], c))
            merged_communities.append(best_community)
        else:
            merged_communities.append(community)
    
    return merged_communities

def calculate_modularity(graph, communities):
    partition = {}
    for idx, community in enumerate(communities):
        for node in community:
            partition[node] = idx
    modularity = community_louvain.modularity(partition, graph)
    return modularity

def lcd_sn_algorithm(graph, alpha=0.7, beta=0.3, gamma=6, mc=4):
    imp = calculate_imp(graph, alpha, beta, gamma)
    communities = form_initial_communities(graph, imp)
    communities = resolve_overlaps(graph, communities)
    final_communities = merge_small_weak_communities(graph, communities, mc)
    modularity_value = calculate_modularity(graph, final_communities)
    return final_communities, modularity_value

# Example usage
if __name__ == '__main__':
    # Load the graph from the .mtx file
    G = load_graph_from_mtx('netscience.mtx')
    
    # Run LCD-SN Algorithm
    final_communities, modularity_value = lcd_sn_algorithm(G)
    
    # Output final communities
    for idx, community in enumerate(final_communities):
        print(f"Community {idx + 1}: {sorted(community)}")
    
    # Output modularity value
    print(f"Modularity: {modularity_value}")
