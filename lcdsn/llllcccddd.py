import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score as NMI
from scipy.io import mmread

# Load the dataset from a .mtx file
def load_graph_from_mtx(filename):
    matrix = mmread(filename).tocoo()  # Read the .mtx file into a COO matrix
    graph = nx.Graph()
    for i, j, value in zip(matrix.row, matrix.col, matrix.data):
        graph.add_edge(i, j, weight=value)
    return graph

# Step 1: Calculate Node Importance (IMP)
def calculate_imp(graph, alpha=0.7, beta=0.3, max_iter=6):
    imp = {node: 1.0 for node in graph.nodes}  # Initialize importance values
    
    for _ in range(max_iter):  # Iterate to compute importance values
        new_imp = {}
        for node in graph.nodes:
            first_neighbors = list(graph.neighbors(node))  # First-degree neighbors
            second_neighbors = set()
            for neighbor in first_neighbors:
                second_neighbors.update(graph.neighbors(neighbor))
            second_neighbors.discard(node)  # Remove self-loops

            imp_value = 0
            
            # Contribution of first-degree neighbors
            for neighbor in first_neighbors:
                weight = graph[node][neighbor].get('weight', 1)
                neighbor_weight_sum = sum(graph[neighbor][n].get('weight', 1) for n in graph.neighbors(neighbor))
                if neighbor_weight_sum > 0:
                    imp_value += alpha * weight * imp[neighbor] / neighbor_weight_sum

            # Contribution of second-degree neighbors
            for second_neighbor in second_neighbors:
                weight = graph[node][second_neighbor].get('weight', 1) if graph.has_edge(node, second_neighbor) else 1
                second_neighbor_weight_sum = sum(
                    graph[second_neighbor][n].get('weight', 1) for n in graph.neighbors(second_neighbor)
                )
                if second_neighbor_weight_sum > 0:
                    imp_value += beta * weight * imp[second_neighbor] / second_neighbor_weight_sum
            
            new_imp[node] = imp_value
        
        imp = new_imp
    
    return imp

# Step 2: Form Initial Communities
def form_initial_communities(graph, node_importance):
    sorted_nodes = sorted(node_importance, key=node_importance.get, reverse=True)
    communities = []
    assigned_nodes = set()
    
    for node in sorted_nodes:
        if node not in assigned_nodes:
            community = {node}
            community.update(graph.neighbors(node))  # Add all first-degree neighbors to the community
            communities.append(community)
            assigned_nodes.update(community)
    
    return communities

# Step 3: Resolve Overlapping Nodes using Similarity (GLHN index)
def glhn_similarity(graph, node1, node2):
    neighbors1 = set(graph.neighbors(node1))
    neighbors2 = set(graph.neighbors(node2))
    shared_neighbors = neighbors1 & neighbors2
    return len(shared_neighbors) / (len(neighbors1) * len(neighbors2))

def resolve_overlaps(graph, communities):
    node_to_community = {}
    
    # Assign nodes to the community they belong to
    for comm_id, community in enumerate(communities):
        for node in community:
            if node not in node_to_community:
                node_to_community[node] = comm_id
            else:
                # Resolve overlaps by comparing similarity
                current_comm_id = node_to_community[node]
                current_similarity = sum(glhn_similarity(graph, node, n) for n in communities[current_comm_id])
                new_similarity = sum(glhn_similarity(graph, node, n) for n in communities[comm_id])
                if new_similarity > current_similarity:
                    node_to_community[node] = comm_id

    # Rebuild communities based on final assignments
    final_communities = {comm_id: set() for comm_id in range(len(communities))}
    for node, comm_id in node_to_community.items():
        final_communities[comm_id].add(node)

    return [comm for comm in final_communities.values() if comm]  # Filter out empty communities

# Step 4: Merge Small and Weak Communities
def merge_small_communities(graph, communities, mc=4):
    merged_communities = communities.copy()
    
    while True:
        # Calculate internal and external edges for each community
        weak_communities = []
        for community in merged_communities:
            internal_edges = sum(1 for node in community for neighbor in graph.neighbors(node) if neighbor in community)
            external_edges = sum(1 for node in community for neighbor in graph.neighbors(node) if neighbor not in community)
            
            # If a community is weak, mark it for merging
            if internal_edges <= mc * external_edges:
                weak_communities.append(community)
        
        # Stop if no weak communities are found
        if not weak_communities:
            break
        
        # Merge weak communities with the most similar neighboring community
        for weak_community in weak_communities:
            best_merge_candidate = None
            best_merge_similarity = -1
            for strong_community in merged_communities:
                if weak_community != strong_community:
                    similarity = sum(glhn_similarity(graph, node1, node2)
                                     for node1 in weak_community for node2 in strong_community)
                    if similarity > best_merge_similarity:
                        best_merge_similarity = similarity
                        best_merge_candidate = strong_community
            if best_merge_candidate:
                best_merge_candidate.update(weak_community)
                merged_communities.remove(weak_community)
    
    return merged_communities

# Step 5: Calculate Modularity
def calculate_modularity(graph, communities):
    m = graph.size(weight='weight')
    modularity = 0
    for community in communities:
        for u in community:
            for v in community:
                A_uv = graph[u][v].get('weight', 1) if graph.has_edge(u, v) else 0
                modularity += A_uv - (graph.degree[u] * graph.degree[v]) / (2 * m)
    return modularity / (2 * m)

# Step 6: Evaluate NMI
def calculate_nmi(true_labels, predicted_labels):
    return NMI(true_labels, predicted_labels)

# LCD-SN Algorithm
def lcd_sn_algorithm(graph, true_labels=None):
    # Step 1: Calculate Node Importance
    node_importance = calculate_imp(graph)
    
    # Step 2: Form Initial Communities
    initial_communities = form_initial_communities(graph, node_importance)
    
    # Step 3: Resolve Overlapping Nodes
    resolved_communities = resolve_overlaps(graph, initial_communities)
    
    # Step 4: Merge Small and Weak Communities
    final_communities = merge_small_communities(graph, resolved_communities)
    
    # Step 5: Calculate Modularity
    modularity_value = calculate_modularity(graph, final_communities)
    
    # Step 6: Calculate NMI if true labels are provided
    if true_labels:
        predicted_labels = [-1] * len(graph.nodes)
        for comm_id, community in enumerate(final_communities):
            for node in community:
                predicted_labels[node] = comm_id
        nmi_value = calculate_nmi(true_labels, predicted_labels)
    else:
        nmi_value = None

    return final_communities, modularity_value, nmi_value

# Example Usage
if __name__ == '__main__':
    # Load the graph from the .mtx file
    G = load_graph_from_mtx('dolphins.mtx')  # Update with your dataset
    
    # Ensure true_labels matches the number of nodes in the graph
    num_nodes = G.number_of_nodes()
    # Example placeholder: Assign all nodes to the same community (0)
    true_labels = [0] * num_nodes
    
    # Run LCD-SN Algorithm
    final_communities, modularity_value, nmi_value = lcd_sn_algorithm(G, true_labels=true_labels)
    
    # Output final communities
    for idx, community in enumerate(final_communities):
        print(f"Community {idx + 1}: {sorted(community)}")
    
    # Output modularity value
    print(f"Modularity: {modularity_value}")
    
    # Output NMI value
    if nmi_value is not None:
        print(f"NMI: {nmi_value}")
