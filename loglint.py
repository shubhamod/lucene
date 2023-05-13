import collections
import json
import sys

import networkx as nx

def print_shortest_path(L, start, end):
    shortest_path = nx.shortest_path(L, source=start, target=end)
    print(' -> '.join(map(str, shortest_path)))

def print_level_digraphs(level_digraphs):
    for level, graph in sorted(level_digraphs.items()):
        print(f"# Level {level}")
        for from_node, to_nodes in graph.adj.items():
            to_nodes_str = ' '.join(map(str, to_nodes))
            print(f"  {from_node} ->  {to_nodes_str}")
        print()
        
def reconstruct_graph(json_log):
    G = collections.defaultdict(nx.DiGraph)
    top_level = {}
    active_level = {}
    for entry in json_log:
        op = entry["op"]

        if op == "begin":
            node_id = entry["node_id"]
            level = entry["level"]
            top_level[node_id] = level
            G[level].add_node(node_id)

        elif op == "finish":
            node_id = entry["node_id"]
            for i in range(active_level[node_id], top_level[node_id]):
                L = G[i]
                if not nx.has_path(L, 0, node_id):
                    print(f"Node {node_id} is not reachable from node 0 at level {i}")

        elif op == "link_back":
            level = entry["level"]
            from_nodes = entry.get("from_nodes")
            to_node = entry.get("to_node")
            active_level[to_node] = level
            for from_node in from_nodes:
                G[level].add_edge(from_node, to_node)

        elif op == "link_forwards":
            level = entry["level"]
            from_node = entry.get("from_node")
            to_nodes = entry.get("to_nodes")
            active_level[from_node] = level
            for to_node in to_nodes:
                G[level].add_edge(from_node, to_node)

        elif op == "unlink":
            removed_node = entry["removed_node"]
            from_node = entry["from_node"]
            G[active_level[from_node]].remove_edge(from_node, removed_node)

    for level, L in G.items():
        print(f"Checking level {level} of size {L.size()}")
        if not nx.is_strongly_connected(L):
            print(f"Level {L} is not connected")

    print_level_digraphs(G)
    print_shortest_path(G[0], 0, 340)

json_log_str = sys.stdin.read()
json_log = json.loads(json_log_str)
reconstruct_graph(json_log)
