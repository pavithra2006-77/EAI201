from collections import deque
from queue import PriorityQueue
import math

print("Helping Terry the Rat Escape the Underground Pipes\n")

# ---------------- Graph & Coordinates ----------------
graph = {
    'J1': ['J2', 'J3'],
    'J2': ['J1', 'J4', 'J5'],
    'J3': ['J1', 'J6'],
    'J4': ['J2'],
    'J5': ['J2', 'J6'],
    'J6': ['J3', 'J5']
}

coordinates = {
    'J1': (0, 5),
    'J2': (1, 3),
    'J3': (4, 2),
    'J4': (0, 2),
    'J5': (3, 1),
    'J6': (5, 0)
}

start_node = 'J1'
goal_node = 'J6'

# ---------------- BFS ----------------
def bfs(graph, start, goal):
    queue = deque([[start]])
    visited = {start}
    nodes_explored = 0

    while queue:
        path = queue.popleft()
        node = path[-1]
        nodes_explored += 1

        if node == goal:
            return path, len(path)-1, nodes_explored

        for neighbor in graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(path + [neighbor])

    return None, 0, nodes_explored

# ---------------- DFS ----------------
def dfs(graph, start, goal):
    stack = [(start, [start])]
    nodes_explored = 0

    while stack:
        node, path = stack.pop()
        nodes_explored += 1

        if node == goal:
            return path, len(path)-1, nodes_explored

        for neighbor in reversed(graph.get(node, [])):
            if neighbor not in path:
                stack.append((neighbor, path + [neighbor]))

    return None, 0, nodes_explored

# ---------------- A* ----------------
def heuristic(node, goal):
    x1, y1 = coordinates[node]
    x2, y2 = coordinates[goal]
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def a_star(graph, start, goal):
    pq = PriorityQueue()
    pq.put((0, start))
    came_from = {start: None}
    cost_so_far = {start: 0}
    nodes_explored = 0

    while not pq.empty():
        _, current = pq.get()
        nodes_explored += 1

        if current == goal:
            path = []
            while current:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path, cost_so_far[goal], nodes_explored

        for neighbor in graph.get(current, []):
            edge_cost = 1  # assuming each pipe has cost 1
            new_cost = cost_so_far[current] + edge_cost
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                pq.put((priority, neighbor))
                came_from[neighbor] = current

    return None, float('inf'), nodes_explored

# ---------------- Run All Algorithms ----------------
algorithms = [("Breadth-First Search", bfs),
              ("Depth-First Search", dfs),
              ("A* Search", a_star)]

for name, func in algorithms:
    print(f"\n--- {name} ---")
    path, cost, explored = func(graph, start_node, goal_node)
    if path:
        print(f"Terry starts at {start_node} and wants to reach the cheese at {goal_node}!")
        print("He moves through the junctions in this order: " + " -> ".join(path))
        print(f"Total effort/cost: {cost}")
        print(f"Total junctions in path: {len(path)}")
        print(f"Total junctions explored: {explored}")
    else:
        print("Oh no! Terry could not find a path to the cheese.")

# ---------------- Conclusion ----------------
print("\n--- Conclusion ---\n")
print("After running BFS, DFS, and A* algorithms, we can make the following observations:")

print("\n1. Breadth-First Search (BFS):")
print("- Explores all nearby junctions first.")
print("- Finds the shortest path in terms of number of junctions but ignores cost.")
print("- Explored more junctions than A*.")

print("\n2. Depth-First Search (DFS):")
print("- Goes deep along one path before backtracking.")
print("- Path may be longer and less efficient.")
print("- Explores junctions in order but may not find the cheapest route.")

print("\n3. A* Search (A*):")
print("- Uses a heuristic to guide the search.")
print("- Finds the cheapest path while exploring fewer junctions.")
print("- Most efficient for helping Terry reach the cheese.")

print("\nOverall Observations:")
print("- BFS is good for shortest steps, DFS is quick but unpredictable.")
print("- A* is the best choice as it balances cost and efficiency, helping Terry reach the cheese safely and quickly.")
