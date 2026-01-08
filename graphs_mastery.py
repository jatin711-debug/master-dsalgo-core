# ============================================================================
# MODULE 8: GRAPHS
# ============================================================================
"""
Nodes and Edges. Relationships.

COMMON PATTERNS:
1. BFS: Shortest Path (Unweighted).
2. DFS: Connectivity, Islands, Cycles.
3. Union-Find: Disjoint sets, Cycle detection (Undirected).
4. Topological Sort: Dependencies (DAG).
5. Dijkstra: Shortest Path (Weighted non-negative).
6. MST (Prim/Kruskal): Minimum cost to connect all nodes.

TIME COMPLEXITY: O(V + E) usually. O(E log V) for Heap based (Dijkstra/Prim).
"""

from collections import deque, defaultdict
import heapq

# Helper UnionFind Class
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.count = n # Number of components
        
    def find(self, p):
        if self.parent[p] != p:
            # Path compression
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]
        
    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            # Union by rank
            if self.rank[rootP] > self.rank[rootQ]:
                self.parent[rootQ] = rootP
            elif self.rank[rootP] < self.rank[rootQ]:
                self.parent[rootP] = rootQ
            else:
                self.parent[rootQ] = rootP
                self.rank[rootP] += 1
            self.count -= 1
            return True
        return False

# ============================================================================
# PATTERN 1: DFS (ISLANDS)
# ============================================================================
def num_islands(grid):
    """
    LeetCode #200: Number of Islands
    
    Approach: DFS or BFS.
    - Iterate through grid.
    - If '1' found, increment count and sink (turn to '0') all connected '1's.
    """
    if not grid: return 0
    rows, cols = len(grid), len(grid[0])
    count = 0
    
    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or grid[r][c] == '0'):
            return
        
        grid[r][c] = '0' # Sink
        dfs(r+1, c)
        dfs(r-1, c)
        dfs(r, c+1)
        dfs(r, c-1)
        
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)
    return count

# ============================================================================
# PATTERN 2: TOPOLOGICAL SORT (KAHN'S ALGORITHM)
# ============================================================================
def can_finish(num_courses, prerequisites):
    """
    LeetCode #207: Course Schedule
    
    Approach: Kahn's Algorithm (BFS)
    - Count in-degrees (dependencies) for each node.
    - Queue all nodes with 0 in-degree.
    - Process queue, reducing in-degree of neighbors.
    - If in-degree becomes 0, add to queue.
    - If processed count == Total nodes, possible. Else cycle.
    """
    adj = defaultdict(list)
    in_degree = [0] * num_courses
    
    for dest, src in prerequisites:
        adj[src].append(dest)
        in_degree[dest] += 1
        
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    processed_count = 0
    
    while queue:
        node = queue.popleft()
        processed_count += 1
        
        for neighbor in adj[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
                
    return processed_count == num_courses

# ============================================================================
# PATTERN 3: DIJKSTRA (SHORTEST PATH)
# ============================================================================
def network_delay_time(times, n, k):
    """
    LeetCode #743: Network Delay Time
    
    Approach: Dijkstra's Algorithm
    - Priority Queue: (time, node).
    - Map dist: stores min time to reach node.
    """
    adj = defaultdict(list)
    for u, v, w in times:
        adj[u].append((v, w))
        
    pq = [(0, k)] # (time, node)
    visited = set()
    max_time = 0
    
    while pq:
        time, node = heapq.heappop(pq)
        
        if node in visited:
            continue
        visited.add(node)
        max_time = max(max_time, time)
        
        for neighbor, weight in adj[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (time + weight, neighbor))
                
    return max_time if len(visited) == n else -1

# ============================================================================
# PATTERN 4: MINIMUM SPANNING TREE (MST - PRIM'S/KRUSKAL'S)
# ============================================================================
def min_cost_connect_points(points):
    """
    LeetCode #1584: Min Cost to Connect All Points
    
    Approach: Prim's Algorithm
    - Start from arbitrary point 0.
    - Heap stores (cost, point_index).
    - Add neighbors with Manhattan distance cost.
    - Add cheapest edge that connects to unvisited set.
    """
    n = len(points)
    pq = [(0, 0)] # cost, point_index
    visited = set()
    total_cost = 0
    edges_used = 0
    
    while edges_used < n:
        cost, u = heapq.heappop(pq)
        
        if u in visited:
            continue
            
        visited.add(u)
        total_cost += cost
        edges_used += 1
        
        for v in range(n):
            if v not in visited:
                dist = abs(points[u][0] - points[v][0]) + abs(points[u][1] - points[v][1])
                heapq.heappush(pq, (dist, v))
                
    return total_cost

if __name__ == "__main__":
    print("Testing Graphs:")
    print("-" * 30)
    grid = [["1","1","0"],["0","1","0"],["0","0","0"]]
    print("1. Islands:", num_islands(grid))
    print("2. Course Schedule (2, [[1,0]]):", can_finish(2, [[1,0]]))
    print("3. Network Delay (dijkstra):", network_delay_time([[2,1,1],[2,3,1],[3,4,1]], 4, 2))
    print("4. MST Cost ([[0,0],[2,2],[3,10],[5,2],[7,0]]):", min_cost_connect_points([[0,0],[2,2],[3,10],[5,2],[7,0]]))
