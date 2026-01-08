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
    """
    Standard Union-Find (Disjoint Set) data structure with Path Compression and Union by Rank.
    Used for: Connected components, Cycle detection in undirected graphs, Kruskal's Algorithm.
    """
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n
        self.count = n # Number of components
        
    def find(self, p):
        if self.parent[p] != p:
            # Path compression: Point directly to root
            self.parent[p] = self.find(self.parent[p])
        return self.parent[p]
        
    def union(self, p, q):
        rootP = self.find(p)
        rootQ = self.find(q)
        if rootP != rootQ:
            # Union by rank: Attach smaller tree to larger tree
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
    Given a grid of '1's (land) and '0's (water), count the number of islands.
    
    APPROACH:
    - Iterate through every cell.
    - If we find a '1':
      - We found a new island. Increment count.
      - Start a traversal (DFS or BFS) from that cell.
      - "Sink" the island: Mark every connected '1' as '0' (visited) so we don't count it again.
    
    WHY IT WORKS:
    - Every island is counted exactly once at its first discovered cell. The "sinking" process ensures no other cell of that island triggers a count.
    
    TIME COMPLEXITY: O(R * C)
    - We visit every cell once in the loop.
    - DFS visits each '1' cell a constant number of times (4 edges).
    
    SPACE COMPLEXITY: O(R * C)
    - Recursion stack in worst case (grid full of land).
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
    Can you finish all courses given prerequisite dependencies? (Detect Cycle in Directed Graph).
    
    APPROACH:
    - Kahn's Algorithm (BFS based).
    - 1. Build Adjacency List and In-Degree array (count of incoming edges).
    - 2. Enqueue all nodes with In-Degree 0 (Courses with no prereqs).
    - 3. Process Queue:
         - For each neighbor of a processed node, decrement its In-Degree.
         - If In-Degree becomes 0, enqueue it.
    - 4. If processed count == total courses, Success. Else, Cycle exists.
    
    WHY IT WORKS:
    - Nodes in a cycle will never reach In-Degree 0, so they (and their dependents) will never be added to the queue.
    
    TIME COMPLEXITY: O(V + E)
    - Visit every node and edge once.
    
    SPACE COMPLEXITY: O(V + E)
    - Adjacency list and queue.
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
    Find min time for signal to reach all nodes from k. (Shortest Path from Source).
    
    APPROACH:
    - Dijkstra's Algorithm.
    - Use a Min-Priority Queue to explore nodes by "current known shortest distance".
    - `visited` set to keep track of finalized nodes.
    - Pop node with smallest time. If not visited, process neighbors.
    - Neighbors: new_time = current_time + edge_weight. Push to PQ.
    
    WHY IT WORKS:
    - Greedy: Always expand the closest reachable node. Since weights are non-negative, once we extract a node from PQ, its shortest path is finalized.
    
    TIME COMPLEXITY: O(E * log V)
    - Each edge might be added to the heap once. Heap operations take log V.
    
    SPACE COMPLEXITY: O(V + E)
    - Graph storage + Heap.
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
# PATTERN 4: MINIMUM SPANNING TREE (MST - PRIM'S)
# ============================================================================
def min_cost_connect_points(points):
    """
    LeetCode #1584: Min Cost to Connect All Points
    Return minimum cost to make all points connected (Manhattan distance).
    
    APPROACH:
    - Prim's Algorithm (Dense Graph version tailored with Heap).
    - Start from an arbitrary node (0).
    - Maintain PQ of (cost, next_node) representing edges from the "Visited Set" to the "Unvisited Set".
    - Initially: Push (0, 0).
    - Loop until we visit all N nodes:
      1. Pop min cost edge.
      2. If node visited, skip.
      3. Mark node visited, add cost.
      4. Add all edges from this new node to unvisited nodes into PQ.
    
    WHY IT WORKS:
    - Greedily adds the cheapest edge that expands the connected component until all nodes are included.
    
    TIME COMPLEXITY: O(E log V) -> O(N^2 log N)
    - In dense graph, E approx N^2.
    - Note: For very dense graphs, specialized Prim's (without Heap) is O(N^2). This Heap version is suitable generally.
    
    SPACE COMPLEXITY: O(N^2)
    - In worst case heap might store all edges.
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
