1. **Module 1: Arrays & Strings**  
   The foundation. Most problems here deal with indexing, iteration, and memory layout.  
   **Patterns:**  
   - Two Pointers (Opposite Ends): Iterating from start and end towards the middle. Used for reversing or checking palindromes.  
   - Two Pointers (Same Direction/Fast & Slow): Used to detect cycles or find intersection points.  
   - Sliding Window (Fixed Size): Moving a window of size k across the array.  
   - Sliding Window (Variable Size): Expanding/shrinking a window based on constraints (e.g., "Longest substring without repeating characters").  
   - Prefix Sum: Pre-calculating sums to answer range queries in O(1).  
   - Kadane’s Algorithm: Finding the maximum sum subarray.  
   - Dutch National Flag: Sorting 3 unique elements (0, 1, 2) in-place.  
   - Matrix Traversal: Spiral order, rotating images (matrix manipulation).  
   - Modified Binary Search: Searching in rotated sorted arrays.  
   **How to Spot It:**  
   - "Sorted Array" → Think Two Pointers or Binary Search.  
   - "Subarray" or "Substring" → Think Sliding Window or Prefix Sum.  
   - "Palindrome" → Think Two Pointers.  
   - "In-place" → Think swapping elements or overwriting indices.  

1.5. **Module 1.5: Intervals**  
   Handling overlapping or merging time ranges.  
   **Patterns:**  
   - Merge Intervals: Sorting by start time to merge overlapping intervals.  
   - Non-overlapping Intervals: Sorting by end time (Greedy) to find max compatible set.  
   - Insert Interval: Handling new interval insertion while maintaining order.  
   **How to Spot It:**  
   - "Overlapping times" → Sort by Start Time.  
   - "Minimum intervals to remove" → Sort by End Time (Greedy).  

2. **Module 2: Hashing (Hash Maps & Sets)**  
   The utility knife. Trading space O(N) for time O(1).  
   **Patterns:**  
   - Counting Frequencies: Storing how often an element appears.  
   - Two Sum Variation: Checking if Target - Current_Value exists in the map.  
   - Grouping/Anagrams: Using sorted strings or tuple keys to group similar items.  
   - Subarray Sum Equals K: Using a Hash Map to store Prefix Sum frequencies.  
   - Longest Consecutive Sequence: Using a Set to look up neighbors in O(1).  
   **How to Spot It:**  
   - "Find duplicates" → Hash Set.  
   - "Pairs that sum to..." → Hash Map.  
   - "Frequency of elements" → Hash Map.  
   - "Look up in O(1)" → Hash Map/Set.  

3. **Module 3: Linked Lists**  
   Pointer manipulation. The difficulty is usually handling edge cases (nulls) and reference management.  
   **Patterns:**  
   - Fast & Slow Pointers (Tortoise & Hare): Cycle detection, finding the middle node.  
   - In-place Reversal: Reversing the whole list or a sub-segment (K-group reverse).  
   - Dummy Head: Using a sentinel node to handle edge cases where the head changes (e.g., deleting nodes).  
   - Merge Technique: Merging two sorted lists (similar to Merge Sort merge step).  
   **How to Spot It:**  
   - "Middle of the list" → Fast & Slow Pointers.  
   - "Cycle" → Fast & Slow Pointers.  
   - "Reverse" → Pointer manipulation (Prev, Curr, Next).  
   - "Remove node" → Dummy Head pattern.  

4. **Module 4: Stacks & Queues**  
   Ordering data (LIFO vs. FIFO).  
   **Patterns:**  
   - Valid Parentheses: Matching opening/closing brackets.  
   - Monotonic Stack: Finding the "Next Greater Element" or "Next Smaller Element". (Crucial for Histogram problems).  
   - String Decoding/Calculator: Handling nested operations or undo functionality.  
   - BFS (Breadth-First Search): Using a Queue for level-order traversal (Trees/Graphs).  
   - Monotonic Queue: Finding the max/min in a sliding window (Deque).  
   **How to Spot It:**  
   - "Nested" structures ({{}}) → Stack.  
   - "Next Greater/Smaller" → Monotonic Stack.  
   - "Level by Level" → Queue (BFS).  
   - "Undo/Back button" → Stack.  

5. **Module 5: Recursion & Backtracking**  
   Solving by breaking down, or brute-forcing smartly.  
   **Patterns:**  
   - Subsets / Power Set: Choose vs. Don't Choose logic.  
   - Permutations: Trying every ordering (using a used boolean array).  
   - Combinations: Selecting items with constraints (e.g., Combination Sum).  
   - Grid/Maze Exploration: DFS on a matrix (Flood Fill, Word Search).  
   - N-Queens / Sudoku: Placing items and checking validity, backtracking if invalid.  
   **How to Spot It:**  
   - "Generate all..." → Backtracking.  
   - "Return all valid paths..." → Backtracking.  
   - "Solve a puzzle" → Backtracking.  

6. **Module 6: Trees (Binary Trees & BST)**  
   Hierarchical data. Most solutions are recursive.  
   **Patterns:**  
   - DFS (Pre-order, In-order, Post-order): Visiting nodes in specific orders.  
   - BFS (Level Order Traversal): Using a Queue to process by depth.  
   - Divide and Conquer: Solving for left subtree, right subtree, and combining results (e.g., Height of Tree).  
   - BST Property: Left < Root < Right. Used for validation or searching.  
   - Path Sum: Accumulating values from Root to Leaf.  
   - Serialization: Converting a tree to a string and back.  
   - Lowest Common Ancestor (LCA): Finding the deepest common ancestor.  
   - Tree Construction: Rebuilding unique tree from Preorder + Inorder arrays.  
   **How to Spot It:**  
   - "Hierarchical" or "Organization" → Tree.  
   - "Depth" → DFS.  
   - "Level by Level" → BFS.  
   - "Sorted Tree" → Binary Search Tree (BST).  

7. **Module 7: Heaps (Priority Queue)**  
   Ordering based on priority (Min or Max).  
   **Patterns:**  
   - Top 'K' Elements: Keep a heap of size K.  
   - Merge 'K' Sorted Lists: Pushing the head of each list into a Min-Heap.  
   - Two Heaps: Maintaining a Median in a stream (one Min-Heap, one Max-Heap).  
   - Scheduling Tasks: Greedily picking the task with the highest priority/frequency.  
   **How to Spot It:**  
   - "Find the Kth largest/smallest" → Heap.  
   - "Top K items" → Heap.  
   - "Median of data stream" → Two Heaps.  

8. **Module 8: Graphs**  
   Nodes and Edges. Relationships.  
   **Patterns:**  
   - BFS (Shortest Path): Finding the shortest path in an unweighted graph.  
   - DFS (Connectivity): Detecting cycles, checking if path exists, counting islands.  
   - Union-Find (Disjoint Set): Grouping components, detecting cycles in undirected graphs.  
   - Topological Sort (Kahn’s Algo / DFS): Scheduling dependencies (e.g., Course Schedule).  
   - Dijkstra’s Algorithm: Shortest path in a weighted graph (No negative weights).  
   - Minimum Spanning Tree (MST): Kruskal’s or Prim’s algo for connecting graph with min cost.  
   **How to Spot It:**  
   - "Network", "Connections", "Islands" → Graph DFS/BFS/Union-Find.  
   - "Shortest Path" → BFS (Unweighted) or Dijkstra (Weighted).  
   - "Prerequisites" or "Dependencies" → Topological Sort.  
   - "Cycle detection" → DFS (Directed) or Union-Find (Undirected).  

9. **Module 9: Dynamic Programming (DP)**  
   Optimization. Calculating past results to find the future.  
   **Patterns:**  
   - 1D DP (Fibonacci Style): dp[i] depends on dp[i-1] and dp[i-2] (Climbing Stairs).  
   - Knapsack (0/1): Include or Exclude item to maximize value with capacity constraints.  
   - Unbounded Knapsack: Use items unlimited times (Coin Change).  
   - Longest Common Subsequence (LCS): Comparing two strings (2D DP).  
   - Longest Increasing Subsequence (LIS): Finding sorted subsequence (O(N^2) or O(NlogN)).  
   - Palindromic DP: Expanding from center or checking substrings [i...j].  
   **How to Spot It:**  
   - "Maximize/Minimize value" → DP.  
   - "Count number of ways" → DP.  
   - "Can we form..." → DP.  
   - Problem has overlapping subproblems.  

10. **Module 10: Advanced/Niche**  
    Tries (Prefix Tree): Autocomplete, spell checker, prefix matching.  
    Bit Manipulation: XOR (finding unique number), Bit Masking (representing sets as integers).  
    Binary Search on Answer: When the answer is within a range (e.g., 1 to 1000) and you can check validity in O(N). "Minimize the Max" problems.
