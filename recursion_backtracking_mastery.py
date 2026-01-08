# ============================================================================
# MODULE 5: RECURSION & BACKTRACKING
# ============================================================================
"""
Solving by breaking down (Recursion) or brute-forcing smartly (Backtracking).

COMMON PATTERNS:
1. Subsets/Combinations: Choose vs Don't Choose (Decision Tree).
2. Permutations: Ordering (Used array/swapping).
3. Grid DFS: Flood Fill, Word Search (Matrix exploration).
4. Pruning: Stopping early if constraints violated (Sudoku, N-Queens).

TIME COMPLEXITY: Exponential (O(2^N), O(N!)).
SPACE COMPLEXITY: O(N) recursion stack.
"""

# ============================================================================
# PATTERN 1: SUBSETS (POWER SET)
# ============================================================================
def subsets(nums):
    """
    LeetCode #78: Subsets
    Given an integer array nums of unique elements, return all possible subsets (the power set).
    
    Example:
        Input: nums = [1,2,3]
        Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
    
    APPROACH:
    - Backtracking / Decision Tree Strategy.
    - For each element, we have two choices:
      1. Include it in the current subset.
      2. Exclude it from the current subset.
    - Recurse until we process all elements.

    WHY IT WORKS:
    - This creates a binary decision tree of height N.
    - The leaves of this tree represent all 2^N possible combinations.
    
    TIME COMPLEXITY: O(N * 2^N)
    - 2^N possible subsets.
    - Copying the subset list takes O(N).

    SPACE COMPLEXITY: O(N)
    - Recursion depth is N.
    """
    result = []
    
    def backtrack(start_index, current_path):
        # Add copy of current path
        result.append(current_path[:])
        
        for i in range(start_index, len(nums)):
            # Include nums[i]
            current_path.append(nums[i])
            # Move to next
            backtrack(i + 1, current_path)
            # Backtrack (remove)
            current_path.pop()
            
    backtrack(0, [])
    return result

# ============================================================================
# PATTERN 2: PERMUTATIONS
# ============================================================================
def permute(nums):
    """
    LeetCode #46: Permutations
    Given an array nums of distinct integers, return all the possible permutations.
    
    Example:
        Input: [1,2,3]
        Output: 6 permutations
    
    APPROACH:
    - Trying every ordering.
    - Loop through available numbers. Check if number is used.
    - If not used, add to path and mark as used.
    - Recurse.
    - After recursion returns, backtrack: remove from path, mark as unused.

    WHY IT WORKS:
    - Generates the factorial tree (N options, then N-1, etc.).
    - Ensures every unique ordering is explored exactly once.

    TIME COMPLEXITY: O(N * N!)
    - N! permutations.
    - Operations per node can take O(N).

    SPACE COMPLEXITY: O(N)
    - Recursion depth N. Used set O(N).
    """
    result = []
    
    def backtrack(current_path, used_set):
        if len(current_path) == len(nums):
            result.append(current_path[:])
            return

        for num in nums:
            if num not in used_set:
                used_set.add(num)
                current_path.append(num)
                
                backtrack(current_path, used_set)
                
                current_path.pop()
                used_set.remove(num)
                
    backtrack([], set())
    return result

# ============================================================================
# PATTERN 3: COMBINATION SUM
# ============================================================================
def combination_sum(candidates, target):
    """
    LeetCode #39: Combination Sum
    Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations where the chosen numbers sum to target. (Can reuse elements).
    
    Example:
        Input: [2,3,6,7], target=7
        Output: [[2,2,3],[7]]
    
    APPROACH:
    - Decision: Include candidates[i] OR don't include candidates[i].
    - Since we can reuse, if we choose to include, we recurse with same index `i`.
    - If we choose not to include, we move to `i+1`.
    - Base Case: remaining == 0 (Add to result), remaining < 0 (Stop).

    WHY IT WORKS:
    - DFS exploration of the valid sum space.
    - Passing index `i` avoids duplicates (we never look back at elements before `i`).

    TIME COMPLEXITY: O(N^(T/M)) roughly
    - N = number of candidates. T = target. M = min candidate value.
    - Exponential in worst case.

    SPACE COMPLEXITY: O(T/M)
    - Recursion depth.
    """
    result = []
    
    def backtrack(remaining, start_index, path):
        if remaining == 0:
            result.append(path[:])
            return
        if remaining < 0:
            return
            
        for i in range(start_index, len(candidates)):
            path.append(candidates[i])
            # Pass 'i' because we can reuse the same element
            backtrack(remaining - candidates[i], i, path)
            path.pop()
            
    backtrack(target, 0, [])
    return result

# ============================================================================
# PATTERN 4: WORD SEARCH (GRID DFS)
# ============================================================================
def exist(board, word):
    """
    LeetCode #79: Word Search
    Given an m x n grid of characters board and a string word, return true if word exists in the grid.
    
    APPROACH:
    - Loop through every cell. If cell matches word[0], start DFS.
    - DFS:
      - Boundaries check? Match char check? If fail, return False.
      - If index == len(word), We found it! Returns True.
      - Mark current cell as visited (e.g. use a special char '#').
      - Recurse in 4 directions.
      - Backtrack: Restore cell to original char.

    WHY IT WORKS:
    - Systematically explores all paths from a potential start point.
    - Backtracking ensures we can reuse the cell for other paths if the current one fails.

    TIME COMPLEXITY: O(Rows * Cols * 4^L)
    - Loop R*C times.
    - DFS depth L (word length). Branching factor 3 (effectively, since we don't go back). 4^L worst case.

    SPACE COMPLEXITY: O(L)
    - Recursion stack depth is length of word.
    """
    if not board: return False
    rows, cols = len(board), len(board[0])
    
    def dfs(r, c, index):
        if index == len(word):
            return True
        
        if (r < 0 or r >= rows or c < 0 or c >= cols or 
            board[r][c] != word[index]):
            return False
            
        temp = board[r][c]
        board[r][c] = '#' # Mark visited
        
        found = (dfs(r+1, c, index+1) or 
                 dfs(r-1, c, index+1) or 
                 dfs(r, c+1, index+1) or 
                 dfs(r, c-1, index+1))
                 
        board[r][c] = temp # Backtrack
        return found
        
    for r in range(rows):
        for c in range(cols):
            if board[r][c] == word[0] and dfs(r, c, 0):
                return True
    return False

if __name__ == "__main__":
    print("Testing Recursion/Backtracking:")
    print("-" * 30)
    print("1. Subsets ([1,2,3]):", subsets([1,2,3]))
    print("2. Permutations ([1,2,3]):", permute([1,2,3]))
    print("3. Combination Sum ([2,3,6,7], 7):", combination_sum([2,3,6,7], 7))
    board = [['A','B','C','E'],['S','F','C','S'],['A','D','E','E']]
    print("4. Word Search (ABCCED):", exist(board, "ABCCED"))
