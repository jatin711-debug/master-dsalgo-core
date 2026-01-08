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
    
    Approach: Cascading or Backtracking.
    Backtracking: For each element, include it or don't.
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
    
    Approach:
    - Try every number at current position.
    - Keep track of used numbers (or swap in-place).
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
    
    Approach:
    - Same number can be used unlimited times -> pass 'i' not 'i+1'.
    - Base case: sum == target (add), sum > target (return).
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
    
    Approach:
    - DFS from every cell matching first char.
    - Mark visited (temporarily with '#').
    - Check 4 directions.
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
