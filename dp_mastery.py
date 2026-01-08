# ============================================================================
# MODULE 9: DYNAMIC PROGRAMMING (DP)
# ============================================================================
"""
Optimization. Calculating past results to find the future.

COMMON PATTERNS:
1. 1D DP: dp[i] depends on prior states (Climbing Stairs, Robber).
2. Knapsack: Include/Exclude items (capacity limited).
3. Unbounded Knapsack: Unlimited item usage (Coin Change).
4. LCS: Longest Common Subsequence (String comparison).
5. LIS: Longest Increasing Subsequence.
6. Palindromes: Substring checks expanding from center.

TIME COMPLEXITY: O(N) or O(N*M).
SPACE COMPLEXITY: O(N) or O(N*M) (Can often optimize to O(N) or O(1)).
"""

# ============================================================================
# PATTERN 1: 1D DP (FIBONACCI STYLE)
# ============================================================================
def climb_stairs(n):
    """
    LeetCode #70: Climbing Stairs
    You are climbing a staircase. It takes n steps to reach the top. Each time you can either climb 1 or 2 steps. How many distinct ways can you climb to the top?
    
    APPROACH:
    - Base Logic: To reach step `i`, you must have come from step `i-1` (took 1 step) or step `i-2` (took 2 steps).
    - Recurrence: `ways(i) = ways(i-1) + ways(i-2)`.
    - Loop from 3 to n.
    
    WHY IT WORKS:
    - The problem breaks down into identical subproblems. Since we only need the last two values, we can optimize space.
    
    TIME COMPLEXITY: O(N)
    - Single loop up to N.
    
    SPACE COMPLEXITY: O(1)
    - Only store two previous variables.
    """
    if n <= 2: return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    return prev1

# ============================================================================
# PATTERN 2: UNBOUNDED KNAPSACK (COIN CHANGE)
# ============================================================================
def coin_change(coins, amount):
    """
    LeetCode #322: Coin Change
    Return the fewest number of coins that you need to make up that amount.
    
    APPROACH:
    - `dp[i]` = Minimum coins needed to make amount `i`.
    - Initialize `dp` array with `inf`, set `dp[0] = 0`.
    - Iterate through every amount `a` from 1 to `amount`.
    - For every coin `c`, if `a >= c`, try to update: `dp[a] = min(dp[a], dp[a - c] + 1)`.
    
    WHY IT WORKS:
    - We build the solution from the bottom up. For any amount, the best solution is 1 coin plus the best solution for the remaining amount.
    
    TIME COMPLEXITY: O(Amount * Coins)
    - Nested loop.
    
    SPACE COMPLEXITY: O(Amount)
    - DP array of size Amount + 1.
    """
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    
    for a in range(1, amount + 1):
        for c in coins:
            if a - c >= 0:
                dp[a] = min(dp[a], dp[a - c] + 1)
                
    return dp[amount] if dp[amount] != float('inf') else -1

# ============================================================================
# PATTERN 3: LONGEST COMMON SUBSEQUENCE (2D DP)
# ============================================================================
def longest_common_subsequence(text1, text2):
    """
    LeetCode #1143: Longest Common Subsequence
    Return the length of their longest common subsequence.
    
    APPROACH:
    - 2D Grid `dp[i][j]` representing LCS of `text1[0...i]` and `text2[0...j]`.
    - If characters match (`text1[i] == text2[j]`): `dp[i][j] = 1 + dp[i-1][j-1]` (extend the diagonal match).
    - If mismatch: `dp[i][j] = max(dp[i-1][j], dp[i][j-1])` (carry over the best from either top or left).
    
    WHY IT WORKS:
    - Compares every prefix of text1 against every prefix of text2.
    
    TIME COMPLEXITY: O(M * N)
    - Fill M*N table.
    
    SPACE COMPLEXITY: O(M * N)
    - The DP table. (Can be optimized to O(min(M,N)) using only 2 rows).
    """
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = 1 + dp[i-1][j-1]
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                
    return dp[m][n]

# ============================================================================
# PATTERN 4: LONGEST INCREASING SUBSEQUENCE
# ============================================================================
def length_of_lis(nums):
    """
    LeetCode #300: Longest Increasing Subsequence
    
    APPROACH (Standard DP):
    - `dp[i]` = Length of LIS ending at index `i`.
    - For every `i`, look back at all `j < i`.
    - If `nums[i] > nums[j]`, we can extend the sequence ending at `j`.
    - `dp[i] = max(dp[i], dp[j] + 1)`.
    
    WHY IT WORKS:
    - Guaranteed to find the optimal subsequence ending at each position. The global max of `dp` is the answer.
    
    TIME COMPLEXITY: O(N^2)
    - Nested loops.
    - Note: Can be optimized to O(N log N) using Binary Search (Patience Sorting).
    
    SPACE COMPLEXITY: O(N)
    - DP array.
    """
    if not nums: return 0
    
    dp = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
                
    return max(dp)

# ============================================================================
# PATTERN 5: PALINDROMIC SUBSTRINGS
# ============================================================================
def count_substrings(s):
    """
    LeetCode #647: Palindromic Substrings
    Count how many palindromic substrings exist.
    
    APPROACH:
    - Expand Around Center.
    - There are `2N - 1` centers (N single characters, N-1 spaces between characters).
    - For each center, expand `left` and `right` indices while `s[left] == s[right]`.
    - Increment count for each valid expansion.
    
    WHY IT WORKS:
    - Every palindrome has a center. By checking all centers, we find all palindromes without checking all O(N^2) substrings from scratch.
    
    TIME COMPLEXITY: O(N^2)
    - Expanding takes O(N), we do it for N centers.
    
    SPACE COMPLEXITY: O(1)
    - No extra storage needed.
    """
    count = 0
    for i in range(len(s)):
        # Odd length (center is i)
        l, r = i, i
        while l >= 0 and r < len(s) and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1
            
        # Even length (center is i, i+1)
        l, r = i, i + 1
        while l >= 0 and r < len(s) and s[l] == s[r]:
            count += 1
            l -= 1
            r += 1
    return count

if __name__ == "__main__":
    print("Testing DP:")
    print("-" * 30)
    print("1. Climbing Stairs (5):", climb_stairs(5))
    print("2. Coin Change ([1,2,5], 11):", coin_change([1,2,5], 11))
    print("3. LCS (abcde, ace):", longest_common_subsequence("abcde", "ace"))
    print("4. LIS ([10,9,2,5,3,7,101,18]):", length_of_lis([10,9,2,5,3,7,101,18]))
    print("5. Palindromic Substrings (abc):", count_substrings("abc"))
