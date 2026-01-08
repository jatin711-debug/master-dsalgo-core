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
    
    Approach:
    - Ways(i) = Ways(i-1) + Ways(i-2)
    - Base cases: Ways(1)=1, Ways(2)=2
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
    
    Approach:
    - dp[i] = min coins to make amount 'i'.
    - dp[i] = min(dp[i - coin]) + 1
    - Base: dp[0] = 0, others infinity.
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
    
    Approach:
    - dp[i][j] = LCS of text1[0:i] and text2[0:j]
    - If chars match: dp[i][j] = 1 + dp[i-1][j-1]
    - If no match: max(dp[i-1][j], dp[i][j-1])
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
    
    Approach O(N^2): dp[i] = max(dp[j]) + 1 for j < i and nums[j] < nums[i].
    Approach O(N log N): Patience sorting / BS.
    """
    if not nums: return 0
    # Let's implement O(N^2) for clarity, or O(N logN) for mastery?
    # O(N^2) is "Standard DP". O(N log N) is "Greedy+BS".
    # I'll include O(N^2) as it's the core DP pattern.
    
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
    
    Approach: Expand Around Center.
    - Treat every index (and gap) as a center.
    - Expand while valid palindrome.
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
