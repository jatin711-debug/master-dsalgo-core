# ============================================================================
# MODULE 4: STACKS & QUEUES
# ============================================================================
"""
Ordering data (LIFO vs. FIFO).

COMMON PATTERNS:
1. Valid Parentheses: Matching opening/closing brackets.
2. Monotonic Stack: Next Greater/Smaller Element. (Histogram, Daily Temp).
3. Monotonic Queue: Sliding Window Maximum (Deque).
4. BFS (Queue): Level-order traversal.

TIME COMPLEXITY: O(N) usually.
SPACE COMPLEXITY: O(N).
"""

from collections import deque

# ============================================================================
# PATTERN 1: VALID PARENTHESES (STACK)
# ============================================================================
def isValid(s):
    """
    LeetCode #20: Valid Parentheses
    
    Approach:
    - Push opening brackets to stack.
    - When closing, check if matches top of stack.
    - Stack must be empty at end.
    
    Time: O(N)
    Space: O(N)
    """
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}
    
    for char in s:
        if char in mapping:
            # Closing bracket
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            # Opening bracket
            stack.append(char)
            
    return not stack

# ============================================================================
# PATTERN 2: MONOTONIC STACK (NEXT GREATER ELEMENT)
# ============================================================================
def daily_temperatures(temperatures):
    """
    LeetCode #739: Daily Temperatures
    Find number of days to wait for a warmer temperature.
    
    Approach: Monotonic Decreasing Stack.
    - Store INDICES in stack.
    - If current temp > temp at stack top:
      We found the "next greater" for the stack top. Pop and record distance.
    - Else push current index.
    
    Time: O(N) - each element pushed/popped once
    Space: O(N)
    """
    result = [0] * len(temperatures)
    stack = [] # indices
    
    for i, temp in enumerate(temperatures):
        while stack and temperatures[stack[-1]] < temp:
            prev_index = stack.pop()
            result[prev_index] = i - prev_index
        stack.append(i)
        
    return result

# ============================================================================
# PATTERN 3: MONOTONIC QUEUE (SLIDING WINDOW MAX)
# ============================================================================
def max_sliding_window(nums, k):
    """
    LeetCode #239: Sliding Window Maximum
    
    Approach: Monotonic Decreasing Deque
    - Deque stores INDICES.
    - Maintain deque such that values corresponding to indices are decreasing.
    - Front of deque is always the max for current window.
    - Step 1: Remove indices out of window (i - k).
    - Step 2: Remove indices with values < current num (they can't be max anymore).
    - Step 3: Add current index.
    - Step 4: Add result (from index k-1 onwards).
    
    Time: O(N)
    Space: O(K)
    """
    dq = deque()
    result = []
    
    for i, num in enumerate(nums):
        # 1. Remove out of window indices
        if dq and dq[0] < i - k + 1:
            dq.popleft()
            
        # 2. Maintain Monotonic Decreasing (remove smaller elements from back)
        while dq and nums[dq[-1]] < num:
            dq.pop()
            
        dq.append(i)
        
        # 3. Add to result
        if i >= k - 1:
            result.append(nums[dq[0]])
            
    return result

if __name__ == "__main__":
    print("Testing Stacks & Queues:")
    print("-" * 30)
    print("1. Valid Parentheses ('()[]{}'):", isValid("()[]{}"))
    print("2. Daily Temps ([73,74,75,71,69,72,76,73]):", daily_temperatures([73,74,75,71,69,72,76,73]))
    print("3. Sliding Window Max ([1,3,-1,-3,5,3,6,7], k=3):", max_sliding_window([1,3,-1,-3,5,3,6,7], 3))
