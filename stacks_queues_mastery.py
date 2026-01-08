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
    Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.
    
    Example:
        Input: s = "()[]{}"
        Output: True
    
    APPROACH:
    - Use a Stack to keep track of opening brackets.
    - When we encounter a closing bracket:
      - Check if stack is empty (Invalid).
      - Check if the top of stack matches the current closing bracket.
      - If match, pop from stack. If not match, return False.
    - At the end, if stack is empty, return True.

    WHY IT WORKS:
    - Stack (LIFO) perfectly models nested structures. The most recently opened bracket must be the first one closed.

    TIME COMPLEXITY: O(N)
    - Traverse string once. Push/Pop are O(1).

    SPACE COMPLEXITY: O(N)
    - Worst case: "(((((" stores all chars in stack.
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
    Given an array of temperatures, return an array answer such that answer[i] is the number of days you have to wait after the ith day to get a warmer temperature.
    
    Example:
        Input: [73,74,75,71,69,72,76,73]
        Output: [1, 1, 4, 2, 1, 1, 0, 0]
    
    APPROACH:
    - Monotonic Decreasing Stack.
    - We want to find the *next* element that is larger.
    - We store **indices** in the stack.
    - Invariant: Elements in stack (referenced by index) are always in decreasing order.
    - Logic:
      - When we see a new temperature `T[i]`, we compare it with `T[stack.top()]`.
      - While `T[i] > T[stack.top()]`: We found a warmer day for the index at stack top!
        - Pop index, calculate distance `i - top_index`, store in result.
      - Push `i` onto stack.

    WHY IT WORKS:
    - By waiting to resolve indices until we find a larger value, we efficiently solve the "Next Greater Element" problem.
    - Each element is pushed once and popped once.

    TIME COMPLEXITY: O(N)
    - Even though there is a while loop, each element is pushed and popped at most once.

    SPACE COMPLEXITY: O(N)
    - Stack size.
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
    You are given an array of integers nums, there is a sliding window of size k which is moving from the very left of the array to the very right. Return the max sliding window.
    
    Example:
        Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
        Output: [3,3,5,5,6,7]
    
    APPROACH:
    - Use a Deque (Double-ended queue) to store indices.
    - We want the Deque to store indices of potential maximums.
    - Invariant: The values corresponding to indices in Deque are Strictly Decreasing. `nums[dq[0]]` is always the MAX.
    - Steps for each element `nums[i]`:
      1. Remove indices from front that are out of the new window (`i - k`).
      2. Maintain Monotonicity: Remove indices from back if `nums[back] < nums[i]` (because `nums[i]` is newer and larger, so `nums[back]` is useless).
      3. Add `i` to back.
      4. If `i >= k-1`, add `nums[dq[0]]` to result.

    WHY IT WORKS:
    - `dq[0]` always holds the index of the largest element in current window.
    - We remove smaller elements because they can never be the maximum if a larger, newer element exists.

    TIME COMPLEXITY: O(N)
    - Each element added and removed from deque at most once.

    SPACE COMPLEXITY: O(K)
    - Deque stores at most K elements (indices).
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
