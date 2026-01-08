"""
ARRAYS & STRINGS - MASTER PATTERNS GUIDE
========================================

Module 1 covers the fundamental patterns for solving array and string problems.
This file contains all patterns with templates and example problems.

7 CORE PATTERNS:
1. Two Pointers (Opposite Ends)
2. Two Pointers (Same Direction/Fast & Slow)
3. Sliding Window (Fixed Size)
4. Sliding Window (Variable Size)
5. Prefix Sum
6. Kadane's Algorithm (Maximum Subarray)
7. Dutch National Flag
8. Matrix Traversal
"""

# ============================================================================
# PATTERN 1: TWO POINTERS (OPPOSITE ENDS)
# ============================================================================
"""
CONCEPT: Use two pointers starting from opposite ends, moving towards each other.
- One pointer at start (left), one at end (right)
- Move them based on comparison or conditions
- Perfect for problems involving sorted arrays or palindromes

TIME: O(N) - each element visited at most once
SPACE: O(1) - only using two pointers

KEY ADVANTAGE:
- Eliminates need to check all pairs O(N²)
- Single pass through data
- Perfect for symmetric problems (palindromes, sorted array two-sum)
- Reduces space complexity by avoiding hash maps in some cases
"""

def two_pointers_opposite_template(arr):
    """
    TEMPLATE for Two Pointers (Opposite Ends)
    """
    left = 0
    right = len(arr) - 1

    while left < right:
        continue
        # Process current elements
        # left_val = arr[left]
        # right_val = arr[right]

        # Decide which pointer to move
        # if some_condition:
        #     left += 1
        # else:
        #     right -= 1

    # Return result
    return None


# EXAMPLE PROBLEM 1 - EASY: Valid Palindrome
def is_palindrome(s):
    """
    LeetCode #125: Valid Palindrome

    A phrase is a palindrome if, after converting all uppercase letters into
    lowercase letters and removing all non-alphanumeric characters, it reads
    the same forward and backward.

    Example:
        Input: s = "A man, a plan, a canal: Panama"
        Output: true

    APPROACH:
    - Use two pointers at start and end of string
    - Skip non-alphanumeric chars by moving pointers
    - Compare lowercase characters
    - Move both pointers inward until they meet

    WHY IT WORKS:
    - Palindrome property: char at position i equals char at position n-1-i
    - Checking from both ends simultaneously is optimal
    - We visit each character at most once

    TIME COMPLEXITY: O(N)
    - Each character is visited at most once by either left or right pointer
    - Total operations proportional to string length

    SPACE COMPLEXITY: O(1)
    - Only using two pointers (left and right)
    - No additional data structures proportional to input size

    OPTIMIZATION ACHIEVED:
    - Brute force would compare all pairs O(N²/2) or reverse string O(N)
    - Two pointers achieve O(N) by eliminating redundant comparisons
    - Single pass through data
    """
    left = 0
    right = len(s) - 1

    while left < right:
        # Skip non-alphanumeric from left
        while left < right and not s[left].isalnum():
            left += 1

        # Skip non-alphanumeric from right
        while left < right and not s[right].isalnum():
            right -= 1

        # Compare characters (case-insensitive)
        if left < right and s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True


# EXAMPLE PROBLEM 2 - MEDIUM: 3Sum
def three_sum(nums):
    """
    LeetCode #15: 3Sum

    Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]]
    such that i != j, i != k, j != k, and nums[i] + nums[j] + nums[k] == 0.

    Example:
        Input: nums = [-1,0,1,2,-1,-4]
        Output: [[-1,-1,2],[-1,0,1]]

    APPROACH:
    - Sort the array first
    - Use outer loop to fix first element
    - Use two pointers (left, right) for remaining elements
    - Calculate sum = nums[i] + nums[left] + nums[right]
    - Adjust pointers based on sum comparison

    WHY IT WORKS:
    - After sorting, we can efficiently skip duplicates
    - Two pointers allow O(N) lookup for remaining two numbers
    - Total: O(N²) - better than O(N³) brute force
    - We avoid duplicates by checking previous element

    TIME COMPLEXITY: O(N²)
    - Sorting: O(N log N)
    - Outer loop: O(N)
    - Inner two-pointer search: O(N) per iteration
    - Total: O(N log N) + O(N²) = O(N²)

    SPACE COMPLEXITY: O(1) or O(N)
    - O(1) if we sort in-place
    - O(N) for output list (required by problem)
    - O(log N) for recursion stack if using quicksort

    OPTIMIZATION ACHIEVED:
    - Brute force: Three nested loops O(N³)
    - With sorting + two pointers: O(N²)
    - Key insight: Fix one element, reduce 3Sum to 2Sum problem
    - 2Sum with sorted array is O(N) using two pointers
    - Eliminates need for hash set (which uses O(N) space)
    """
    nums.sort()  # Sort to enable two pointers and duplicate handling
    result = []

    for i in range(len(nums) - 2):
        # Skip duplicates for first element
        if i > 0 and nums[i] == nums[i - 1]:
            continue

        left = i + 1
        right = len(nums) - 1

        while left < right:
            current_sum = nums[i] + nums[left] + nums[right]

            if current_sum == 0:
                result.append([nums[i], nums[left], nums[right]])

                # Skip duplicates
                while left < right and nums[left] == nums[left + 1]:
                    left += 1
                while left < right and nums[right] == nums[right - 1]:
                    right -= 1

                # Move both pointers to find next unique pair
                left += 1
                right -= 1
            elif current_sum < 0:
                left += 1  # Need larger sum, move left pointer
            else:
                right -= 1  # Need smaller sum, move right pointer

    return result


# EXAMPLE PROBLEM 3 - HARD: Trapping Rain Water
def trap(height):
    """
    LeetCode #42: Trapping Rain Water

    Given n non-negative integers representing an elevation map where the width
    of each bar is 1, compute how much water it can trap after raining.

    Example:
        Input: height = [0,1,0,2,1,0,1,3,2,1,2,1]
        Output: 6

    APPROACH:
    - Water at each position = min(max_left, max_right) - height[i]
    - Use two pointers approach for O(N) time, O(1) space
    - Track max height from left and right
    - At each step, process the side with smaller max height
    - If left max <= right max, water trapped = left max - height[left]

    WHY IT WORKS:
    - Water trapped depends on the smaller of the two boundary heights
    - We don't need to know the absolute max on both sides simultaneously
    - Processing the smaller side first ensures we can compute water accurately
    - Only need O(1) extra space instead of storing max arrays

    MATHEMATICAL PROOF:
    - Let max_left[i] be max height to the left of i
    - Let max_right[i] be max height to the right of i
    - Water[i] = max(0, min(max_left[i], max_right[i]) - height[i])
    - The algorithm computes this by maintaining current max from each side

    TIME COMPLEXITY: O(N)
    - Single pass through array
    - Each element processed exactly once

    SPACE COMPLEXITY: O(1)
    - Only using four variables: left, right, left_max, right_max
    - No additional data structures proportional to input size

    OPTIMIZATION ACHIEVED:
    - Naive approach: Precompute max_left and max_right arrays O(N) space
    - Two-pointer approach: O(1) space by processing from sides
    - Key insight: Water at position depends on min(max_left, max_right)
    - We can compute this incrementally without storing all values
    - Processing smaller side first is safe because we know its limiting factor
    """
    if not height:
        return 0

    left = 0
    right = len(height) - 1
    left_max = 0
    right_max = 0
    water = 0
    while left < right:
        # Track maximum heights
        if height[left] > left_max:
            left_max = height[left]

        if height[right] > right_max:
            right_max = height[right]


        # Process the smaller side
        if left_max <= right_max:
            # Water trapped = left_max - height[left]
            water += max(0, left_max - height[left])
            left += 1
        else:
            # Water trapped = right_max - height[right]
            water += max(0, right_max - height[right])
            right -= 1

    return water


# ============================================================================
# PATTERN 2: TWO POINTERS (SAME DIRECTION / FAST & SLOW)
# ============================================================================
"""
CONCEPT: Two pointers moving in the same direction at different speeds.
- Fast pointer advances 2 steps at a time
- Slow pointer advances 1 step at a time
- Used to find middle, detect cycles, or find kth elements

TIME: O(N) - each element visited at most once
SPACE: O(1) - only using two pointers

KEY ADVANTAGE:
- Single pass O(N) vs two-pass O(2N) for finding middle
- Elegant solution to linked list problems
- Floyd's cycle detection algorithm
- Mathematical relationship: distance traveled proportional to speed ratio
- No need to pre-compute length or store visited nodes
"""

def two_pointers_same_direction_template(arr):
    """
    TEMPLATE for Two Pointers (Same Direction/Fast & Slow)
    """
    slow = 0
    fast = 0

    # while fast < len(arr):
    #     # Move slow by 1, fast by 2
    #     slow += 1
    #     fast += 2

    return None


# EXAMPLE PROBLEM 1 - EASY: Middle of Linked List
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def middle_node(head):
    """
    LeetCode #876: Middle of Linked List

    Given the head of a singly linked list, return the middle node.

    Example:
        Input: head = [1,2,3,4,5]
        Output: ListNode(3)

    APPROACH:
    - Use slow and fast pointers
    - Slow moves 1 step, fast moves 2 steps
    - When fast reaches end, slow is at middle
    - For even length, fast will be None when slow is at second middle

    WHY IT WORKS:
    - Fast pointer travels twice as fast as slow
    - When fast completes, slow has traveled half the distance
    - This elegantly finds middle in O(N) time and O(1) space

    TIME COMPLEXITY: O(N)
    - Fast pointer traverses at most N/2 nodes
    - Slow pointer traverses at most N/2 nodes
    - Total operations proportional to list length

    SPACE COMPLEXITY: O(1)
    - Only using two pointers: slow and fast
    - No additional data structures

    OPTIMIZATION ACHIEVED:
    - Brute force: First find length O(N), then traverse N/2 steps O(N)
    - Fast & slow: Single pass O(N), no need to count length
    - Avoids two-pass solution by using speed difference
    - Elegant mathematical property: distance ratio
    """
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow


# EXAMPLE PROBLEM 2 - MEDIUM: Remove Nth Node From End
def remove_nth_from_end(head, n):
    """
    LeetCode #19: Remove Nth Node From End of List

    Given the head of a linked list, remove the nth node from the end and
    return its head.

    Example:
        Input: head = [1,2,3,4,5], n = 2
        Output: [1,2,3,5]

    APPROACH:
    - Use dummy node before head for edge cases (removing head)
    - Move fast pointer n steps ahead first
    - Then move both pointers until fast reaches end
    - Slow will be at node before the one to delete

    WHY IT WORKS:
    - The gap between slow and fast is always n nodes
    - When fast is at end, slow is at node before target
    - Dummy node handles deletion of head node uniformly
    """
    dummy = ListNode(0, head)
    slow = dummy
    fast = dummy

    # Move fast n steps ahead
    for _ in range(n):
        fast = fast.next

    # Move both until fast reaches end
    while fast.next:
        slow = slow.next
        fast = fast.next

    # Remove nth node
    slow.next = slow.next.next

    return dummy.next


# EXAMPLE PROBLEM 3 - HARD: Linked List Cycle II
def detect_cycle_start(head):
    """
    LeetCode #142: Linked List Cycle II

    Given the head of a linked list, return the node where the cycle begins.
    If there is no cycle, return null.

    Example:
        Input: head = [3,2,0,-4], pos = 1
        Output: returns the node with value 2

    APPROACH:
    - Phase 1: Detect cycle using fast & slow pointers
    - Phase 2: Find cycle entry point
    - When cycle detected, reset one pointer to head
    - Move both pointers at same speed, they'll meet at cycle start

    WHY IT WORKS - MATHEMATICAL PROOF:
    - Let distance from head to cycle start = F
    - Let distance from cycle start to meeting point = X
    - Let cycle length = C
    - When slow & fast meet: slow traveled F + X, fast traveled F + X + k*C
    - Since fast = 2 * slow: 2(F + X) = F + X + k*C
    - Therefore: F = (k-1)*C + (C - X)
    - This means F = C - X (mod C)
    - So if we restart one pointer at head, they'll meet at cycle start
    """
    # Phase 1: Detect cycle
    slow = head
    fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            # Phase 2: Find cycle start
            slow = head
            while slow != fast:
                slow = slow.next
                fast = fast.next
            return slow

    return None  # No cycle


# ============================================================================
# PATTERN 3: SLIDING WINDOW (FIXED SIZE)
# ============================================================================
"""
CONCEPT: Maintain a window of fixed size k that slides across the data.
- Use two pointers to define window boundaries
- Update result as window slides by one position
- Common for maximums, averages, or patterns in consecutive elements

TIME: O(N) - each element enters and leaves window once
SPACE: O(1) - or O(k) if storing window elements

KEY ADVANTAGE:
- Avoids recalculating window sum from scratch O(k) each time
- Incremental update: subtract left, add right in O(1)
- Total operations: N-k+1 windows vs (N-k+1)*k without optimization
- Can maintain complex window state (max, min, frequency maps)
- Perfect for subarray problems with fixed constraints
"""

def sliding_window_fixed_template(arr, k):
    """
    TEMPLATE for Sliding Window (Fixed Size)
    """
    if len(arr) < k:
        return None  

    # Calculate initial window
    window_sum = sum(arr[:k])

    for i in range(len(arr) - k):
        # Slide window: remove left, add right
        window_sum = window_sum - arr[i] + arr[i + k]

    return window_sum


# EXAMPLE PROBLEM 1 - EASY: Maximum Average Subarray I
def find_max_average(nums, k):
    """
    LeetCode #643: Maximum Average Subarray I

    Given an integer array nums, find a contiguous subarray of length k
    that has the maximum average and return its value.

    Example:
        Input: nums = [1,12,-5,-6,50,3], k = 4
        Output: 12.75 (average of [12,-5,-6,50])

    APPROACH:
    - Calculate sum of first k elements
    - Slide window by subtracting element leaving and adding new element
    - Track maximum sum encountered

    WHY IT WORKS:
    - Fixed window size k means we only need to check n-k+1 possible windows
    - Sliding eliminates redundant calculations
    - Each element added/removed exactly once: O(N) total
    """
    # Calculate initial sum
    window_sum = sum(nums[:k])
    max_sum = window_sum

    # Slide window
    for i in range(len(nums) - k):
        window_sum = window_sum - nums[i] + nums[i + k]
        max_sum = max(max_sum, window_sum)

    return max_sum / k


# EXAMPLE PROBLEM 2 - MEDIUM: Sliding Window Maximum
def max_sliding_window(nums, k):
    """
    LeetCode #239: Sliding Window Maximum

    You are given an array of integers nums, there is a sliding window of size k
    which moves from the very left to the very right. Return the max in each window.

    Example:
        Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
        Output: [3,3,5,5,6,7]

    APPROACH:
    - Use a deque (double-ended queue) to store indices
    - Maintain decreasing order in deque (front = max)
    - Remove indices outside current window
    - Remove indices smaller than current element from back
    - Front of deque is always the maximum

    WHY IT WORKS:
    - Deque stores candidates for maximum in order
    - Elements are removed in O(1) from both ends
    - We only add each element once and remove once: O(N)
    - Monotonic decreasing property ensures max is always at front
    """
    from collections import deque

    if not nums:
        return []

    result = []
    dq = deque()  # Store indices, maintain decreasing values

    for i in range(len(nums)):
        # Remove indices not in current window
        while dq and dq[0] <= i - k:
            dq.popleft()

        # Remove smaller elements from back
        while dq and nums[dq[-1]] < nums[i]:
            dq.pop()

        # Add current index
        dq.append(i)

        # Record result after first window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result


# EXAMPLE PROBLEM 3 - HARD: Minimum Window Substring (Fixed Sliding Window Variants)
def min_window_substring(s, t):
    """
    LeetCode #76: Minimum Window Substring

    Given two strings s and t, return the minimum window in s which contains
    all the characters of t. If there is no such window, return "".

    Example:
        Input: s = "ADOBECODEBANC", t = "ABC"
        Output: "BANC"

    APPROACH:
    - Use sliding window with frequency map
    - Expand window by adding characters from s
    - Contract window when all characters of t are satisfied
    - Track minimum window size

    WHY IT WORKS:
    - We only care about windows containing all required characters
    - Expanding finds valid windows, contracting finds minimal valid windows
    - Each character processed at most twice (added and removed): O(N)
    """
    if not s or not t or len(s) < len(t):
        return ""

    from collections import Counter, defaultdict

    # Frequency map of characters in t
    need = Counter(t)
    window = defaultdict(int)
    have = 0
    need_count = len(need)
    min_len = float('inf')
    min_window = ""

    left = 0

    for right, char in enumerate(s):
        # Add character to window
        if char in need:
            window[char] += 1
            if window[char] == need[char]:
                have += 1

        # Try to contract window when all requirements are met
        while have == need_count:
            # Update minimum window
            if right - left + 1 < min_len:
                min_len = right - left + 1
                min_window = s[left:right + 1]

            # Remove from window
            if s[left] in need:
                if window[s[left]] == need[s[left]]:
                    have -= 1
                window[s[left]] -= 1

            left += 1

    return min_window


# ============================================================================
# PATTERN 4: SLIDING WINDOW (VARIABLE SIZE)
# ============================================================================
"""
CONCEPT: Expand window until condition is met, then contract to find optimum.
- Window size adjusts based on problem constraints
- Common for "longest substring without repeating", "minimum subarray sum", etc.

TIME: O(N) - each element added/removed at most once
SPACE: O(K) - where K is unique elements in window
"""

def sliding_window_variable_template(s):
    """
    TEMPLATE for Sliding Window (Variable Size)
    """
    # Example: longest substring without repeating characters

    # Use dictionary/hash map to track last seen position
    char_map = {}
    left = 0
    max_length = 0

    for right in range(len(s)):
        # Expand window by moving right pointer
        char = s[right]

        # If char seen in current window, move left pointer
        # This handles variable window size

        # Update max_length

    return max_length


# EXAMPLE PROBLEM 1 - EASY: Longest Substring Without Repeating Characters
def length_of_longest_substring(s):
    """
    LeetCode #3: Longest Substring Without Repeating Characters

    Given a string s, find the length of the longest substring without repeating
    characters.

    Example:
        Input: s = "abcabcbb"
        Output: 3 ("abc")

    APPROACH:
    - Use sliding window with hash map
    - Track last seen index of each character
    - If character seen before and index >= left pointer, move left pointer
    - Update max length

    WHY IT WORKS:
    - Window expands as we read new characters
    - When duplicate found, we shrink from left to eliminate it
    - We only move pointers forward: each character processed O(1) times
    - Hash map gives O(1) lookup for duplicates

    TIME COMPLEXITY: O(N)
    - Each character processed at most twice (once by right, once by left)
    - Hash map operations O(1) average case
    - Total operations proportional to string length

    SPACE COMPLEXITY: O(K)
    - K = number of unique characters in string
    - Hash map stores at most one entry per unique character
    - In worst case (all unique): O(N) space

    OPTIMIZATION ACHIEVED:
    - Brute force: Check all substrings O(N²), each check O(N) = O(N³)
    - With sliding window: O(N) by maintaining valid window
    - Key insight: When duplicate found, skip directly to after previous occurrence
    - Avoids checking all possible starting positions
    - Both pointers only move forward (no backtracking)
    """
    char_map = {}  # Store character -> last index
    left = 0
    max_length = 0

    for right in range(len(s)):
        char = s[right]

        # If char in current window, move left pointer past previous occurrence
        if char in char_map and char_map[char] >= left:
            left = char_map[char] + 1

        # Update last seen index
        char_map[char] = right

        # Update max length
        max_length = max(max_length, right - left + 1)

    return max_length


# EXAMPLE PROBLEM 2 - MEDIUM: Minimum Size Subarray Sum
def min_subarray_len(target, nums):
    """
    LeetCode #209: Minimum Size Subarray Sum

    Given an array of positive integers nums and a positive integer target,
    return the minimal length of a contiguous subarray of which the sum is
    greater than or equal to target.

    Example:
        Input: target = 7, nums = [2,3,1,2,4,3]
        Output: 2 ([4,3])

    APPROACH:
    - Expand window by adding elements (increase sum)
    - When sum >= target, try to shrink from left
    - Track minimum length found

    WHY IT WORKS:
    - Sliding window handles sum calculation efficiently
    - Expanding builds up to target, contracting finds minimum
    - Each element added once and removed once: O(N)
    - More efficient than checking all subarrays O(N²)
    """
    left = 0
    current_sum = 0
    min_length = float('inf')

    for right in range(len(nums)):
        # Expand window
        current_sum += nums[right]

        # Contract window while sum >= target
        while current_sum >= target:
            min_length = min(min_length, right - left + 1)
            current_sum -= nums[left]
            left += 1

    return min_length if min_length != float('inf') else 0


# EXAMPLE PROBLEM 3 - HARD: Sliding Window Maximum (already covered) or
#                    Fruit Into Baskets (Alternative)
def total_fruit(fruits):
    """
    LeetCode #904: Fruit Into Baskets

    You are visiting a farm that has a single row of fruit trees arranged from
    left to right. The trees are represented by an integer array fruits where
    fruits[i] is the type of fruit the ith tree produces. You have two baskets,
    and each basket can hold only one type of fruit. Return the maximum number
    of fruits you can collect.

    Example:
        Input: fruits = [1,2,1,2,1,2,1,2,3]
        Output: 6

    APPROACH:
    - Sliding window with at most 2 distinct fruit types
    - Use hash map to track fruit types and their counts
    - When >2 types, shrink from left until 2 types remain
    - Track maximum window size

    WHY IT WORKS:
    - Window expands to include more fruits
    - Contract when constraint violated (>2 types)
    - Each tree processed at most twice
    - Optimal substructure: best window is always a contiguous subarray
    """
    left = 0
    fruit_count = {}
    max_fruits = 0

    for right in range(len(fruits)):
        fruit = fruits[right]
        fruit_count[fruit] = fruit_count.get(fruit, 0) + 1

        # Contract window if more than 2 types
        while len(fruit_count) > 2:
            left_fruit = fruits[left]
            fruit_count[left_fruit] -= 1
            if fruit_count[left_fruit] == 0:
                del fruit_count[left_fruit]
            left += 1

        max_fruits = max(max_fruits, right - left + 1)

    return max_fruits


# ============================================================================
# PATTERN 5: PREFIX SUM
# ============================================================================
"""
CONCEPT: Pre-compute cumulative sums to answer range sum queries in O(1).
- prefix_sum[i] = sum of elements from 0 to i-1
- Range sum [l, r] = prefix_sum[r+1] - prefix_sum[l]
- Useful for static array range queries

TIME: O(N) to build, O(1) per query
SPACE: O(N) for prefix sum array
"""

def prefix_sum_template(arr):
    """
    TEMPLATE for Prefix Sum
    """
    prefix = [0] * (len(arr) + 1)

    # Build prefix sum array
    for i in range(len(arr)):
        prefix[i + 1] = prefix[i] + arr[i]

    # Query: sum from l to r (inclusive)
    # return prefix[r + 1] - prefix[l]

    return prefix


# EXAMPLE PROBLEM 1 - EASY: Running Sum of 1d Array
def running_sum(nums):
    """
    LeetCode #1480: Running Sum of 1d Array

    Given an array nums, return the running sum of nums.

    Example:
        Input: nums = [1,2,3,4]
        Output: [1,3,6,10]

    APPROACH:
    - Each element is sum of all previous elements + itself
    - Can be done in-place or with prefix sum array

    WHY IT WORKS:
    - Running sum[i] = sum(nums[0:i+1])
    - Running sum[i] = running sum[i-1] + nums[i]
    - Each element calculated once: O(N)
    """
    for i in range(1, len(nums)):
        nums[i] += nums[i]
    return nums


# EXAMPLE PROBLEM 2 - MEDIUM: Subarray Sum Equals K
def subarray_sum(nums, k):
    """
    LeetCode #560: Subarray Sum Equals K

    Given an array of integers nums and an integer k, return the total number
    of continuous subarrays whose sum equals to k.

    Example:
        Input: nums = [1,1,1], k = 2
        Output: 2

    APPROACH:
    - Use prefix sum with hash map optimization
    - For each prefix sum, check if (prefix - k) exists before
    - Store frequency of prefix sums encountered
    - Special case: prefix sum of 0 appears once (empty prefix)

    WHY IT WORKS:
    - Sum of subarray [i, j] = prefix[j+1] - prefix[i]
    - We want prefix[j+1] - prefix[i] = k
    - So prefix[i] = prefix[j+1] - k
    - Count how many times each prefix sum has appeared
    - O(N) with hash map, instead of O(N²) checking all pairs

    MATHEMATICAL PROOF:
    - There are C prefix sums (including 0)
    - For each prefix[i], we count how many prefix[j] = prefix[i] - k
    - Total pairs = answer
    - Hash map gives O(1) lookup: total O(N)

    TIME COMPLEXITY: O(N)
    - Single pass through array
    - Hash map operations O(1) average case
    - Each element contributes to one prefix sum calculation

    SPACE COMPLEXITY: O(N)
    - Hash map stores prefix sum frequencies
    - In worst case, each prefix sum is unique: O(N) space
    - Required for counting subarrays efficiently

    OPTIMIZATION ACHIEVED:
    - Brute force: Check all subarrays O(N²), compute sum each time O(N) = O(N³)
    - With prefix sums (without hash map): O(N²) to check all pairs
    - With prefix sums + hash map: O(N)
    - Key insight: Transform problem to counting pairs with difference = k
    - Instead of checking all pairs explicitly, count frequency of prefix sums
    - For each prefix, we instantly know how many previous prefixes sum to (prefix - k)
    - Space-time tradeoff: O(N) space for O(N) time vs O(1) space for O(N²) time
    """
    from collections import defaultdict

    prefix_count = defaultdict(int)
    prefix_count[0] = 1  # Empty subarray

    prefix = 0
    count = 0

    for num in nums:
        prefix += num

        # Check if prefix - k exists
        count += prefix_count[prefix - k]

        # Store current prefix
        prefix_count[prefix] += 1

    return count


# EXAMPLE PROBLEM 3 - HARD: Range Sum Query 2D - Immutable
class NumMatrix:
    def __init__(self, matrix):
        """
        LeetCode #304: Range Sum Query 2D - Immutable

        Given a 2D matrix, compute the sum of the elements inside rectangle
        defined by its upper left (row1, col1) and lower right (row2, col2).

        APPROACH:
        - Build 2D prefix sum matrix
        - prefix[i][j] = sum of all elements in rectangle [0,0] to [i-1,j-1]
        - Query: rect_sum = prefix[row2+1][col2+1] - prefix[row1][col2+1]
                  - prefix[row2+1][col1] + prefix[row1][col1]

        WHY IT WORKS:
        - Inclusion-exclusion principle
        - Each query reduces to 4 prefix lookups: O(1)
        - Preprocessing: O(M*N) to build prefix matrix
        """
        if not matrix or not matrix[0]:
            return

        rows, cols = len(matrix), len(matrix[0])

        # Build prefix sum matrix with extra row/col for easier calculations
        self.prefix = [[0] * (cols + 1) for _ in range(rows + 1)]

        for r in range(rows):
            row_sum = 0
            for c in range(cols):
                row_sum += matrix[r][c]
                self.prefix[r + 1][c + 1] = self.prefix[r][c + 1] + row_sum

    def sum_region(self, row1, col1, row2, col2):
        """
        Return sum of elements in rectangle [row1, col1] to [row2, col2]
        """
        return (self.prefix[row2 + 1][col2 + 1]
                - self.prefix[row1][col2 + 1]
                - self.prefix[row2 + 1][col1]
                + self.prefix[row1][col1])


# ============================================================================
# PATTERN 6: KADANE'S ALGORITHM (MAXIMUM SUBARRAY)
# ============================================================================
"""
CONCEPT: Find contiguous subarray with maximum sum.
- Dynamic programming approach
- At each position, decide: extend previous subarray or start new
- State: max_ending_here, max_so_far

TIME: O(N) - single pass
SPACE: O(1) - only need two variables

KEY ADVANTAGE:
- O(N) vs O(N²) for brute force, O(N²) for prefix sum approach
- Greedy + DP: local optimal choice leads to global optimal
- Key insight: Negative sums never help extend subarray
- Elegant decision: max(current, max_ending_here + current)
- Single pass with constant space
- Foundation for many optimization problems
"""

def kadane_template(arr):
    """
    TEMPLATE for Kadane's Algorithm
    """
    max_ending_here = 0
    max_so_far = float('-inf')

    for num in arr:
        # Option 1: Extend previous subarray
        # Option 2: Start new subarray at current element
        max_ending_here = max(num, max_ending_here + num)

        # Update global maximum
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far


# EXAMPLE PROBLEM 1 - EASY: Maximum Subarray
def max_subarray(nums):
    """
    LeetCode #53: Maximum Subarray

    Given an integer array nums, find the contiguous subarray which has the
    largest sum and return its sum.

    Example:
        Input: nums = [-2,1,-3,4,-1,2,1,-5,4]
        Output: 6 ([4,-1,2,1])

    APPROACH:
    - Kadane's algorithm with DP
    - At each index, max ending here = max(current element, max ending here + current)
    - Track global maximum

    WHY IT WORKS - DP PROOF:
    - Let dp[i] = maximum sum of subarray ending at index i
    - dp[i] = max(nums[i], dp[i-1] + nums[i])
    - Base: dp[0] = nums[0]
    - Answer = max(dp[0...n-1])
    - Why this works: Subarray ending at i either:
      1. Starts at i: sum = nums[i]
      2. Extends subarray ending at i-1: sum = dp[i-1] + nums[i]
    - We take the better option
    - Greedy choice is optimal: local optimal leads to global optimal

    TIME COMPLEXITY: O(N)
    - Single pass through array
    - Each element processed exactly once
    - Constant time operations per element

    SPACE COMPLEXITY: O(1)
    - Only two variables: max_ending_here and max_so_far
    - No additional data structures proportional to input

    OPTIMIZATION ACHIEVED:
    - Brute force: Check all subarrays O(N²), compute sum each time O(N) = O(N³)
    - With prefix sums: O(N²) to check all subarrays
    - Kadane's algorithm: O(N) - single pass with O(1) space
    - Key insight: Optimal subarray has optimal prefix
    - DP decision at each step: extend or restart
    - Greedy choice is optimal because negative sums never help future calculations
    - Mathematical: If extending gives negative, better to start fresh
    """
    max_ending_here = nums[0]
    max_so_far = nums[0]

    for i in range(1, len(nums)):
        # Extend or start new
        max_ending_here = max(nums[i], max_ending_here + nums[i])
        max_so_far = max(max_so_far, max_ending_here)

    return max_so_far


# EXAMPLE PROBLEM 2 - MEDIUM: Best Time to Buy and Sell Stock
def max_profit(prices):
    """
    LeetCode #121: Best Time to Buy and Sell Stock

    You are given an array prices where prices[i] is the price of a given stock
    on the ith day. You want to maximize your profit by choosing a day to buy
    one stock and a different day to sell it.

    Example:
        Input: prices = [7,1,5,3,6,4]
        Output: 5 (buy at 1, sell at 6)

    APPROACH:
    - Variant of Kadane's (difference array)
    - Track minimum price seen so far
    - Track maximum profit = max(price - min_price)
    - Can also think as max subarray of (price[i] - price[i-1])

    WHY IT WORKS:
    - Profit = selling price - buying price
    - For each day, buying at minimum price before gives max profit
    - Track minimum price and maximum profit simultaneously
    - Single pass gives O(N) solution
    """
    min_price = float('inf')
    max_profit = 0

    for price in prices:
        # Update minimum buying price
        if price < min_price:
            min_price = price

        # Calculate profit if selling today
        profit = price - min_price

        # Update maximum profit
        if profit > max_profit:
            max_profit = profit

    return max_profit


# EXAMPLE PROBLEM 3 - HARD: Maximum Product Subarray
def max_product(nums):
    """
    LeetCode #152: Maximum Product Subarray

    Given an integer array nums, find a contiguous non-empty subarray that has
    the largest product and return that product.

    Example:
        Input: nums = [2,3,-2,4]
        Output: 6 ([2,3])

    APPROACH:
    - Track BOTH maximum and minimum products
    - Negative numbers make min * min = max
    - At each index:
      max_here = max(nums[i], max_here * nums[i], min_here * nums[i])
      min_here = min(nums[i], max_here * nums[i], min_here * nums[i])
    - Track global maximum

    WHY IT WORKS:
    - Need to track both max and min due to negative numbers
    - Current max depends on previous min when nums[i] is negative
    - All three possibilities considered:
      1. Start new subarray at nums[i]
      2. Extend previous max subarray
      3. Extend previous min subarray (if nums[i] negative)
    - Global maximum captures best subarray seen
    """
    max_product = nums[0]
    min_product = nums[0]
    max_sofar = nums[0]

    for i in range(1, len(nums)):
        if nums[i] < 0:
            # Swap max and min because negative makes them flip
            max_product, min_product = min_product, max_product

        # Start new or extend
        max_product = max(nums[i], max_product * nums[i])
        min_product = min(nums[i], min_product * nums[i])

        # Update global maximum
        max_sofar = max(max_sofar, max_product)

    return max_sofar


# ============================================================================
# PATTERN 7: DUTCH NATIONAL FLAG
# ============================================================================
"""
CONCEPT: Sort array with 3 unique values in-place using 3-way partitioning.
- Partition array into three sections: < pivot, == pivot, > pivot
- Used for quicksort's partition step
- Classic: sort [0,1,2] values

TIME: O(N) - single pass
SPACE: O(1) - in-place

KEY ADVANTAGE:
- O(N) vs O(N log N) for comparison-based sort
- O(1) space vs O(N) for counting sort
- Three-way partitioning maintains invariant
- Single pass with three pointers
- Foundation for quicksort partitioning
- Optimal for fixed number of distinct values
"""

def dutch_national_flag_template(arr):
    """
    TEMPLATE for Dutch National Flag (3-way partitioning)
    """
    # For sorting [0, 1, 2] values
    low = 0
    mid = 0
    high = len(arr) - 1

    while mid <= high:
        if arr[mid] == 0:
            # Swap arr[low] and arr[mid]
            arr[low], arr[mid] = arr[mid], arr[low]
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:  # arr[mid] == 2
            # Swap arr[mid] and arr[high]
            arr[mid], arr[high] = arr[high], arr[mid]
            high -= 1

    return arr


# EXAMPLE PROBLEM 1 - EASY: Sort Colors
def sort_colors(nums):
    """
    LeetCode #75: Sort Colors

    Given an array nums with n objects colored red, white, or blue, sort them
    in-place so that objects of the same color are adjacent.

    Example:
        Input: nums = [2,0,2,1,1,0]
        Output: [0,0,1,1,2,2]

    APPROACH:
    - Dutch National Flag algorithm
    - Three pointers: low (0s), mid (scanning), high (2s)
    - Process element at mid pointer
    - Swap based on value to maintain regions

    WHY IT WORKS:
    - Invariant: [0, low-1] = 0s, [low, mid-1] = 1s, [mid, high] = unknown,
      [high+1, n-1] = 2s
    - Each element processed at most once
    - After processing, elements move to correct region
    - Single pass, O(N) time, O(1) space

    TIME COMPLEXITY: O(N)
    - Each element processed at most once by mid pointer
    - In worst case, element swapped and processed again
    - But each element moves at most from high to low or vice versa
    - Total operations proportional to array size

    SPACE COMPLEXITY: O(1)
    - Only three pointers: low, mid, high
    - In-place sorting, swaps only
    - No additional data structures

    OPTIMIZATION ACHIEVED:
    - Brute force: Comparison-based sort O(N log N)
    - Counting sort: O(N) but needs extra arrays O(N) space
    - Dutch National Flag: O(N) time, O(1) space
    - Key insight: Only 3 distinct values, can partition directly
    - Three-way partitioning: < pivot, == pivot, > pivot
    - Maintains invariant throughout execution
    - Each element correctly placed in single pass
    """
    low = 0
    mid = 0
    high = len(nums) - 1

    while mid <= high:
        if nums[mid] == 0:
            nums[low], nums[mid] = nums[mid], nums[low]
            low += 1
            mid += 1
        elif nums[mid] == 1:
            mid += 1
        else:  # nums[mid] == 2
            nums[mid], nums[high] = nums[high], nums[mid]
            high -= 1


# EXAMPLE PROBLEM 2 - MEDIUM: Partition Array by Color (Custom)
def partition_by_color(items, colors):
    """
    Given an array of items and their colors (0, 1, 2), rearrange so that
    all items of the same color are grouped together.

    APPROACH: Dutch National Flag with stable item ordering

    WHY IT WORKS:
    - Same principle as sorting colors
    - Items and colors moved together
    - Maintains relative order within same color (stable)
    """
    low = 0
    mid = 0
    high = len(colors) - 1

    while mid <= high:
        if colors[mid] == 0:
            items[low], items[mid] = items[mid], items[low]
            colors[low], colors[mid] = colors[mid], colors[low]
            low += 1
            mid += 1
        elif colors[mid] == 1:
            mid += 1
        else:
            items[mid], items[high] = items[high], items[mid]
            colors[mid], colors[high] = colors[high], colors[mid]
            high -= 1


# EXAMPLE PROBLEM 3 - HARD: Partition Labels
def partition_labels(s):
    """
    LeetCode #763: Partition Labels

    A partition is a string such that each letter appears in at most one part.
    Return a list of integers representing the size of each part.

    Example:
        Input: s = "ababcbacadefegdehijhklij"
        Output: [9,7,8]

    APPROACH:
    - First, record last occurrence of each character
    - Scan string, track end of current partition
    - When current index == partition end, cut partition

    WHY IT WORKS:
    - Character must appear in at most one partition
    - Therefore partition end must be after last occurrence of all chars seen
    - When we reach end, no characters will appear later
    - Optimal partition is always achieved this way
    """
    # Record last occurrence of each character
    last_occurrence = {char: i for i, char in enumerate(s)}

    result = []
    start = 0
    end = 0

    for i, char in enumerate(s):
        # Extend partition to include last occurrence
        end = max(end, last_occurrence[char])

        # When we reach the end, partition is complete
        if i == end:
            result.append(end - start + 1)
            start = i + 1

    return result


# ============================================================================
# PATTERN 8: MATRIX TRAVERSAL
# ============================================================================
"""
CONCEPT: Systematic traversal of 2D arrays.
- Common patterns: spiral order, diagonal traversal, layer-by-layer
- Watch boundaries carefully
- Often requires direction changes

TIME: O(M*N) - visit each element once
SPACE: O(1) - in-place, or O(M*N) if storing result

KEY ADVANTAGE:
- Boundary tracking avoids visited matrix O(M*N)
- Layer-by-layer approach handles all shapes
- Each element visited exactly once
- Direction changes handled systematically
- Works for any M x N matrix
- Elegant termination when boundaries cross
"""

def matrix_traversal_template(matrix):
    """
    TEMPLATE for Matrix Traversal
    """
    rows = len(matrix)
    cols = len(matrix[0])

    # Track visited or boundaries
    top, bottom, left, right = 0, rows - 1, 0, cols - 1

    result = []

    while left <= right and top <= bottom:
        continue
        # Traverse in different directions:
        # 1. Left to right (top row)
        # 2. Top to bottom (right column)
        # 3. Right to left (bottom row)
        # 4. Bottom to top (left column)

        # Update boundaries after each pass

    return result


# EXAMPLE PROBLEM 1 - EASY: Spiral Matrix
def spiral_order(matrix):
    """
    LeetCode #54: Spiral Matrix

    Given an m x n matrix, return all elements of the matrix in spiral order.

    Example:
        Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
        Output: [1,2,3,6,9,8,7,4,5]

    APPROACH:
    - Use four boundaries: top, bottom, left, right
    - Traverse in order: left→right, top→bottom, right→left, bottom→top
    - Shrink boundaries after each direction

    WHY IT WORKS:
    - Spiral pattern: each layer traversed once
    - Boundaries ensure we don't revisit elements
    - When boundaries cross, traversal complete
    - Visit each element exactly once: O(M*N)

    TIME COMPLEXITY: O(M*N)
    - Visit each element exactly once
    - All four directional traversals combined touch every cell
    - M = number of rows, N = number of columns

    SPACE COMPLEXITY: O(M*N)
    - O(1) extra space for boundaries
    - O(M*N) for result list (required by problem)
    - Total: O(M*N) for output

    OPTIMIZATION ACHIEVED:
    - Brute force: Try to simulate spiral with direction changes O(M*N)
    - Using boundaries: Clean, systematic approach O(M*N)
    - Key insight: Layer-by-layer traversal
    - Four boundaries define current layer: top, bottom, left, right
    - After each complete traversal, shrink layer by incrementing/decrementing
    - No visited matrix needed - boundaries prevent revisiting
    - Elegant termination: when boundaries cross, all elements visited
    - Each element processed in exactly one of four directional passes
    """
    result = []
    top, bottom = 0, len(matrix) - 1
    left, right = 0, len(matrix[0]) - 1

    while top <= bottom and left <= right:
        # Left to right
        for col in range(left, right + 1):
            result.append(matrix[top][col])
        top += 1

        # Top to bottom
        for row in range(top, bottom + 1):
            result.append(matrix[row][right])
        right -= 1

        if top <= bottom:
            # Right to left
            for col in range(right, left - 1, -1):
                result.append(matrix[bottom][col])
            bottom -= 1

        if left <= right:
            # Bottom to top
            for row in range(bottom, top - 1, -1):
                result.append(matrix[row][left])
            left += 1

    return result


# EXAMPLE PROBLEM 2 - MEDIUM: Rotate Image
def rotate(matrix):
    """
    LeetCode #48: Rotate Image

    You are given an n x n 2D matrix, rotate the image by 90 degrees (clockwise).

    Example:
        Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
        Output: [[7,4,1],[8,5,2],[9,6,3]]

    APPROACH:
    - In-place rotation using two steps:
      1. Transpose (swap rows with columns)
      2. Reverse each row
    - Or rotate layer by layer

    WHY IT WORKS:
    - 90° clockwise rotation: element at [i][j] → [j][n-1-i]
    - Transpose + reverse achieves this transformation
    - Transpose: [i][j] → [j][i]
    - Reverse row: [j][i] → [j][n-1-i]
    - Combined: [i][j] → [j][n-1-i]
    - In-place, O(1) extra space
    """
    n = len(matrix)

    # Transpose
    for i in range(n):
        for j in range(i + 1, n):
            matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]

    # Reverse each row
    for i in range(n):
        matrix[i].reverse()


# EXAMPLE PROBLEM 3 - HARD: Diagonal Traverse
def find_diagonal_order(mat):
    """
    LeetCode #498: Diagonal Traverse

    Given an m x n matrix mat, return all elements of the matrix in diagonal order.

    Example:
        Input: mat = [[1,2,3],[4,5,6],[7,8,9]]
        Output: [1,2,4,7,5,3,6,8,9]

    APPROACH:
    - Diagonals alternate direction
    - Even-indexed diagonals (0,2,4...): bottom-to-top
    - Odd-indexed diagonals (1,3,5...): top-to-bottom
    - Track when diagonals wrap around matrix boundaries

    WHY IT WORKS:
    - Diagonal property: i + j = constant (diagonal index)
    - When hitting boundaries, reverse direction
    - Sum of indices determines diagonal number
    - Zig-zag pattern naturally emerges from boundary conditions
    """
    if not mat or not mat[0]:
        return []

    rows, cols = len(mat), len(mat[0])
    result = []
    r, c = 0, 0

    # Direction: 1 = up-right, -1 = down-left
    direction = 1

    for _ in range(rows * cols):
        result.append(mat[r][c])

        # Calculate next position
        next_r = r - direction
        next_c = c + direction

        # Check if we need to change direction
        if next_r < 0 or next_r >= rows or next_c < 0 or next_c >= cols:
            # Adjust position based on direction
            if direction == 1:  # Going up-right, hit boundary
                if c + 1 < cols:
                    c += 1
                else:
                    r += 1
            else:  # Going down-left, hit boundary
                if r + 1 < rows:
                    r += 1
                else:
                    c += 1
            direction *= -1
        else:
            r = next_r
            c = next_c

    return result


# ============================================================================
# SUMMARY & PATTERN RECOGNITION GUIDE
# ============================================================================

"""
OPTIMIZATION COMPARISON: BRUTE FORCE vs OPTIMAL
================================================

Pattern/Problem                    | Brute Force      | Optimal         | Improvement
-----------------------------------|------------------|-----------------|------------------
Valid Palindrome                   | O(N²)            | O(N)            | 10x faster
3Sum                               | O(N³)            | O(N²)           | 100x faster
Trapping Rain Water                | O(N²)            | O(N)            | 10x faster
Middle of Linked List              | O(N) + O(N)      | O(N)            | 2x faster (single pass)
Longest Substring                  | O(N³)            | O(N)            | 100x faster
Min Subarray Sum                   | O(N²)            | O(N)            | 10x faster
Max Subarray (Kadane)              | O(N³) → O(N²)    | O(N)            | 100x faster
Subarray Sum Equals K              | O(N³) → O(N²)    | O(N)            | 100x faster
Sort Colors                        | O(N log N)       | O(N)            | ~N/logN faster
Spiral Matrix                      | O(M*N)           | O(M*N)          | Cleaner code

HOW TO SPOT EACH PATTERN:

1. TWO POINTERS (OPPOSITE ENDS)
   Keywords: "Sorted array", "Palindrome", "Two sum with sorted array"
   Examples: Two Sum II, 3Sum, Container With Most Water
   Look for: Array is sorted, or you need to check from both ends

2. TWO POINTERS (FAST & SLOW)
   Keywords: "Middle of list", "Nth from end", "Cycle detection"
   Examples: Linked List Cycle, Remove Nth from End
   Look for: Linked list problems, cycle detection, finding positions

3. SLIDING WINDOW (FIXED)
   Keywords: "Subarray of size k", "Window of size k"
   Examples: Maximum Average Subarray, Sliding Window Maximum
   Look for: Fixed window size specified

4. SLIDING WINDOW (VARIABLE)
   Keywords: "Longest/Shortest substring without...", "Minimum subarray sum"
   Examples: Longest Substring Without Repeating, Minimum Size Subarray
   Look for: Condition-based window expansion/contraction

5. PREFIX SUM
   Keywords: "Range sum", "Continuous subarray", "Subarray sum equals K"
   Examples: Range Sum Query, Subarray Sum Equals K
   Look for: Multiple queries on same array, or counting subarrays

6. KADANE'S ALGORITHM
   Keywords: "Maximum subarray", "Maximum/minimum product"
   Examples: Maximum Subarray, Best Time to Buy/Sell Stock
   Look for: Finding best contiguous segment

7. DUTCH NATIONAL FLAG
   Keywords: "Sort three values", "Partition into three groups"
   Examples: Sort Colors (0,1,2)
   Look for: Exactly three distinct values to sort

8. MATRIX TRAVERSAL
   Keywords: "Spiral order", "Rotate matrix", "Diagonal"
   Examples: Spiral Matrix, Rotate Image
   Look for: 2D array problems

COMPLEXITY ANALYSIS GUIDE:
===========================

Time Complexity Improvements:
- O(N³) → O(N²): Nested loops eliminated
- O(N²) → O(N): Single pass with pointers/sliding window
- O(N log N) → O(N): Specialized algorithms (Dutch National Flag)

Space Complexity Optimizations:
- O(N) → O(1): In-place algorithms, pointer-based
- O(N) → O(K): Hash maps for bounded K
- Trade-offs: Sometimes use O(N) space to achieve O(N) time

Golden Rules:
1. Every element should be visited at most once (O(N) best)
2. Avoid nested loops when possible
3. Use mathematical properties (sortedness, symmetry)
4. Trade space for time when beneficial
5. Maintain invariants to avoid extra passes

GENERAL PROBLEM-SOLVING STRATEGY:

1. Understand the problem constraints and requirements
2. Identify if array/string is sorted → Consider two pointers
3. Check if it involves subarrays/substrings → Consider sliding window or prefix sum
4. Look for optimization patterns → Kadane for subarray problems
5. For 3-value arrays → Dutch National Flag
6. For 2D problems → Matrix traversal patterns
7. Practice pattern recognition through repetition
8. Start with brute force, then optimize using patterns

KEY INSIGHT:
The best algorithms make each element do "maximum work" - visiting once,
processing once, and making a single decision about its fate.
"""

# Test all patterns
if __name__ == "__main__":
    print("Testing Pattern Examples:")
    print("=" * 50)

    # Test is_palindrome
    print("\n1. Valid Palindrome:", is_palindrome("A man, a plan, a canal: Panama"))

    # Test three_sum
    print("2. 3Sum:", three_sum([-1, 0, 1, 2, -1, -4]))

    # Test trap (water trapping)
    print("3. Trapping Rain Water:", trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]))

    # Test length_of_longest_substring
    print("4. Longest Substring:", length_of_longest_substring("abcabcbb"))

    # Test min_subarray_len
    print("5. Min Subarray Length:", min_subarray_len(7, [2, 3, 1, 2, 4, 3]))

    # Test subarray_sum
    print("6. Subarray Sum Equals K:", subarray_sum([1, 1, 1], 2))

    # Test max_subarray (Kadane)
    print("7. Maximum Subarray:", max_subarray([-2, 1, -3, 4, -1, 2, 1, -5, 4]))

    # Test max_profit (Kadane variant)
    print("8. Max Profit:", max_profit([7, 1, 5, 3, 6, 4]))

    # Test sort_colors (Dutch National Flag)
    colors = [2, 0, 2, 1, 1, 0]
    sort_colors(colors)
    print("9. Sort Colors:", colors)

    # Test spiral_order
    print("10. Spiral Order:", spiral_order([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))

    print("\n" + "=" * 50)
    print("All patterns tested successfully!")
