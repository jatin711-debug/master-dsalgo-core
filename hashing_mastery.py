# ============================================================================
# MODULE 2: HASHING (HASH MAPS & SETS)
# ============================================================================
"""
The "Utility Knife" of Data Structures.
Trade O(N) space for O(1) access time.

COMMON PATTERNS:
1. Two Sum Variation: Check if (Target - Current) exists in map.
2. Frequency Counting: Store count of elements.
3. Grouping/Anagrams: Key sorted string or tuple -> Value list of strings.
4. Prefix Sum + Hash Map: Store (PrefixSum -> Count/Index) to find subarrays.
5. Set Lookup: O(1) check for existence (e.g., Longest Consecutive Sequence).

TIME COMPLEXITY: O(1) average for insert/lookup.
SPACE COMPLEXITY: O(N) to store elements.
"""

from collections import defaultdict, Counter

# ============================================================================
# PATTERN 1: TWO SUM VARIATION
# ============================================================================
def two_sum(nums, target):
    """
    LeetCode #1: Two Sum
    Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

    Example:
        Input: nums = [2,7,11,15], target = 9
        Output: [0,1]
    
    APPROACH:
    - Use a Hash Map to store `value -> index` mapping as we iterate.
    - For each `num`, check if `complement = target - num` exists in the map.
    - If yes, we found the pair! Return current index and map[complement].
    - Otherwise, store `num` in map.

    WHY IT WORKS:
    - By storing elements seen so far, we can look "backwards" in O(1) time.
    - `a + b = target` -> `a = target - b`.
    
    TIME COMPLEXITY: O(N)
    - Single pass through the array.
    - Map lookup is O(1) on average.

    SPACE COMPLEXITY: O(N)
    - Worst case: we store all N elements in the hash map.
    """
    seen = {} # val -> index
    
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# ============================================================================
# PATTERN 2: GROUPING / ANAGRAMS
# ============================================================================
def group_anagrams(strs):
    """
    LeetCode #49: Group Anagrams
    Given an array of strings strs, group the anagrams together.
    
    Example:
        Input: strs = ["eat","tea","tan","ate","nat","bat"]
        Output: [["bat"],["nat","tan"],["ate","eat","tea"]]
    
    APPROACH:
    - Anagrams share the exact same character counts.
    - We need a canonical representation (key) for each group.
    - Option 1: Sort the string ("ate" -> "aet", "tea" -> "aet"). O(K log K).
    - Option 2: Character count tuple (a=1, e=1, t=1...). O(K).
    - Map Key: Tuple of counts (requires tuple for immutability).
    - Map Value: List of original strings.

    WHY IT WORKS:
    - All anagrams map to the same key because they have identical character frequencies.
    - Hash Map automatically groups them.

    TIME COMPLEXITY: O(N * K)
    - N is number of strings, K is max length of a string.
    - Counting chars takes O(K). we do this for N strings.

    SPACE COMPLEXITY: O(N * K)
    - Storing all strings and keys in the map.
    """
    groups = defaultdict(list)
    
    for s in strs:
        # Key: tuple of counts (26 chars)
        count = [0] * 26
        for c in s:
            count[ord(c) - ord('a')] += 1
        
        # Lists are not hashable, use tuple
        key = tuple(count)
        groups[key].append(s)
        
    return list(groups.values())

# ============================================================================
# PATTERN 3: LONGEST CONSECUTIVE SEQUENCE (SET LOOKUP)
# ============================================================================
def longest_consecutive(nums):
    """
    LeetCode #128: Longest Consecutive Sequence
    Given an unsorted array of integers nums, return the length of the longest consecutive elements sequence.
    Example: [100, 4, 200, 1, 3, 2] -> [1, 2, 3, 4] -> 4
    
    APPROACH:
    - Convert array to a Set for O(1) lookups.
    - Iterate through the set.
    - Key Insight: Only attempt to build a sequence if the current number is the START of a sequence.
    - How do we know if `x` is the start? If `x - 1` is NOT in the set.
    - If it is the start, keep checking `x+1, x+2...` until the sequence breaks.

    WHY IT WORKS:
    - We avoid redundant work. For sequence [1,2,3,4], we only process it when we see '1'.
    - When we visit '2', we see '1' exists, so we skip it.
    - This ensures every element is visited at most twice (once in outer loop, once in inner loop).

    TIME COMPLEXITY: O(N)
    - Set creation: O(N).
    - Outer loop: O(N).
    - Inner loop runs only when start of sequence found. Total inner loop iterations across entire runtime is O(N).
    
    SPACE COMPLEXITY: O(N)
    - To store the set.
    """
    num_set = set(nums)
    longest = 0
    
    for num in num_set:
        # Check if 'num' is the start of a sequence
        if (num - 1) not in num_set:
            current_num = num
            current_streak = 1
            
            while (current_num + 1) in num_set:
                current_num += 1
                current_streak += 1
            
            longest = max(longest, current_streak)
            
    return longest

# ============================================================================
# PATTERN 4: SUBARRAY SUM EQUALS K (PREFIX SUM + HASH MAP)
# ============================================================================
def subarray_sum_k(nums, k):
    """
    LeetCode #560: Subarray Sum Equals K
    Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
    
    Example:
        Input: nums = [1,1,1], k = 2
        Output: 2 ([1,1] at start, [1,1] at end)

    APPROACH:
    - Use Prefix Sum concept: Sum(i, j) = PrefixSum[j] - PrefixSum[i-1].
    - We want Sum(i, j) == k.
    - Therefore: `PrefixSum[j] - PrefixSum[i-1] == k`
    - Rearranging: `PrefixSum[i-1] == PrefixSum[j] - k`
    - As we iterate (calculating current prefix sum for 'j'), we check if `current_sum - k` has appeared before.
    - Use a Hash Map `prefix_map` to store `{ sum : count_of_occurrences }`.

    WHY IT WORKS:
    - If `current_sum - k` exists in our map 3 times, it means there are 3 distinct start points that result in a subarray sum of k ending at current index.
    
    TIME COMPLEXITY: O(N)
    - Single pass. Map operations are O(1).

    SPACE COMPLEXITY: O(N)
    - Map can store up to N distinct prefix sums.
    """
    count = 0
    current_sum = 0
    prefix_map = {0: 1} # sum -> count (0 sum exists once effectively)
    
    for num in nums:
        current_sum += num
        # Check if valid prefix exists
        diff = current_sum - k
        if diff in prefix_map:
            count += prefix_map[diff]
            
        # Add current sum to map (get(key, 0) + 1)
        prefix_map[current_sum] = prefix_map.get(current_sum, 0) + 1
        
    return count

if __name__ == "__main__":
    print("Testing Hashing Pattern:")
    print("-" * 30)
    print("1. Two Sum ([2,7,11,15], 9):", two_sum([2,7,11,15], 9))
    print("2. Group Anagrams (['eat','tea','tan','ate','nat','bat']):", group_anagrams(["eat","tea","tan","ate","nat","bat"]))
    print("3. Longest Consecutive ([100,4,200,1,3,2]):", longest_consecutive([100,4,200,1,3,2]))
    print("4. Subarray Sum K ([1,1,1], 2):", subarray_sum_k([1,1,1], 2))
