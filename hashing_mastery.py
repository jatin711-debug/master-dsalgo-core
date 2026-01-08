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
    
    Approach: Hash Map
    - Map value -> index
    - For each num, check if (target - num) is in map.
    
    Time: O(N)
    Space: O(N)
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
    
    Approach:
    - Use a tuple of character counts (or sorted string) as key.
    - Value is list of strings matching that key.
    
    Time: O(N * K) where K is max string length (using count array)
    Space: O(N * K)
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
    Given unsorted array, find length of longest consecutive elements sequence.
    Example: [100, 4, 200, 1, 3, 2] -> [1, 2, 3, 4] -> 4
    
    Approach:
    - Put all nums in a Set for O(1) lookup.
    - Only start counting if (num - 1) is NOT in set (Start of sequence).
    - While (num + 1) in set, increment.
    
    Time: O(N) - distinct elements visited at most twice
    Space: O(N)
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
    
    Approach:
    - Sum(i, j) = PrefixSum[j] - PrefixSum[i-1]
    - We want Sum(i, j) = k
    - So, PrefixSum[j] - PrefixSum[i-1] = k
    - PrefixSum[i-1] = PrefixSum[j] - k
    - Store count of prefix sums seen so far.
    
    Time: O(N)
    Space: O(N)
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
