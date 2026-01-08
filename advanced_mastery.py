# ============================================================================
# MODULE 10: ADVANCED / NICHE
# ============================================================================
"""
Tries, Bit Manipulation, and Advanced Binary Search.

COMMON PATTERNS:
1. Trie (Prefix Tree): Autocomplete, word validation.
2. Bit Manipulation: XOR for unique numbers, Masking.
3. Modified Binary Search: Rotated arrays.
4. Binary Search on Answer: Solution space is sorted (Monotonic).

TIME COMPLEXITY: O(L) for Trie, O(log N) for BS, O(1) for Bit ops.
"""

# ============================================================================
# PATTERN 1: TRIE (PREFIX TREE)
# ============================================================================
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    """
    LeetCode #208: Implement Trie (Prefix Tree)
    """
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

# ============================================================================
# PATTERN 2: BIT MANIPULATION (SINGLE NUMBER)
# ============================================================================
def single_number(nums):
    """
    LeetCode #136: Single Number
    
    Approach: XOR.
    - A ^ A = 0
    - A ^ 0 = A
    - A ^ B ^ A = B
    """
    res = 0
    for num in nums:
        res ^= num
    return res

# ============================================================================
# PATTERN 3: MODIFIED BINARY SEARCH (ROTATED ARRAY)
# ============================================================================
def search_rotated(nums, target):
    """
    LeetCode #33: Search in Rotated Sorted Array
    
    Approach:
    - One half is always sorted.
    - Check if target is in the sorted half.
    """
    l, r = 0, len(nums) - 1
    
    while l <= r:
        mid = (l + r) // 2
        if nums[mid] == target:
            return mid
            
        # Left half sorted?
        if nums[l] <= nums[mid]:
            if nums[l] <= target < nums[mid]:
                r = mid - 1
            else:
                l = mid + 1
        # Right half sorted?
        else:
            if nums[mid] < target <= nums[r]:
                l = mid + 1
            else:
                r = mid - 1
    return -1

# ============================================================================
# PATTERN 4: BINARY SEARCH ON ANSWER
# ============================================================================
def min_eating_speed(piles, h):
    """
    LeetCode #875: Koko Eating Bananas
    
    Approach:
    - We want min speed 'k'. Range of k is [1, max(piles)].
    - 'Can we finish in h hours with speed k?' is a monotone function.
    - Use BS to find first 'k' that works.
    """
    def can_finish(k):
        hours = 0
        for p in piles:
            # ceil(p / k) => (p + k - 1) // k
            hours += (p + k - 1) // k
        return hours <= h
        
    l, r = 1, max(piles)
    res = r
    
    while l <= r:
        k = (l + r) // 2
        if can_finish(k):
            res = k
            r = k - 1 # Try smaller speed
        else:
            l = k + 1 # Need faster speed
            
    return res

if __name__ == "__main__":
    print("Testing Advanced:")
    print("-" * 30)
    
    trie = Trie()
    trie.insert("apple")
    print("1. Trie Search (apple):", trie.search("apple"))
    print("1. Trie Prefix (app):", trie.startsWith("app"))
    
    print("2. Single Number ([4,1,2,1,2]):", single_number([4,1,2,1,2]))
    
    print("3. Rotated Search ([4,5,6,7,0,1,2], 0):", search_rotated([4,5,6,7,0,1,2], 0))
    
    print("4. Koko Bananas ([3,6,7,11], h=8):", min_eating_speed([3,6,7,11], 8))
