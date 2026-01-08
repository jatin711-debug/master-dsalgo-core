# ============================================================================
# MODULE 7: HEAPS (PRIORITY QUEUE)
# ============================================================================
"""
Ordering based on priority (Min or Max).

COMMON PATTERNS:
1. Top 'K' Elements: Keep a heap of size K.
2. Merge 'K' Sorted Lists: Always pick min from heads of lists.
3. Two Heaps: Median in data stream (Balance Min and Max heap).
4. Scheduling: Greedily pick task with highest priority.

TIME COMPLEXITY: O(log K) for push/pop. O(N log K) for processing N items.
SPACE COMPLEXITY: O(N) or O(K).
"""

import heapq

# ============================================================================
# PATTERN 1: TOP K ELEMENTS
# ============================================================================
def find_kth_largest(nums, k):
    """
    LeetCode #215: Kth Largest Element in an Array
    
    Approach: Min-Heap of size K.
    - Push elements.
    - If size > K, pop min.
    - Remaining min is the Kth largest.
    """
    min_heap = []
    
    for num in nums:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)
            
    return min_heap[0]

# ============================================================================
# PATTERN 2: MERGE K SORTED LISTS
# ============================================================================
# Definition must be included if not importing
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_k_lists(lists):
    """
    LeetCode #23: Merge k Sorted Lists
    
    Approach:
    - Push head of every list into Min-Heap.
    - Tuple (val, index, node) to handle tie-breaking.
    - Pop min, append to result, push next of popped node.
    
    Time: O(N log K)
    """
    min_heap = []
    
    for i, l in enumerate(lists):
        if l:
            # i serves as tie breaker
            heapq.heappush(min_heap, (l.val, i, l))
            
    dummy = ListNode(0)
    curr = dummy
    
    while min_heap:
        val, i, node = heapq.heappop(min_heap)
        curr.next = node
        curr = curr.next
        
        if node.next:
            heapq.heappush(min_heap, (node.next.val, i, node.next))
            
    return dummy.next

# ============================================================================
# PATTERN 3: TWO HEAPS (MEDIAN)
# ============================================================================
class MedianFinder:
    """
    LeetCode #295: Find Median from Data Stream
    
    Approach:
    - Max-Heap (small_half): Stores smaller half of numbers.
    - Min-Heap (large_half): Stores larger half of numbers.
    - Balance sizes so diff is at most 1.
    """
    def __init__(self):
        self.small_half = [] # Max Heap (invert values)
        self.large_half = [] # Min Heap

    def addNum(self, num: int) -> None:
        # Provide to small half first
        heapq.heappush(self.small_half, -num)
        
        # Ensure max of small <= min of large
        if self.small_half and self.large_half and (-self.small_half[0] > self.large_half[0]):
            val = -heapq.heappop(self.small_half)
            heapq.heappush(self.large_half, val)
            
        # Balance sizes
        if len(self.small_half) > len(self.large_half) + 1:
            val = -heapq.heappop(self.small_half)
            heapq.heappush(self.large_half, val)
        if len(self.large_half) > len(self.small_half) + 1:
            val = heapq.heappop(self.large_half)
            heapq.heappush(self.small_half, -val)

    def findMedian(self) -> float:
        if len(self.small_half) > len(self.large_half):
            return -self.small_half[0]
        if len(self.large_half) > len(self.small_half):
            return self.large_half[0]
        
        return (-self.small_half[0] + self.large_half[0]) / 2.0

if __name__ == "__main__":
    print("Testing Heaps:")
    print("-" * 30)
    print("1. Kth Largest ([3,2,1,5,6,4], k=2):", find_kth_largest([3,2,1,5,6,4], 2))
    
    mf = MedianFinder()
    mf.addNum(1)
    mf.addNum(2)
    print("2. Median (1,2):", mf.findMedian())
    mf.addNum(3)
    print("2. Median (1,2,3):", mf.findMedian())
