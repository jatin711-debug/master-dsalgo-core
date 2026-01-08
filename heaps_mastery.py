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
    Return the kth largest element in the array.
    
    Example:
        Input: nums = [3,2,1,5,6,4], k = 2
        Output: 5
    
    APPROACH:
    - Use a Min-Heap of fixed size `k`.
    - Iterate through all numbers. Pushthe current number into the heap.
    - If the heap size exceeds `k`, pop the smallest element (remove the minimum).
    - After processing all elements, the heap contains the `k` largest elements. The root of the min-heap (index 0) is the smallest of the top `k`, which is exactly the Kth largest.
    
    WHY IT WORKS:
    - By discarding the smallest elements once we have more than K, we end up keeping only the "heavyweights".
    
    TIME COMPLEXITY: O(N * log K)
    - We perform N insertions into a heap of size K.

    SPACE COMPLEXITY: O(K)
    - To store the heap elements.
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
    Merge k sorted linked lists and return it as one sorted list.
    
    APPROACH:
    - Use a Min-Heap to track the smallest node currently available among all K list heads.
    - Initially push the head of every list into the heap.
    - While heap is not empty:
      1. Pop the smallest node. Add it to our result list.
      2. If the popped node has a `.next`, push that next node into the heap.
    
    WHY IT WORKS:
    - Since all sub-lists are sorted, the next smallest element must be one of the heads. A Min-Heap gives us access to this Minimum in O(1) and removal in O(log K).
    
    TIME COMPLEXITY: O(N * log K)
    - N is total number of nodes across all lists.
    - K is the number of linked lists (heap size).
    
    SPACE COMPLEXITY: O(K)
    - Heap holds at most K elements at any time.
    """
    min_heap = []
    
    for i, l in enumerate(lists):
        if l:
            # i serves as tie breaker so we don't compare ListNode objects directly
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
    Design a structure to add numbers and find the median efficiently.
    
    APPROACH:
    - Maintain two heaps:
      1. `small_half`: A Max-Heap storing the smaller 50% of numbers. (Python has only Min-Heap, so store negatives).
      2. `large_half`: A Min-Heap storing the larger 50% of numbers.
    - Property: `max(small_half) <= min(large_half)`.
    - Balance: Sizes should not differ by more than 1.
    
    WHY IT WORKS:
    - The median is either the peak of the small half, the peak of the large half, or the average of both.
    - Heaps allow O(log N) insertion and O(1) access to these peaks.
    
    TIME COMPLEXITY:
    - addNum: O(log N)
    - findMedian: O(1)
    
    SPACE COMPLEXITY: O(N)
    - To store all elements.
    """
    def __init__(self):
        self.small_half = [] # Max Heap (invert values)
        self.large_half = [] # Min Heap

    def addNum(self, num: int) -> None:
        # Provide to small half first (Max-Heap)
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
