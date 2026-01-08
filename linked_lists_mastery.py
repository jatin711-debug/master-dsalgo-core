# ============================================================================
# MODULE 3: LINKED LISTS
# ============================================================================
"""
Pointer manipulation.
Difficulty usually handling edge cases (nulls) and reference management.

COMMON PATTERNS:
1. Fast & Slow Pointers (Tortoise & Hare): Cycle detection, finding middle.
2. In-place Reversal: Reverse whole list or sub-segment.
3. Dummy Head: Sentinel node to handle edge cases (deleting head).
4. Merge Technique: Merging sorted lists.

TIME COMPLEXITY: Usually O(N).
SPACE COMPLEXITY: O(1) (In-place).
"""

# START: Helper Code for Linked List
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def create_linked_list(arr):
    if not arr: return None
    head = ListNode(arr[0])
    curr = head
    for x in arr[1:]:
        curr.next = ListNode(x)
        curr = curr.next
    return head

def print_linked_list(head):
    arr = []
    curr = head
    while curr:
        arr.append(curr.val)
        curr = curr.next
    return arr
# END: Helper Code

# ============================================================================
# PATTERN 1: IN-PLACE REVERSAL
# ============================================================================
def reverse_list(head):
    """
    LeetCode #206: Reverse Linked List
    
    Approach:
    - Three pointers: prev, curr, next_temp.
    - Save next, flip arrow, move forward.
    
    Time: O(N)
    Space: O(1)
    """
    prev = None
    curr = head
    while curr:
        next_temp = curr.next # Save next
        curr.next = prev      # Reverse
        prev = curr           # Move prev
        curr = next_temp      # Move curr
    return prev

# ============================================================================
# PATTERN 2: FAST & SLOW POINTERS (MIDDLE)
# ============================================================================
def middle_node(head):
    """
    LeetCode #876: Middle of the Linked List
    
    Approach:
    - Slow moves 1 step.
    - Fast moves 2 steps.
    - When Fast reaches end, Slow is at middle.
    
    Time: O(N)
    Space: O(1)
    """
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

# ============================================================================
# PATTERN 3: FAST & SLOW POINTERS (CYCLE)
# ============================================================================
def has_cycle(head):
    """
    LeetCode #141: Linked List Cycle
    
    Approach:
    - If Fast meets Slow, there is a cycle.
    - If Fast reaches None, no cycle.
    
    Time: O(N)
    Space: O(1)
    """
    slow = head
    fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# ============================================================================
# PATTERN 4: DUMMY HEAD (MERGE)
# ============================================================================
def merge_two_lists(l1, l2):
    """
    LeetCode #21: Merge Two Sorted Lists
    
    Approach:
    - Use Dummy Node to simplify head handling.
    - Compare heads of l1 and l2, append smaller to current.
    
    Time: O(N + M)
    Space: O(1)
    """
    dummy = ListNode(-1)
    curr = dummy
    
    while l1 and l2:
        if l1.val < l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
        
    # Append remainder
    curr.next = l1 if l1 else l2
    
    return dummy.next

# ============================================================================
# PATTERN 5: REMOVE N-TH NODE (TWO POINTERS + DUMMY)
# ============================================================================
def remove_nth_from_end(head, n):
    """
    LeetCode #19: Remove Nth Node From End of List
    
    Approach:
    - Use Dummy to handle removing head.
    - Move Fast n+1 steps ahead.
    - Move Fast and Slow until Fast hits end.
    - Slow is now before the target node.
    """
    dummy = ListNode(0, head)
    slow = dummy
    fast = dummy
    
    # Move fast n+1 steps to create gap
    for _ in range(n + 1):
        fast = fast.next
        
    # Move both
    while fast:
        slow = slow.next
        fast = fast.next
        
    # Skip node
    slow.next = slow.next.next
    
    return dummy.next

if __name__ == "__main__":
    print("Testing Linked Lists:")
    print("-" * 30)
    
    # Reverse
    h1 = create_linked_list([1,2,3,4,5])
    print("1. Reverse ([1,2,3,4,5]):", print_linked_list(reverse_list(h1)))
    
    # Middle
    h2 = create_linked_list([1,2,3,4,5])
    print("2. Middle ([1,2,3,4,5]):", middle_node(h2).val)
    
    # Merge
    l1 = create_linked_list([1,2,4])
    l2 = create_linked_list([1,3,4])
    merged = merge_two_lists(l1, l2)
    print("3. Merge ([1,2,4], [1,3,4]):", print_linked_list(merged))
