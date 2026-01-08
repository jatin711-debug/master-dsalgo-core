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
    Given the head of a singly linked list, reverse the list, and return the reversed list.

    Example:
        Input: head = [1,2,3,4,5]
        Output: [5,4,3,2,1]
    
    APPROACH:
    - Iterative In-Place Reversal.
    - Maintain 3 pointers: `prev` (initially None), `curr` (initially head), `next_temp`.
    - Loop logic:
      1. Save `next` node (`next_temp = curr.next`).
      2. Point `curr` backwards (`curr.next = prev`).
      3. Move `prev` forward (`prev = curr`).
      4. Move `curr` forward (`curr = next_temp`).
    
    WHY IT WORKS:
    - We essentially flip the arrows one by one while traversing.
    - `prev` eventually becomes the new head (the last element).
    
    TIME COMPLEXITY: O(N)
    - Visit every node exactly once.

    SPACE COMPLEXITY: O(1)
    - Only 3 pointers used regardless of list size.
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
    Given the head of a singly linked list, return the middle node of the linked list.
    
    Example:
        Input: [1,2,3,4,5]
        Output: Node with value 3
    
    APPROACH:
    - Two Pointers Strategy (Tortoise and Hare).
    - `slow` moves 1 step at a time.
    - `fast` moves 2 steps at a time.
    - When `fast` reaches the end (or None), `slow` will be exactly at the midpoint.

    WHY IT WORKS:
    - Distance covered by fast = 2 * Distance covered by slow.
    - If fast covers L (length of list), slow covers L/2.
    
    TIME COMPLEXITY: O(N)
    - One pass. `fast` traverses the list once.

    SPACE COMPLEXITY: O(1)
    - No extra space allocated.
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
    Given head, determine if the linked list has a cycle in it.
    
    APPROACH:
    - Floyd’s Cycle Finding Algorithm (Tortoise and Hare).
    - `slow` moves 1 step. `fast` moves 2 steps.
    - If there is a cycle, `fast` will eventually lap `slow` (they will be equal).
    - If `fast` reaches None, there is no cycle.

    WHY IT WORKS:
    - If there's a loop, the fast runner is "chasing" the slow runner inside the loop.
    - The distance between them decreases by 1 in every step.
    - Eventually distance becomes 0 (collision).

    TIME COMPLEXITY: O(N)
    - If no cycle: O(N) to reach end.
    - If cycle: O(N + K) where K is cycle length. Linear.

    SPACE COMPLEXITY: O(1)
    - Standard two pointer approach.
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
    Merge two sorted linked lists and return it as a sorted list.
    
    Example:
        Input: l1 = [1,2,4], l2 = [1,3,4]
        Output: [1,1,2,3,4,4]
    
    APPROACH:
    - Use a `Dummy` node. This acts as a placeholder for the start of the new list.
    - Maintain a `curr` pointer starting at `dummy`.
    - While both lists have nodes:
      - Compare `l1.val` and `l2.val`.
      - Build the link: `curr.next` = smaller node.
      - Advance the chosen list pointer and `curr`.
    - Attach the remaining non-empty list at the end.

    WHY IT WORKS:
    - Dummy head handles edge cases (like empty lists) cleanly without needing special 'if head is None' logic for initialization.
    - Similar to merge sort's merge step.

    TIME COMPLEXITY: O(N + M)
    - Iterate through both lists once.

    SPACE COMPLEXITY: O(1)
    - We are splicing together existing nodes, not creating new ones (except dummy).
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
    
    Example:
        Input: head = [1,2,3,4,5], n = 2
        Output: [1,2,3,5]
    
    APPROACH:
    - Two Pointers with a Gap.
    - Use Dummy node to handle edge case of removing the HEAD itself.
    - Move `fast` pointer n+1 steps ahead.
    - Move `slow` and `fast` together until `fast` hits the end.
    - `slow` is now pointing to the node BEFORE the target.
    - Delete target: `slow.next = slow.next.next`.

    WHY IT WORKS:
    - By creating a gap of size `n`, when `fast` is at the end (Null), `slow` is `n` nodes from the end.
    - The extra +1 step is to land `slow` on the *previous* node to facilitate deletion logic.
    
    TIME COMPLEXITY: O(N)
    - Single pass.

    SPACE COMPLEXITY: O(1)
    - In-place modification.
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
