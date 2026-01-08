# ============================================================================
# MODULE 6: TREES (BINARY TREES & BST)
# ============================================================================
"""
Hierarchical data. Recursion is king here.

COMMON PATTERNS:
1. DFS: Pre-order, In-order, Post-order.
2. BFS: Level Order Traversal (Queue).
3. BST Property: Left < Root < Right.
4. LCA: Lowest Common Ancestor.
5. Construction: Rebuilding from traversal arrays.

TIME COMPLEXITY: O(N) (visit every node).
SPACE COMPLEXITY: O(H) (Height of tree, recursion stack).
"""

from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# Helper to create tree from list (BFS order)
def create_tree(arr):
    if not arr: return None
    root = TreeNode(arr[0])
    queue = deque([root])
    i = 1
    while queue and i < len(arr):
        node = queue.popleft()
        if i < len(arr) and arr[i] is not None:
            node.left = TreeNode(arr[i])
            queue.append(node.left)
        i += 1
        if i < len(arr) and arr[i] is not None:
            node.right = TreeNode(arr[i])
            queue.append(node.right)
        i += 1
    return root

# ============================================================================
# PATTERN 1: DFS TRAVERSALS (INORDER)
# ============================================================================
def inorder_traversal(root):
    """
    LeetCode #94: Binary Tree Inorder Traversal
    Return the inorder traversal of its nodes' values: [Left, Root, Right].
    
    Example:
        Input: [1,null,2,3]
        Output: [1,3,2]
    
    APPROACH:
    - Recursively visit components in order: Left subtree -> Current Node -> Right subtree.
    
    WHY IT WORKS:
    - This order processes nodes "bottom-up" from the leftmost side first.
    - For BSTs, this yields sorted order.

    TIME COMPLEXITY: O(N)
    - We visit every node exactly once.

    SPACE COMPLEXITY: O(H)
    - H is the height of the tree (recursion stack).
    - Worst case (skewed tree) O(N). Balanced tree O(log N).
    """
    res = []
    def dfs(node):
        if not node: return
        dfs(node.left)
        res.append(node.val)
        dfs(node.right)
    dfs(root)
    return res

# ============================================================================
# PATTERN 2: BFS (LEVEL ORDER)
# ============================================================================
def level_order(root):
    """
    LeetCode #102: Binary Tree Level Order Traversal
    Return the level order traversal of its nodes' values. (i.e., from left to right, level by level).
    
    Example:
        Input: [3,9,20,null,null,15,7]
        Output: [[3],[9,20],[15,7]]
    
    APPROACH:
    - Use a Queue (FIFO).
    - Start with root in queue.
    - While queue is not empty:
      - Get current level size (k).
      - Pop k elements, add their children to queue.
      - Add popped values to current level list.

    WHY IT WORKS:
    - Queue naturally processes nodes in the order they were discovered.
    - Processing `level_size` nodes at a time ensures we group level siblings together.

    TIME COMPLEXITY: O(N)
    - Each node is enqueued and dequeued exactly once.

    SPACE COMPLEXITY: O(W)
    - W is the maximum width of the tree.
    - In a perfect binary tree, W = N/2 => O(N).
    """
    if not root: return []
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        current_level = []
        
        for _ in range(level_size):
            node = queue.popleft()
            current_level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
            
        result.append(current_level)
    return result

# ============================================================================
# PATTERN 3: VALIDATE BST
# ============================================================================
def is_valid_bst(root):
    """
    LeetCode #98: Validate Binary Search Tree
    Determine if it is a valid binary search tree (BST).
    
    APPROACH:
    - Each node must satisfy: `min_val < node.val < max_val`.
    - Recursively pass down the valid range.
    - Left child: Updates `max_val` to parent's value.
    - Right child: Updates `min_val` to parent's value.
    - Initial range: (-inf, +inf).

    WHY IT WORKS:
    - A local check (left < node < right) is insufficient; we must check against the *entire* ancestry constraints.
    - Passing the range ensures global validity.

    TIME COMPLEXITY: O(N)
    - Visit each node once.

    SPACE COMPLEXITY: O(H)
    - Recursion stack depth.
    """
    def validate(node, low, high):
        if not node: return True
        
        if not (low < node.val < high):
            return False
            
        return (validate(node.left, low, node.val) and 
                validate(node.right, node.val, high))
                
    return validate(root, float('-inf'), float('inf'))

# ============================================================================
# PATTERN 4: LOWEST COMMON ANCESTOR (LCA)
# ============================================================================
def lowest_common_ancestor(root, p, q):
    """
    LeetCode #236: Lowest Common Ancestor of a Binary Tree
    Find the lowest node that has both p and q as descendants.
    
    APPROACH:
    - Post-order DFS.
    - If current node is `p` or `q`, return current node (found one).
    - If current node is null, return null.
    - Recurse Left and Right.
    - Logic:
      1. If Left and Right both return non-null, current node is the split point (LCA).
      2. If only one returns non-null, propagate that result up (LCA is higher up or in that subtree).
    
    WHY IT WORKS:
    - We bubble up the discovery of p and q. The first node where these bubbles meet (left has one, right has one) is the LCA.

    TIME COMPLEXITY: O(N)
    - Worst case we visit every node.

    SPACE COMPLEXITY: O(H)
    - Recursion stack depth.
    """
    if not root or root == p or root == q:
        return root
        
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)
    
    if left and right:
        return root # Found one in left, one in right
    
    return left if left else right

# ============================================================================
# PATTERN 5: CONSTRUCT TREE (PREORDER + INORDER)
# ============================================================================
def build_tree(preorder, inorder):
    """
    LeetCode #105: Construct Binary Tree from Preorder and Inorder Traversal
    
    APPROACH:
    - Preorder: [ROOT, LEFT_SUBTREE, RIGHT_SUBTREE] -> First item is always root.
    - Inorder:  [LEFT_SUBTREE, ROOT, RIGHT_SUBTREE] -> Root splits the array.
    - Use a hash map to quickly find index of root in `inorder` array to compute subtree sizes.
    - Recurse.

    WHY IT WORKS:
    - Preorder tells us *what* the root is.
    - Inorder tells us the *structure* (size of left vs right subtrees).
    
    TIME COMPLEXITY: O(N)
    - Map building is O(N). Each node processed once.
    
    SPACE COMPLEXITY: O(N)
    - Hash map stores N entries. Recursion stack O(H).
    """
    inorder_map = {val: i for i, val in enumerate(inorder)}
    pre_iter = iter(preorder)
    
    def builder(left_idx, right_idx):
        if left_idx > right_idx:
            return None
            
        root_val = next(pre_iter)
        root = TreeNode(root_val)
        
        mid = inorder_map[root_val]
        
        # Build left then right
        root.left = builder(left_idx, mid - 1)
        root.right = builder(mid + 1, right_idx)
        
        return root
        
    return builder(0, len(inorder) - 1)

if __name__ == "__main__":
    print("Testing Trees:")
    print("-" * 30)
    root = create_tree([3,9,20,None,None,15,7])
    print("1. Inorder:", inorder_traversal(root))
    print("2. Level Order:", level_order(root))
    print("3. Valid BST:", is_valid_bst(root)) # False (tree is not BST)
    
    # Construction test
    pre = [3,9,20,15,7]
    ino = [9,3,15,20,7]
    new_root = build_tree(pre, ino)
    print("4. Constructed Tree Root:", new_root.val)
