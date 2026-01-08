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
    Left -> Root -> Right
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
    
    Approach: Queue.
    - Process size of queue at start of level to distinguish levels.
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
    
    Approach: Pass range (min, max) down via recursion.
    Root value must be within (min, max).
    Left child must be (min, root.val).
    Right child must be (root.val, max).
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
    
    Approach:
    - If root is p or q, root is LCA.
    - Search left and right.
    - If both return non-null, root is LCA (p and q are in different subtrees).
    - If only one returns non-null, that one is the answer (p and q in same subtree).
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
    
    Approach:
    - Preorder first element is ROOT.
    - Find Root in Inorder array (split into Left and Right subtrees).
    - Recursively build keys.
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
