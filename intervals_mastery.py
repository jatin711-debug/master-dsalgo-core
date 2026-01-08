# ============================================================================
# MODULE 1.5: INTERVALS
# ============================================================================
"""
The "Intervals" pattern deals with ranges of time or numbers (e.g., [start, end]).
These problems almost always require SORTING as a first step.

COMMON PATTERNS:
1. Merge Intervals: Sort by start time. Iterate and merge if current.start <= prev.end.
2. Non-overlapping Intervals: Sort by END time (Greedy). Pick first ending to maximize capacity.
3. Insert Interval: Handle "before", "overlapping", and "after" parts relative to the new interval.

TIME COMPLEXITY: Usually O(N log N) due to sorting.
SPACE COMPLEXITY: O(1) or O(N) depending on output requirements.
"""

# ============================================================================
# PATTERN 1: MERGE INTERVALS
# ============================================================================
"""
CONCEPT: Merge all overlapping intervals.
STRATEGY:
1. Sort by start time.
2. Initialize 'merged' list with the first interval.
3. Iterate through remaining intervals:
   - If current overlaps with last merged (current.start <= last_merged.end):
     Merge them: last_merged.end = max(last_merged.end, current.end)
   - Else:
     Add current to 'merged'.
"""

def merge(intervals):
    """
    LeetCode #56: Merge Intervals
    Given an array of intervals where intervals[i] = [start, end], merge all overlapping intervals.

    Example:
        Input: intervals = [[1,3],[2,6],[8,10],[15,18]]
        Output: [[1,6],[8,10],[15,18]]
    
    APPROACH:
    - Sort intervals by start time to ensure we process them in order.
    - Keep a 'merged' list. Initialize with the first interval.
    - For each subsequent interval, check if it overlaps with the last added interval in 'merged'.
    - Overlap condition: `current_start <= last_end`.
    - If overlap: Update `last_end` to `max(last_end, current_end)`.
    - If no overlap: Append current interval to 'merged'.

    WHY IT WORKS:
    - Sorting puts potential overlaps adjacent to each other.
    - By carrying forward the "extended" end time, we daisy-chain merges (e.g., [1,3], [2,6], [5,10] all merge).
    - We only need to compare with the *last* merged interval because the list is sorted.
    
    TIME COMPLEXITY: O(N log N)
    - Sorting takes O(N log N).
    - The linear scan takes O(N).
    - Total: O(N log N).

    SPACE COMPLEXITY: O(N)
    - O(N) to store the `merged` list in the worst case (no merges).
    - Sorting might take O(log N) stack space depending on implementation (Timsort).
    """
    if not intervals:
        return []
    
    # 1. Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for i in range(1, len(intervals)):
        current = intervals[i]
        last_merged = merged[-1]
        
        # 2. Check overlap
        if current[0] <= last_merged[1]:
            # Merge: Use the furthest end time
            last_merged[1] = max(last_merged[1], current[1])
        else:
            # No overlap, add to result
            merged.append(current)
            
    return merged

# ============================================================================
# PATTERN 2: NON-OVERLAPPING INTERVALS (GREEDY)
# ============================================================================
"""
CONCEPT: Find min intervals to remove to make rest non-overlapping.
EQUIVALENT TO: Find MAX number of non-overlapping intervals (Activity Selection Problem).
STRATEGY:
1. Sort by END time. (Greedy choice: finish earliest to leave room for others).
2. Track 'end' of last valid interval.
3. If next interval starts before last ended -> Overlap! Increment remove count.
4. Else -> No overlap. Update 'end'.
"""

def erase_overlap_intervals(intervals):
    """
    LeetCode #435: Non-overlapping Intervals
    Given an array of intervals, return the minimum number of intervals to remove 
    to make the rest of the intervals non-overlapping.
    
    Example:
        Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
        Output: 1 (Remove [1,3])

    APPROACH:
    - Greedy Algorithm (Activity Selection).
    - Sort by END time. Why? Because the interval that ends easiest leaves the most room for future intervals.
    - Iterate and keep track of the `end` of the last selected non-overlapping interval.
    - If `current_start < last_end`: Intersection found. We effectively "remove" the current interval (increment count) because keeping the previous one (which ends earlier) is optimal.
    - Else: Update `last_end` to `current_end`.

    WHY IT WORKS:
    - By always picking the interval that finishes earliest, we maximize the remaining time line for other intervals.
    - Example: [1,4], [2,3], [3,5]. Sorting by end gives [2,3], [1,4], [3,5].
      - Keep [2,3].
      - [1,4] overlaps (starts at 1 < 3). Remove it.
      - [3,5] valid (starts at 3 >= 3). Keep it.
    
    TIME COMPLEXITY: O(N log N)
    - Sorting takes O(N log N).
    - Iteration takes O(N).

    SPACE COMPLEXITY: O(1)
    - If we ignore sorting stack space, we only use integer variables.
    """
    if not intervals:
        return 0
        
    # Sort by END time
    intervals.sort(key=lambda x: x[1])
    
    end = intervals[0][1]
    count = 0 # Intervals to remove
    
    for i in range(1, len(intervals)):
        if intervals[i][0] < end:
            # Overlap found. We effectively "remove" the current one because
            # keeping the previous one (which ends earlier) is always better optimized.
            count += 1
        else:
            # No overlap, update end
            end = intervals[i][1]
            
    return count

# ============================================================================
# PATTERN 3: INSERT INTERVAL
# ============================================================================
"""
CONCEPT: Insert new interval into sorted non-overlapping list and merge if needed.
STRATEGY:
Three phases loop:
1. Add all intervals ending BEFORE new_interval starts.
2. Merge all intervals that OVERLAP with new_interval.
   (new_start = min, new_end = max)
3. Add all intervals starting AFTER new_interval ends.
"""

def insert(intervals, new_interval):
    """
    LeetCode #57: Insert Interval
    
    Example:
        Input: intervals = [[1,3],[6,9]], newInterval = [2,5]
        Output: [[1,5],[6,9]]

    APPROACH:
    - Since the list is already sorted, we can process in one pass O(N).
    - Phase 1: Skip and add all intervals that come strictly *before* the new interval (`current.end < new.start`).
    - Phase 2: Handle overlaps. While `current.start <= new.end`, we are overlapping.
      - Merge them into `new_interval`: `new.start = min`, `new.end = max`.
    - Phase 3: Add the finalized `new_interval`, then add all remaining intervals.

    WHY IT WORKS:
    - The "before" and "after" parts are untouched.
    - The "overlap" part effectively collapses multiple intervals into one big interval.
    - Maintains sorted property.

    TIME COMPLEXITY: O(N)
    - We iterate through the list exactly once.

    SPACE COMPLEXITY: O(N)
    - Allocate a new list for the result (standard for immutable input or when required).
    """
    result = []
    i = 0
    n = len(intervals)
    
    # 1. Add intervals before
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
        
    # 2. Merge overlapping intervals
    # While overlap exists: start <= new_end
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)
    
    # 3. Add intervals after
    while i < n:
        result.append(intervals[i])
        i += 1
        
    return result

if __name__ == "__main__":
    print("Testing Intervals Pattern:")
    print("-" * 30)
    print("1. Merge Intervals ([[1,3],[2,6],[8,10],[15,18]]):", merge([[1,3],[2,6],[8,10],[15,18]]))
    print("2. Non-overlapping Removal ([[1,2],[2,3],[3,4],[1,3]]):", erase_overlap_intervals([[1,2],[2,3],[3,4],[1,3]]))
    print("3. Insert Interval ([[1,3],[6,9]], [2,5]):", insert([[1,3],[6,9]], [2,5]))
