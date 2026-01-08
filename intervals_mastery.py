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
    
    Time: O(N log N) - Sorting dominates
    Space: O(N) - To store output
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
    given an array of intervals intervals where intervals[i] = [starti, endi], return 
    the minimum number of intervals you need to remove to make the rest of the intervals non-overlapping.
    
    Example:
        Input: intervals = [[1,2],[2,3],[3,4],[1,3]]
        Output: 1 (Remove [1,3])

    Time: O(N log N)
    Space: O(1) (if we ignore sorting stack space)
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

    Time: O(N) - Single pass (list is already sorted)
    Space: O(N) - Output list
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
