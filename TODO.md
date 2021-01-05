#Analysis for function reuse.
* The function must have all arguments of primitive types.
* We need the worst latency of the function.
* The two instances of the function must not overlap.
* For each time region (instructions using same time variable), determine the
    min and max initiation interval.
* If two instructions are in same time region, they will have fixed delay.
* Check if delay+latency < min_II (for now) to see if the instructions are
    non-aliasing. Merge them in a group in that case.

#Algo.
1. Find Time regions - list of ops + pointer to the originating region:
   mapTV2ParentRegion<Value,region*>, mapTV2ops<Value,list<Op>>.
2. For each Time region, find min offset from the tstart of the region:
    mapTV2Offset<Value,unsigned>
3. Use 2. to find yield op in region and minimum delay to reach the yield op : 
    map<region, min_II>.
4. Use 1.0 and 3.0 to check if two ops may-conflict. Fuse them.

#TODO
0. Update numReads for memref parameters in hir.call.
1. check - each loop has a yield statement.
2. check - hir.call signature matches with hir.func.
3. Check - only the captured time variable is used inside for and if body.
4. Add support for rw in memrefs.
