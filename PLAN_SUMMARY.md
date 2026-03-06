# Process Lowering Implementation Plan - Summary

## Quick Reference

**Main Plan Document:** `plan-process-lowering-optimized.md`

**Objective:** Implement process lowering in Arcilator to support basic LLHD processes with time-based delays.

## Key Innovation: Optimized Block Indexing

Unlike the original DESIGN.md which assigns an index to every block, this implementation uses an **optimized indexing scheme**:

- **Index 0:** Entry block (always)
- **Indices 1, 2, 3, ...:** Only blocks that are TARGETS of `llhd.wait` terminators
- **Final index N+1:** Halted state
- **No index:** Fall-through blocks (execute naturally without switch case)

### Example

Given a process with blocks: `^bb0`, `^bb1`, `^bb2`, `^bb3`, `^bb4`

**Original scheme (every block gets index):**
```
^bb0 (entry) → index 0
^bb1 → index 1
^bb2 → index 2
^bb3 → index 3
^bb4 → index 4
halted → index 5
```
Switch has 6 cases.

**Optimized scheme (only wait targets get index):**
```
^bb0 (entry) → index 0
^bb1 (wait target) → index 1
^bb2 (fall-through, no wait targets it) → NO INDEX
^bb3 (wait target) → index 2
^bb4 (wait target) → index 3
halted → index 4
```
Switch has 5 cases. `^bb2` executes naturally via `cf.br`.

## Implementation Phases

### Phase 1: Core Infrastructure (Steps 1-2)
- Create LowerProcesses pass
- Implement optimized block analysis and indexing

### Phase 2: State Management (Steps 3-4)
- Allocate process state (resume_time, resume_block, proc_result)
- Generate process entry logic with time checking

### Phase 3: Control Flow (Steps 5-7)
- Generate cf.switch with optimized cases
- Lower llhd.wait terminators
- Lower llhd.halt terminators

### Phase 4: Integration (Steps 8-11)
- Initialize process state
- Complete end-to-end lowering
- Add validation/rejection logic
- Integrate into Arcilator pipeline

### Phase 5: Testing (Step 12)
- Transformation tests
- JIT execution tests with timing verification
- Edge case tests

## Critical Files

### To Create
1. `lib/Dialect/Arc/Transforms/LowerProcesses.cpp` - Main implementation
2. `test/Dialect/Arc/lower-processes.mlir` - Transformation tests
3. `integration_test/arcilator/JIT/process-basic.mlir` - JIT tests

### To Modify
1. `lib/Dialect/Arc/Transforms/CMakeLists.txt` - Build configuration
2. `include/circt/Dialect/Arc/ArcPasses.td` - Pass definition
3. `lib/Tools/arcilator/pipelines.cpp` - Pipeline integration

## State Representation

Each process requires three state variables:

```cpp
resume_time: i64    // When to resume (femtoseconds), -1 = never
resume_block: i16   // Which block to resume at (index)
proc_result: T      // Current output value (type T = process result type)
```

## Testing Strategy

### Transformation Test
Verify that `llhd.process` is correctly lowered to Arc operations with proper state management and control flow.

### JIT Execution Test
Run actual simulation and verify:
- Time 0ns: output = 0
- Time 1ns: output = 42
- Time 2ns: output = 1337
- Time 3ns: output = 9001
- Time 4ns: output = 0 (halted)

## Estimated Effort

**Total:** 36-50 hours (1-1.5 weeks)
**Complexity:** Medium-High

## Success Metrics

✅ Pass compiles and registers correctly  
✅ Transformation tests pass  
✅ JIT tests show correct timing behavior  
✅ Process outputs change at expected times  
✅ Optimized indexing reduces switch cases  
✅ Fall-through blocks work correctly  

## Next Steps After This PR

1. Track earliest scheduled event time across all processes
2. Generate standalone runner functions
3. Support block operands on llhd.wait
4. Support signal observation in wait conditions
5. Broader LLHD operation support

## Questions or Issues?

Refer to the detailed plan in `plan-process-lowering-optimized.md` for:
- Detailed implementation steps
- Code examples and patterns
- Testing strategies
- Edge cases and error handling
- References to existing code

