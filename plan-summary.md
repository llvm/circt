# Process Lowering Implementation Plan - Quick Summary

## What This Implements

Point 1 of DESIGN.md implementation strategy: Process lowering in Arcilator with manual testing using `arc.sim.*` operations.

## Core Transformation

Transform this LLHD process:
```mlir
%1 = llhd.process -> i42 {
  llhd.wait yield (%c0_i42 : i42), delay %time, ^bb1
^bb1:
  llhd.halt %c42_i42 : i42
}
```

Into this Arc IR:
```mlir
%resume_time = arc.alloc_state %storage : i64
%resume_block = arc.alloc_state %storage : i16
%proc_result = arc.alloc_state %storage : i42

%current_time = arc.current_time %storage
%resume_time_read = arc.state_read %resume_time
%resume = comb.icmp uge %current_time, %resume_time_read

scf.if %resume {
  scf.execute_region {
    %resume_block_read = arc.state_read %resume_block
    cf.switch %resume_block_read : i16, [0: ^bb0, 1: ^bb1, 2: ^halted]
    ^bb0:
      // llhd.wait logic
      arc.state_write %resume_time = <next_time>
      arc.state_write %resume_block = %c1_i16
      arc.state_write %proc_result = %c0_i42
      scf.yield
    ^bb1:
      // llhd.halt logic
      arc.state_write %resume_time = %c-1_i64
      arc.state_write %resume_block = %c2_i16
      arc.state_write %proc_result = %c42_i42
      scf.yield
    ^halted:
      scf.yield
  }
}
%result = arc.state_read %proc_result
```

## Key Components

1. **New Pass**: `LowerProcesses` - Runs after LowerState, before AllocateState
2. **State Allocation**: 3 states per process (resume_time, resume_block, proc_result)
3. **Control Flow**: scf.if + scf.execute_region + cf.switch pattern
4. **Time Management**: Use arc.current_time and i64 femtoseconds

## Files to Create

1. `lib/Dialect/Arc/Transforms/LowerProcesses.cpp` (~500-800 lines)
2. `test/Dialect/Arc/lower-processes.mlir` (~200-400 lines)
3. `integration_test/arcilator/JIT/process-timing.mlir` (~100-150 lines)

## Files to Modify

1. `include/circt/Dialect/Arc/ArcPasses.td` - Add pass definition
2. `include/circt/Dialect/Arc/ArcPasses.h` - Add pass declaration
3. `lib/Dialect/Arc/Transforms/CMakeLists.txt` - Add source file
4. `lib/Tools/arcilator/pipelines.cpp` - Add to pipeline
5. `lib/Dialect/Arc/Transforms/LowerState.cpp` - Skip llhd.process (optional)

## Implementation Steps (11 total)

1. Create pass infrastructure
2. Implement process discovery and validation
3. Allocate process state storage
4. Analyze process control flow
5. Lower llhd.wait operations
6. Lower llhd.halt operations
7. Generate process control flow structure
8. Initialize process state
9. Handle process results
10. Integrate into Arcilator pipeline
11. Create manual tests with arc.sim.* operations

## Testing Strategy

**Unit Tests**: Verify transformation correctness
- Basic lowering
- Multiple blocks
- Error cases (unsupported ops, block operands)

**Integration Tests**: Verify timing behavior
- Manual time stepping with arc.sim.set_time
- Verify output values at each time step
- Test example from DESIGN.md

## Estimated Effort

- Implementation: 2-3 days
- Testing: 1-1.5 days
- **Total: 3-4.5 days**
- **Complexity: Medium-High**

## Success Criteria

✅ llhd.process operations lower to Arc IR as specified in DESIGN.md
✅ Process state is properly allocated and managed
✅ Tests demonstrate correct timing behavior
✅ Validation rejects unsupported patterns

## Out of Scope (Future Work)

- Signal-based wait conditions
- Values live across waits (block operands)
- Earliest event time tracking (point 2)
- Automatic runner generation (point 3)
- Delta and epsilon time support

## Next Steps After This Implementation

1. Point 2: Track earliest scheduled event time across all processes
2. Point 3: Generate standalone runner function for modules
3. Optimize: Merge process state allocations, reduce overhead
4. Extend: Support signal-based waits and live values

---

**See `plan-process-lowering.md` for complete detailed implementation plan.**

