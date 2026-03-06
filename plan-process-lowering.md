# Implementation Plan: Process Lowering in Arcilator

## Overview

This plan implements point 1 of the DESIGN.md implementation strategy: "Process lowering in Arcilator, with a manual test using `arc.sim.*` ops to see that the process changes output at the right points in time."

### Goals
- Lower `llhd.process` operations into the Arc dialect
- Allocate state for process execution (resume_time, resume_block, proc_result)
- Handle `llhd.wait` and `llhd.halt` terminators correctly
- Create manual tests using `arc.sim.*` operations to verify timing behavior

### Success Criteria
- `llhd.process` operations are successfully lowered to Arc IR as shown in DESIGN.md lines 68-141
- Process state is properly allocated and managed
- Tests demonstrate that process outputs change at the correct simulation times
- Validation rejects unsupported LLHD operations and block operands on `llhd.wait`

### Scope Boundaries
**Included:**
- Basic process lowering with time-based delays
- State allocation for process control flow
- Manual testing with `arc.sim.*` operations
- Validation of supported LLHD operations

**Excluded:**
- Signal-based wait conditions (wait on signal changes)
- Values live across waits (block operands on `llhd.wait`)
- Tracking earliest scheduled event time (point 2 of strategy)
- Automatic standalone runner generation (point 3 of strategy)
- Full LLHD operation support beyond process/wait/halt

## Prerequisites

### Required Dependencies
- Arc dialect operations: `arc.state`, `arc.state_read`, `arc.state_write`, `arc.current_time`
- LLHD operations: `llhd.process`, `llhd.wait`, `llhd.halt`, `llhd.time_to_int`
- Control flow: `scf.if`, `scf.execute_region`, `cf.switch`
- Comb operations: `comb.icmp`, `comb.add`

### Environment Requirements
- CIRCT build environment with Arc and LLHD dialects enabled
- Test infrastructure for Arc dialect tests

## Implementation Steps

### Step 1: Create LowerProcesses Pass Infrastructure

**Description:** Set up the basic pass structure for lowering LLHD processes in the Arc dialect.

**Files to modify:**
- `include/circt/Dialect/Arc/ArcPasses.td` - Add pass definition
- `include/circt/Dialect/Arc/ArcPasses.h` - Add pass creation function
- `lib/Dialect/Arc/Transforms/CMakeLists.txt` - Add new source file

**Files to create:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp` - Main pass implementation

**Key implementation details:**
- Pass should run on `arc::ModelOp` (after LowerState converts hw.module to arc.model)
- Use tablegen to define the pass in ArcPasses.td:
  ```tablegen
  def LowerProcesses : Pass<"arc-lower-processes", "arc::ModelOp"> {
    let summary = "Lower LLHD processes to Arc state and control flow";
    let dependentDialects = [
      "arc::ArcDialect",
      "comb::CombDialect",
      "mlir::scf::SCFDialect",
      "mlir::cf::ControlFlowDialect",
      "llhd::LLHDDialect"
    ];
  }
  ```
- Create pass skeleton with `runOnOperation()` method

**Testing considerations:**
- Add basic test to verify pass registration and execution

### Step 2: Implement Process Discovery and Validation

**Description:** Find all `llhd.process` operations in the model and validate they are supported.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
- Walk the model body to find all `llhd.process` operations
- For each process, validate:
  - No block operands on `llhd.wait` operations (reject with error)
  - Only supported LLHD ops: `llhd.process`, `llhd.wait`, `llhd.halt`, `llhd.time_to_int`, `llhd.constant_time`
  - All other operations in process body should be from supported dialects (comb, hw, arith, etc.)
- Emit clear error messages for unsupported patterns
- Return early if validation fails

**Testing considerations:**
- Test rejection of block operands on `llhd.wait`
- Test rejection of unsupported LLHD operations
- Test acceptance of valid process patterns

### Step 3: Allocate Process State Storage

**Description:** For each process, allocate state storage for resume_time, resume_block, and process results.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
- For each `llhd.process` operation:
  - Allocate `arc.alloc_state` for resume_time (i64)
  - Allocate `arc.alloc_state` for resume_block (i16 or appropriate integer type)
  - For each result type, allocate `arc.alloc_state` for proc_result
- Store allocated states in a data structure for later use
- Use the model's storage argument (`%arg0: !arc.storage`) as the storage operand
- Insert allocations at the beginning of the model body (before other operations)

**Testing considerations:**
- Verify correct number and types of state allocations
- Check that allocations use the correct storage argument

### Step 4: Analyze Process Control Flow

**Description:** Analyze the process body to identify all basic blocks and build a mapping for the cf.switch operation.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
- Traverse the process region to identify all basic blocks
- Assign a unique integer index to each block (0, 1, 2, ...)
- Create a mapping from block to index for use in lowering
- Identify the entry block (first block in the region)
- Count total blocks to determine if i16 is sufficient for resume_block state
- Add one extra index for the "halted" state

**Testing considerations:**
- Test processes with multiple blocks
- Verify correct block indexing

### Step 5: Lower llhd.wait Operations

**Description:** Transform `llhd.wait` operations into state writes and control flow.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
- For each `llhd.wait` operation:
  - Extract the delay operand (if present)
  - Convert delay from `!llhd.time` to `i64` using `llhd.time_to_int`
  - Read current_time using `arc.current_time`
  - Compute next_resume = current_time + delay using `comb.add`
  - Write next_resume to resume_time state using `arc.state_write`
  - Write target block index to resume_block state using `arc.state_write`
  - Write yield operands to proc_result states using `arc.state_write`
  - Replace with `scf.yield` to exit the execute_region
- Handle wait without delay (immediate resume) by writing current_time to resume_time

**Testing considerations:**
- Test wait with delay
- Test wait with yield operands
- Verify state writes occur in correct order

### Step 6: Lower llhd.halt Operations

**Description:** Transform `llhd.halt` operations into final state writes.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
- For each `llhd.halt` operation:
  - Write maximum i64 value (-1 as unsigned) to resume_time state
  - Write "halted" block index to resume_block state
  - Write yield operands to proc_result states
  - Replace with `scf.yield` to exit the execute_region

**Testing considerations:**
- Test halt with yield operands
- Verify resume_time is set to maximum value

### Step 7: Generate Process Control Flow Structure

**Description:** Create the main control flow structure with scf.if and cf.switch as shown in DESIGN.md.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
- Read current_time using `arc.current_time`
- Read resume_time from state using `arc.state_read`
- Create resume condition: `comb.icmp uge current_time, resume_time`
- Create `scf.if` with resume condition
- Inside scf.if, create `scf.execute_region`
- Inside execute_region, read resume_block from state
- Create `cf.switch` on resume_block with cases for each original block
- Add final "halted" case that just yields
- Clone original process blocks into the switch cases
- Replace process operation with the new control flow structure

**Testing considerations:**
- Verify scf.if structure is correct
- Verify cf.switch has correct number of cases
- Check that blocks are properly cloned

### Step 8: Initialize Process State

**Description:** Initialize process state on first execution (resume_block = 0, resume_time = 0).

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
- After allocating states, initialize them:
  - Write 0 to resume_time (process should run immediately)
  - Write 0 to resume_block (start at entry block)
  - Initialize proc_result states to default values (0 or undef)
- These initializations should happen in the model body before the control flow

**Testing considerations:**
- Verify initial state values are correct
- Test that process executes on first step

### Step 9: Handle Process Results

**Description:** Ensure process results are properly exposed and can be read.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
- After the scf.if, read proc_result states using `arc.state_read`
- Replace all uses of the original process results with the read values
- Ensure result types match the original process result types

**Testing considerations:**
- Test processes with multiple results
- Verify result values are correct at different times

### Step 10: Integrate into Arcilator Pipeline

**Description:** Add the new pass to the appropriate place in the Arcilator pipeline.

**Files to modify:**
- `lib/Tools/arcilator/pipelines.cpp`
- `include/circt/Tools/arcilator/pipelines.h`

**Key implementation details:**
- Add LowerProcesses pass to the state lowering pipeline
- It should run after LowerState (which converts hw.module to arc.model)
- It should run before state allocation (AllocateState)
- Update populateArcStateLoweringPipeline to include the new pass

**Testing considerations:**
- Test full pipeline with processes
- Verify pass ordering is correct

### Step 11: Create Manual Tests with arc.sim.* Operations

**Description:** Create comprehensive tests that use `arc.sim.*` operations to verify process timing behavior.

**Files to create:**
- `test/Dialect/Arc/lower-processes.mlir` - Unit tests for the pass
- `integration_test/arcilator/JIT/process-timing.mlir` - Integration test with JIT execution

**Key implementation details:**

**Unit test (test/Dialect/Arc/lower-processes.mlir):**
- Test basic process lowering transformation
- Verify state allocations are created
- Check control flow structure (scf.if, cf.switch)
- Test multiple processes in one module
- Test error cases (unsupported operations, block operands)

Example test structure:
```mlir
// RUN: circt-opt %s --arc-lower-state --arc-lower-processes | FileCheck %s

hw.module @Foo(out x : i42) {
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c0_i42 = hw.constant 0 : i42
  %c42_i42 = hw.constant 42 : i42

  %1 = llhd.process -> i42 {
    llhd.wait yield (%c0_i42 : i42), delay %time, ^bb1
  ^bb1:
    llhd.halt %c42_i42 : i42
  }
  hw.output %1 : i42
}

// CHECK-LABEL: arc.model @Foo
// CHECK: %[[RESUME_TIME:.+]] = arc.alloc_state %arg0 : (!arc.storage<>) -> !arc.state<i64>
// CHECK: %[[RESUME_BLOCK:.+]] = arc.alloc_state %arg0 : (!arc.storage<>) -> !arc.state<i16>
// CHECK: %[[PROC_RESULT:.+]] = arc.alloc_state %arg0 : (!arc.storage<>) -> !arc.state<i42>
// CHECK: %[[CURRENT_TIME:.+]] = arc.current_time %arg0
// CHECK: %[[RESUME_TIME_READ:.+]] = arc.state_read %[[RESUME_TIME]]
// CHECK: %[[RESUME:.+]] = comb.icmp uge %[[CURRENT_TIME]], %[[RESUME_TIME_READ]]
// CHECK: scf.if %[[RESUME]]
// CHECK:   scf.execute_region
// CHECK:     %[[RESUME_BLOCK_READ:.+]] = arc.state_read %[[RESUME_BLOCK]]
// CHECK:     cf.switch %[[RESUME_BLOCK_READ]] : i16, [
// CHECK:       0: ^[[BB0:.+]],
// CHECK:       1: ^[[BB1:.+]],
// CHECK:       2: ^[[HALTED:.+]]
// CHECK:     ]
// CHECK:   ^[[BB0]]:
// CHECK:     arc.state_write %[[RESUME_TIME]] =
// CHECK:     arc.state_write %[[RESUME_BLOCK]] =
// CHECK:     arc.state_write %[[PROC_RESULT]] = %c0_i42
// CHECK:     scf.yield
// CHECK:   ^[[BB1]]:
// CHECK:     arc.state_write %[[RESUME_TIME]] =
// CHECK:     arc.state_write %[[RESUME_BLOCK]] =
// CHECK:     arc.state_write %[[PROC_RESULT]] = %c42_i42
// CHECK:     scf.yield
// CHECK:   ^[[HALTED]]:
// CHECK:     scf.yield
// CHECK: %[[RESULT:.+]] = arc.state_read %[[PROC_RESULT]]
```

**Integration test (integration_test/arcilator/JIT/process-timing.mlir):**
- Test the example from DESIGN.md (lines 3-29)
- Use `arc.sim.*` operations to:
  - Instantiate the module
  - Step through simulation time
  - Read output at different times
  - Verify output values match expected timing
- Print results to verify correct behavior

Example test structure:
```mlir
// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

hw.module @Foo(out x : i42) {
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c0_i42 = hw.constant 0 : i42
  %c42_i42 = hw.constant 42 : i42
  %c1337_i42 = hw.constant 1337 : i42
  %c9001_i42 = hw.constant 9001 : i42

  %1 = llhd.process -> i42 {
    llhd.wait yield (%c0_i42 : i42), delay %time, ^bb1
  ^bb1:
    llhd.wait yield (%c42_i42 : i42), delay %time, ^bb2
  ^bb2:
    llhd.wait yield (%c1337_i42 : i42), delay %time, ^bb3
  ^bb3:
    llhd.wait yield (%c9001_i42 : i42), delay %time, ^bb4
  ^bb4:
    llhd.halt %c0_i42 : i42
  }
  hw.output %1 : i42
}

func.func @main() {
  // CHECK: x = {{0*}}0
  // CHECK: x = {{0*}}2a
  // CHECK: x = {{0*}}539
  // CHECK: x = {{0*}}2329
  // CHECK: x = {{0*}}0

  arc.sim.instantiate @Foo as %model {
    // Time 0ns
    arc.sim.step %model : !arc.sim.instance<@Foo>
    %x0 = arc.sim.get_port %model, "x" : i42, !arc.sim.instance<@Foo>
    arc.sim.emit "x", %x0 : i42

    // Advance to 1ns
    %time1 = arith.constant 1000000000 : i64  // 1ns in femtoseconds
    arc.sim.set_time %model, %time1 : !arc.sim.instance<@Foo>
    arc.sim.step %model : !arc.sim.instance<@Foo>
    %x1 = arc.sim.get_port %model, "x" : i42, !arc.sim.instance<@Foo>
    arc.sim.emit "x", %x1 : i42

    // Advance to 2ns
    %time2 = arith.constant 2000000000 : i64
    arc.sim.set_time %model, %time2 : !arc.sim.instance<@Foo>
    arc.sim.step %model : !arc.sim.instance<@Foo>
    %x2 = arc.sim.get_port %model, "x" : i42, !arc.sim.instance<@Foo>
    arc.sim.emit "x", %x2 : i42

    // Advance to 3ns
    %time3 = arith.constant 3000000000 : i64
    arc.sim.set_time %model, %time3 : !arc.sim.instance<@Foo>
    arc.sim.step %model : !arc.sim.instance<@Foo>
    %x3 = arc.sim.get_port %model, "x" : i42, !arc.sim.instance<@Foo>
    arc.sim.emit "x", %x3 : i42

    // Advance to 4ns
    %time4 = arith.constant 4000000000 : i64
    arc.sim.set_time %model, %time4 : !arc.sim.instance<@Foo>
    arc.sim.step %model : !arc.sim.instance<@Foo>
    %x4 = arc.sim.get_port %model, "x" : i42, !arc.sim.instance<@Foo>
    arc.sim.emit "x", %x4 : i42
  }

  return
}
```

**Testing considerations:**
- Verify output values at each time step
- Test that process doesn't execute when resume_time hasn't been reached
- Test halt behavior (no further changes after halt)

## File Changes Summary

### Files to Create
1. **lib/Dialect/Arc/Transforms/LowerProcesses.cpp** - Main pass implementation (~500-800 lines)
2. **test/Dialect/Arc/lower-processes.mlir** - Unit tests for pass (~200-400 lines)
3. **integration_test/arcilator/JIT/process-timing.mlir** - Integration test (~100-150 lines)

### Files to Modify
1. **include/circt/Dialect/Arc/ArcPasses.td**
   - Add LowerProcesses pass definition (~15 lines)

2. **include/circt/Dialect/Arc/ArcPasses.h**
   - Add createLowerProcessesPass() declaration (~1 line)

3. **lib/Dialect/Arc/Transforms/CMakeLists.txt**
   - Add LowerProcesses.cpp to sources (~1 line)

4. **lib/Tools/arcilator/pipelines.cpp**
   - Add pass to populateArcStateLoweringPipeline (~1 line)

5. **lib/Dialect/Arc/Transforms/LowerState.cpp** (optional)
   - May need to ensure llhd.process operations are not lowered by LowerState
   - Add check to skip llhd.process operations (~5-10 lines)

## Testing Strategy

### Unit Tests
- **Basic lowering**: Test simple process with one wait and one halt
- **Multiple blocks**: Test process with multiple wait operations
- **Multiple results**: Test process with multiple result values
- **No delay**: Test wait without delay operand
- **Error cases**:
  - Block operands on llhd.wait (should reject)
  - Unsupported LLHD operations (should reject)
  - Invalid process structure

### Integration Tests
- **Timing verification**: Test the DESIGN.md example with manual time stepping
- **Multiple processes**: Test module with multiple concurrent processes
- **Complex computations**: Test process with non-trivial operations in blocks

### Manual Testing Steps
1. Build CIRCT with the new pass
2. Run unit tests: `ninja -C build check-circt`
3. Run integration tests: `ninja -C build check-circt-integration`
4. Test with example from DESIGN.md
5. Verify output values at each time step

## Rollback Plan

If issues are discovered:
1. Remove the pass from the pipeline in `pipelines.cpp`
2. The pass can be disabled without affecting other functionality
3. Tests can be marked as XFAIL temporarily
4. No data migration needed as this is new functionality

## Estimated Effort

- **Implementation**: 2-3 days
  - Pass infrastructure: 0.5 day
  - Core lowering logic: 1-1.5 days
  - Integration and debugging: 0.5-1 day

- **Testing**: 1-1.5 days
  - Unit tests: 0.5 day
  - Integration tests: 0.5 day
  - Manual verification: 0.5 day

- **Total**: 3-4.5 days

**Complexity**: Medium-High
- Requires understanding of Arc dialect state management
- Control flow transformation is moderately complex
- Integration with existing pipeline requires care
- Testing timing behavior requires careful test design

## Implementation Notes

### Key Design Decisions

1. **Pass Placement**: Run after LowerState but before AllocateState
   - LowerState converts hw.module to arc.model, providing the storage argument
   - AllocateState assigns offsets to all arc.alloc_state operations
   - This placement allows process state to be allocated alongside other state

2. **Resume Block Type**: Use i16 for resume_block state
   - Supports up to 65,536 blocks per process (more than sufficient)
   - Could be optimized to smaller type based on actual block count

3. **Time Representation**: Use i64 for time in femtoseconds
   - Consistent with arc.current_time operation
   - llhd.time_to_int converts LLHD time to i64 femtoseconds
   - Maximum time (-1 as unsigned i64) represents "never resume"

4. **Control Flow Structure**: Use scf.if + scf.execute_region + cf.switch
   - scf.if checks if resume time has been reached
   - scf.execute_region provides a region for the cf.switch
   - cf.switch dispatches to the correct resumption block

5. **State Initialization**: Initialize in model body, not in initializer function
   - Simpler implementation for first version
   - Can be moved to initializer function in future optimization

### Code References

**Similar passes to reference:**
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Shows how to lower hw.module to arc.model
  - ModuleLowering class structure (lines 63-186)
  - State allocation pattern (lines 215-222)
  - Operation lowering with phases (lines 243-261)

- `lib/Dialect/LLHD/Transforms/LowerProcesses.cpp` - Shows LLHD process lowering to combinational
  - Process validation and transformation
  - Wait operation handling

- `lib/Conversion/ConvertToArcs/ConvertToArcs.cpp` - Shows arc creation patterns
  - Block cloning and transformation
  - State operation creation

**Arc operations to use:**
- `arc.alloc_state` - Allocate state storage (include/circt/Dialect/Arc/ArcOps.td:501-508)
- `arc.state_read` - Read state value (include/circt/Dialect/Arc/ArcOps.td:575-585)
- `arc.state_write` - Write state value (include/circt/Dialect/Arc/ArcOps.td:587-612)
- `arc.current_time` - Get current simulation time (include/circt/Dialect/Arc/ArcOps.td:614-627)

**LLHD operations to handle:**
- `llhd.process` - Process definition (include/circt/Dialect/LLHD/LLHDOps.td:514-550)
- `llhd.wait` - Suspend execution (include/circt/Dialect/LLHD/LLHDOps.td:629-664)
- `llhd.halt` - Terminate execution (include/circt/Dialect/LLHD/LLHDOps.td:666-690)
- `llhd.time_to_int` - Convert time to i64 (include/circt/Dialect/LLHD/LLHDOps.td:111-120)

**Test patterns to reference:**
- `test/Dialect/Arc/lower-state.mlir` - Shows FileCheck patterns for arc.model
- `integration_test/arcilator/JIT/basic.mlir` - Shows arc.sim.* usage pattern
- `test/Dialect/LLHD/Transforms/lower-processes.mlir` - Shows LLHD process test patterns

### Potential Issues and Solutions

**Issue 1: Block cloning with SSA values**
- **Problem**: When cloning process blocks into cf.switch cases, SSA values need to be remapped
- **Solution**: Use IRMapping to track value mappings during cloning, similar to LowerState.cpp:887-900

**Issue 2: Multiple processes in one module**
- **Problem**: Each process needs unique state allocations
- **Solution**: Process each llhd.process independently, creating separate state for each

**Issue 3: Process results used before first execution**
- **Problem**: Process results are undefined until first execution
- **Solution**: Initialize proc_result states to zero or appropriate default values

**Issue 4: Time conversion**
- **Problem**: LLHD time has delta and epsilon components, not just real time
- **Solution**: For initial implementation, only support real time delays (reject delta/epsilon)
  - llhd.time_to_int already handles conversion to femtoseconds
  - Can add delta/epsilon support in future work

**Issue 5: Integration with existing LowerState pass**
- **Problem**: LowerState might try to lower llhd.process operations
- **Solution**: Add check in LowerState to skip llhd.process operations, leaving them for LowerProcesses

### Future Enhancements (Out of Scope)

1. **Signal-based wait conditions**: Support `llhd.wait (%signal : i1), ^bb1`
   - Requires tracking signal changes
   - More complex resume condition

2. **Values live across waits**: Support block operands on llhd.wait
   - Requires allocating state for live values
   - More complex state management

3. **Earliest event time tracking**: Implement point 2 of DESIGN.md strategy
   - Track minimum resume_time across all processes
   - Expose as model state

4. **Automatic runner generation**: Implement point 3 of DESIGN.md strategy
   - Generate standalone main function
   - Automatic time advancement loop

5. **Optimization**: Merge process state allocations
   - Reuse storage for processes that don't overlap
   - Reduce total state size

6. **Delta and epsilon time**: Full LLHD time support
   - Handle delta cycles
   - Handle epsilon (scheduling regions)

## References

### CIRCT Documentation
- Arc Dialect: `docs/Dialects/Arc.md`
- LLHD Dialect: LLHD operations and semantics
- Arcilator: `tools/arcilator/arcilator.cpp` - Pipeline structure

### Key Files
- **DESIGN.md**: Lines 1-173 - Full design document
- **DESIGN.md**: Lines 68-141 - Target lowering example
- **DESIGN.md**: Lines 166-173 - Implementation strategy

### Build and Test Commands
```bash
# Build CIRCT
ninja -C build

# Run all tests
ninja -C build check-circt

# Run Arc dialect tests only
ninja -C build check-circt-arc

# Run integration tests
ninja -C build check-circt-integration

# Run specific test file
build/bin/circt-opt test/Dialect/Arc/lower-processes.mlir --arc-lower-processes | FileCheck test/Dialect/Arc/lower-processes.mlir

# Run JIT test
build/bin/arcilator integration_test/arcilator/JIT/process-timing.mlir --run --jit-entry=main
```

### Formatting
```bash
# Format C++ files after implementation
clang-format -i lib/Dialect/Arc/Transforms/LowerProcesses.cpp
```

## Summary

This plan provides a complete roadmap for implementing basic LLHD process lowering in Arcilator. The implementation will:

1. Create a new `LowerProcesses` pass that runs in the Arc state lowering pipeline
2. Transform `llhd.process` operations into Arc state and control flow
3. Allocate state for process execution (resume_time, resume_block, proc_result)
4. Handle time-based delays using `llhd.wait` and `llhd.halt`
5. Provide comprehensive testing with both unit and integration tests
6. Enable manual testing with `arc.sim.*` operations to verify timing behavior

The implementation follows the design in DESIGN.md lines 68-141 and sets the foundation for future enhancements including earliest event time tracking and automatic runner generation.


