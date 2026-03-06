# Implementation Plan: Process Lowering in Arcilator with Optimized Block Indexing

## Overview

This plan implements point 1 of the DESIGN.md implementation strategy: "Process lowering in Arcilator, with a manual test using `arc.sim.*` ops to see that the process changes output at the right points in time."

**Key Innovation**: The block indexing scheme is optimized to only assign indices to:
- Index 0: Entry block of the process
- Indices 1, 2, 3, ...: Only blocks that are TARGETS of `llhd.wait` terminators
- Final index (N+1): Reserved for the "halted" state (when `llhd.halt` is reached)

Blocks that are NOT targeted by wait terminators do NOT get an index and will fall through naturally in the control flow without needing a `cf.switch` case.

### Goals
- Lower `llhd.process` operations to Arc dialect operations
- Allocate state for process resume tracking (resume_time, resume_block, proc_result)
- Handle `llhd.wait` and `llhd.halt` terminators with optimized indexing
- Create manual tests using `arc.sim.*` operations to verify timing behavior

### Success Criteria
- `llhd.process` operations are successfully lowered to `arc.model` with state management
- Process outputs change at the correct simulation times
- Tests pass showing correct timing behavior

### Scope Boundaries
**Included:**
- Basic process lowering with time-based delays
- State allocation for process control flow
- Optimized block indexing (only wait targets get indices)
- Manual testing with `arc.sim.*` operations

**Excluded (for later PRs):**
- Tracking earliest scheduled event time across all processes
- Automatic standalone runner function generation
- Block operands on `llhd.wait` (will be rejected)
- Complex LLHD operations beyond process/wait/halt
- Signal observation in wait conditions

## Prerequisites

### Required Dependencies
- Arc dialect operations: `arc.model`, `arc.state`, `arc.alloc_state`, `arc.state_read`, `arc.state_write`, `arc.current_time`
- LLHD dialect operations: `llhd.process`, `llhd.wait`, `llhd.halt`, `llhd.constant_time`, `llhd.time_to_int`
- Control flow operations: `cf.switch`, `scf.if`, `scf.execute_region`, `scf.yield`
- Comb operations: `comb.icmp`, `comb.add`

### Environment Requirements
- CIRCT build environment with Arc and LLHD dialects enabled
- Test infrastructure for `arc.sim.*` operations

## Implementation Steps

### Step 1: Create LowerProcesses Pass Infrastructure

**Description:** Set up the basic pass infrastructure for lowering LLHD processes.

**Files to create:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Files to modify:**
- `lib/Dialect/Arc/Transforms/CMakeLists.txt` (add new source file)
- `include/circt/Dialect/Arc/ArcPasses.td` (add pass definition)
- `include/circt/Dialect/Arc/ArcPasses.h.inc` (generated)

**Key implementation details:**
```cpp
// In ArcPasses.td
def LowerProcessesPass : Pass<"arc-lower-processes", "arc::ModelOp"> {
  let summary = "Lower LLHD processes to Arc state management";
  let dependentDialects = [
    "arc::ArcDialect",
    "llhd::LLHDDialect",
    "comb::CombDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect",
  ];
}

// In LowerProcesses.cpp
struct LowerProcessesPass : public arc::impl::LowerProcessesPassBase<LowerProcessesPass> {
  void runOnOperation() override;
};
```

**Testing considerations:**
- Verify pass registration
- Test that pass runs without crashing on empty models

### Step 2: Implement Block Analysis with Optimized Indexing

**Description:** Analyze the process CFG to determine which blocks need indices (only wait targets and entry).

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
```cpp
struct ProcessBlockInfo {
  Block *block;
  std::optional<unsigned> index;  // Only set for entry, wait targets, and halted
  bool isWaitTarget = false;
  bool isEntry = false;
  bool isHalted = false;
};

class ProcessAnalyzer {
  // Analyze process and assign indices only to:
  // - Entry block (index 0)
  // - Blocks targeted by llhd.wait (indices 1, 2, 3, ...)
  // - Halted state (final index N+1)
  DenseMap<Block*, ProcessBlockInfo> analyzeProcess(llhd::ProcessOp processOp);
  
  unsigned assignBlockIndices(llhd::ProcessOp processOp);
};
```

**Algorithm:**
1. Identify entry block → assign index 0
2. Walk all blocks, find `llhd.wait` terminators
3. For each wait terminator, mark its target block as needing an index
4. Assign sequential indices (1, 2, 3, ...) to wait target blocks
5. Reserve final index (N+1) for halted state
6. Blocks without indices will fall through naturally

**Testing considerations:**
- Test with processes having various CFG structures
- Verify only necessary blocks get indices
- Test processes with fall-through blocks

### Step 3: Implement State Allocation for Process Control

**Description:** Allocate state storage for process control flow (resume_time, resume_block, proc_result).

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
```cpp
struct ProcessState {
  Value resumeTime;    // arc.alloc_state : !arc.state<i64>
  Value resumeBlock;   // arc.alloc_state : !arc.state<i16>
  Value procResult;    // arc.alloc_state : !arc.state<resultType>
};

ProcessState allocateProcessState(OpBuilder &allocBuilder,
                                  Location loc,
                                  Value storageArg,
                                  llhd::ProcessOp processOp) {
  // Allocate resume_time (i64 for femtoseconds)
  auto resumeTimeState = allocBuilder.create<arc::AllocStateOp>(
      loc, StateType::get(allocBuilder.getI64Type()), storageArg);

  // Allocate resume_block (i16 for block index)
  auto resumeBlockState = allocBuilder.create<arc::AllocStateOp>(
      loc, StateType::get(allocBuilder.getI16Type()), storageArg);

  // Allocate proc_result (type matches process result)
  Value procResultState;
  if (processOp.getNumResults() > 0) {
    auto resultType = processOp.getResultTypes()[0];
    procResultState = allocBuilder.create<arc::AllocStateOp>(
        loc, StateType::get(resultType), storageArg);
  }

  return {resumeTimeState, resumeBlockState, procResultState};
}
```

**Testing considerations:**
- Verify state allocation for processes with and without results
- Check correct types for allocated states

### Step 4: Implement Process Entry Logic

**Description:** Generate the entry logic that checks if the process should resume.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
```cpp
void generateProcessEntry(OpBuilder &builder, Location loc,
                         ProcessState &state, Value storageArg) {
  // Get current time
  auto currentTime = builder.create<arc::CurrentTimeOp>(loc, storageArg);

  // Read resume time
  auto resumeTimeRead = builder.create<arc::StateReadOp>(loc, state.resumeTime);

  // Check if resume time reached: current_time >= resume_time
  auto resume = builder.create<comb::ICmpOp>(
      loc, comb::ICmpPredicate::uge, currentTime, resumeTimeRead);

  // Create scf.if for conditional execution
  auto ifOp = builder.create<scf::IfOp>(loc, resume, /*withElseRegion=*/false);
  builder.setInsertionPointToStart(&ifOp.getThenRegion().front());

  // Inside the if: create scf.execute_region for the switch
  auto executeOp = builder.create<scf::ExecuteRegionOp>(loc, TypeRange{});
  builder.setInsertionPointToStart(&executeOp.getRegion().front());
}
```

**Testing considerations:**
- Verify correct time comparison logic
- Test that processes don't execute before resume time

### Step 5: Implement cf.switch Generation with Optimized Indexing

**Description:** Generate the `cf.switch` operation that jumps to the correct block based on resume_block index.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
```cpp
void generateSwitchOp(OpBuilder &builder, Location loc,
                     ProcessState &state,
                     DenseMap<Block*, ProcessBlockInfo> &blockInfo,
                     Block *haltedBlock) {
  // Read resume_block index
  auto resumeBlockRead = builder.create<arc::StateReadOp>(loc, state.resumeBlock);

  // Build case values and destination blocks
  SmallVector<APInt, 8> caseValues;
  SmallVector<Block*, 8> caseBlocks;

  // Add cases only for blocks with assigned indices
  for (auto &[block, info] : blockInfo) {
    if (info.index.has_value()) {
      caseValues.push_back(APInt(16, *info.index));
      caseBlocks.push_back(block);
    }
  }

  // Add halted state case
  unsigned haltedIndex = caseValues.size();
  caseValues.push_back(APInt(16, haltedIndex));
  caseBlocks.push_back(haltedBlock);

  // Create switch
  builder.create<cf::SwitchOp>(
      loc, resumeBlockRead,
      /*defaultDest=*/haltedBlock,  // Should never happen
      caseValues, caseBlocks);
}
```

**Testing considerations:**
- Verify switch has correct number of cases (only indexed blocks)
- Test that fall-through blocks are not in switch

### Step 6: Implement llhd.wait Lowering

**Description:** Lower `llhd.wait` terminators to state updates and control flow.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
```cpp
LogicalResult lowerWaitOp(llhd::WaitOp waitOp, OpBuilder &builder,
                         ProcessState &state, Value currentTime,
                         DenseMap<Block*, ProcessBlockInfo> &blockInfo) {
  Location loc = waitOp.getLoc();

  // Reject block operands (not supported yet)
  if (!waitOp.getDestOperands().empty()) {
    return waitOp.emitError("block operands on llhd.wait not supported");
  }

  // Reject observed signals (not supported yet)
  if (!waitOp.getObserved().empty()) {
    return waitOp.emitError("signal observation on llhd.wait not supported");
  }

  // Get delay time
  Value delay;
  if (waitOp.getDelay()) {
    // Convert llhd.time to i64 femtoseconds
    delay = builder.create<llhd::TimeToIntOp>(loc, waitOp.getDelay());
  } else {
    // No delay means immediate resume (0 time)
    delay = builder.create<hw::ConstantOp>(loc, builder.getI64Type(), 0);
  }

  // Calculate next resume time: current_time + delay
  auto nextResume = builder.create<comb::AddOp>(loc, currentTime, delay);

  // Write next resume time
  builder.create<arc::StateWriteOp>(loc, state.resumeTime, nextResume, Value{});

  // Get target block index
  Block *targetBlock = waitOp.getDest();
  auto &targetInfo = blockInfo[targetBlock];
  assert(targetInfo.index.has_value() && "Wait target must have index");

  auto targetIndex = builder.create<hw::ConstantOp>(
      loc, builder.getI16Type(), *targetInfo.index);
  builder.create<arc::StateWriteOp>(loc, state.resumeBlock, targetIndex, Value{});

  // Write yield operands to proc_result
  if (!waitOp.getYieldOperands().empty()) {
    assert(waitOp.getYieldOperands().size() == 1 && "Multiple results not supported");
    builder.create<arc::StateWriteOp>(
        loc, state.procResult, waitOp.getYieldOperands()[0], Value{});
  }

  // Replace with scf.yield
  builder.create<scf::YieldOp>(loc);
  waitOp.erase();

  return success();
}
```

**Testing considerations:**
- Test wait with various delay values
- Verify state updates are correct
- Test processes with multiple wait points

### Step 7: Implement llhd.halt Lowering

**Description:** Lower `llhd.halt` terminators to final state updates.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
```cpp
LogicalResult lowerHaltOp(llhd::HaltOp haltOp, OpBuilder &builder,
                         ProcessState &state, unsigned haltedIndex) {
  Location loc = haltOp.getLoc();

  // Set resume time to max i64 (never resume)
  auto maxTime = builder.create<hw::ConstantOp>(
      loc, builder.getI64Type(), APInt(64, -1, /*isSigned=*/false));
  builder.create<arc::StateWriteOp>(loc, state.resumeTime, maxTime, Value{});

  // Set resume block to halted state index
  auto haltedIndexVal = builder.create<hw::ConstantOp>(
      loc, builder.getI16Type(), haltedIndex);
  builder.create<arc::StateWriteOp>(loc, state.resumeBlock, haltedIndexVal, Value{});

  // Write yield operands to proc_result
  if (!haltOp.getYieldOperands().empty()) {
    assert(haltOp.getYieldOperands().size() == 1 && "Multiple results not supported");
    builder.create<arc::StateWriteOp>(
        loc, state.procResult, haltOp.getYieldOperands()[0], Value{});
  }

  // Replace with scf.yield
  builder.create<scf::YieldOp>(loc);
  haltOp.erase();

  return success();
}
```

**Testing considerations:**
- Verify halted processes don't resume
- Test halt with yield operands

### Step 8: Implement Process Initialization

**Description:** Initialize process state in the initial phase (resume at time 0, block 0).

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
```cpp
void initializeProcessState(OpBuilder &initialBuilder, Location loc,
                           ProcessState &state) {
  // Initialize resume_time to 0 (execute immediately)
  auto zeroTime = initialBuilder.create<hw::ConstantOp>(
      loc, initialBuilder.getI64Type(), 0);
  initialBuilder.create<arc::StateWriteOp>(loc, state.resumeTime, zeroTime, Value{});

  // Initialize resume_block to 0 (entry block)
  auto zeroBlock = initialBuilder.create<hw::ConstantOp>(
      loc, initialBuilder.getI16Type(), 0);
  initialBuilder.create<arc::StateWriteOp>(loc, state.resumeBlock, zeroBlock, Value{});

  // Initialize proc_result to zero/default if needed
  if (state.procResult) {
    auto resultType = cast<StateType>(state.procResult.getType()).getType();
    if (auto intType = dyn_cast<IntegerType>(resultType)) {
      auto zeroResult = initialBuilder.create<hw::ConstantOp>(
          loc, intType, 0);
      initialBuilder.create<arc::StateWriteOp>(
          loc, state.procResult, zeroResult, Value{});
    }
  }
}
```

**Testing considerations:**
- Verify processes start at time 0, block 0
- Test initialization with different result types

### Step 9: Implement Complete Process Lowering

**Description:** Tie all pieces together to lower an entire `llhd.process` operation.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
```cpp
LogicalResult lowerProcess(llhd::ProcessOp processOp, arc::ModelOp modelOp) {
  // 1. Analyze process and assign block indices
  ProcessAnalyzer analyzer;
  auto blockInfo = analyzer.analyzeProcess(processOp);

  // 2. Allocate state
  OpBuilder allocBuilder(modelOp.getBodyBlock(), modelOp.getBodyBlock().begin());
  Value storageArg = modelOp.getBodyBlock().getArgument(0);
  auto state = allocateProcessState(allocBuilder, processOp.getLoc(),
                                   storageArg, processOp);

  // 3. Initialize state in initial phase
  // (Assumes model has initial region - handled by LowerState pass)

  // 4. Generate process entry logic in main body
  OpBuilder builder(modelOp.getBodyBlock(), modelOp.getBodyBlock().end());
  generateProcessEntry(builder, processOp.getLoc(), state, storageArg);

  // 5. Clone process blocks into execute_region
  // 6. Generate cf.switch
  // 7. Lower wait/halt terminators
  // 8. Replace process result uses with state reads

  // Replace process results with state reads
  if (processOp.getNumResults() > 0) {
    OpBuilder resultBuilder(processOp);
    auto resultRead = resultBuilder.create<arc::StateReadOp>(
        processOp.getLoc(), state.procResult);
    processOp.getResult(0).replaceAllUsesWith(resultRead);
  }

  processOp.erase();
  return success();
}
```

**Testing considerations:**
- Test complete lowering on simple processes
- Verify all process components are correctly lowered

### Step 10: Add Validation and Rejection Logic

**Description:** Implement validation to reject unsupported LLHD features.

**Files to modify:**
- `lib/Dialect/Arc/Transforms/LowerProcesses.cpp`

**Key implementation details:**
```cpp
LogicalResult validateProcess(llhd::ProcessOp processOp) {
  // Walk the process body and check for unsupported operations
  auto result = processOp.walk([&](Operation *op) {
    // Reject unsupported LLHD ops
    if (isa<llhd::ProbeOp, llhd::DriveOp, llhd::SigOp>(op)) {
      op->emitError("LLHD signal operations not supported in process lowering");
      return WalkResult::interrupt();
    }

    // Check wait ops
    if (auto waitOp = dyn_cast<llhd::WaitOp>(op)) {
      if (!waitOp.getDestOperands().empty()) {
        waitOp.emitError("block operands on llhd.wait not supported");
        return WalkResult::interrupt();
      }
      if (!waitOp.getObserved().empty()) {
        waitOp.emitError("signal observation on llhd.wait not supported");
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  return failure(result.wasInterrupted());
}
```

**Testing considerations:**
- Test rejection of unsupported operations
- Verify error messages are clear

### Step 11: Integrate Pass into Pipeline

**Description:** Add the new pass to the appropriate place in the Arcilator pipeline.

**Files to modify:**
- `lib/Tools/arcilator/pipelines.cpp`
- `include/circt/Tools/arcilator/pipelines.h`

**Key implementation details:**
```cpp
// In pipelines.cpp, in populateArcStateLoweringPipeline:
void circt::populateArcStateLoweringPipeline(
    OpPassManager &pm, const ArcStateLoweringOptions &options) {
  pm.addPass(arc::createLowerStatePass());

  // Add LowerProcesses pass after LowerState
  pm.addPass(arc::createLowerProcessesPass());

  // ... rest of pipeline
}
```

**Pass ordering:**
- After `LowerStatePass` (which converts hw.module to arc.model)
- Before `AllocateStatePass` (which assigns offsets to allocated states)

**Testing considerations:**
- Test full pipeline with processes
- Verify pass ordering is correct

### Step 12: Create Manual Tests with arc.sim.* Operations

**Description:** Create comprehensive tests that use `arc.sim.*` operations to verify process timing behavior.

**Files to create:**
- `test/Dialect/Arc/lower-processes.mlir` (transformation test)
- `integration_test/arcilator/JIT/process-basic.mlir` (JIT execution test)

**Test 1: Basic transformation test**
```mlir
// RUN: circt-opt %s --arc-lower-state --arc-lower-processes | FileCheck %s

hw.module @Foo(out x : i42) {
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c0_i42 = hw.constant 0 : i42
  %c42_i42 = hw.constant 42 : i42
  %c1337_i42 = hw.constant 1337 : i42

  %1 = llhd.process -> i42 {
    llhd.wait yield (%c0_i42 : i42), delay %time, ^bb1
  ^bb1:
    llhd.wait yield (%c42_i42 : i42), delay %time, ^bb2
  ^bb2:
    llhd.halt %c1337_i42 : i42
  }
  hw.output %1 : i42
}

// CHECK-LABEL: arc.model @Foo
// CHECK: %[[RESUME_TIME:.+]] = arc.alloc_state %arg0 : !arc.state<i64>
// CHECK: %[[RESUME_BLOCK:.+]] = arc.alloc_state %arg0 : !arc.state<i16>
// CHECK: %[[PROC_RESULT:.+]] = arc.alloc_state %arg0 : !arc.state<i42>
// CHECK: %[[CURRENT_TIME:.+]] = arc.current_time %arg0
// CHECK: %[[RESUME_TIME_READ:.+]] = arc.state_read %[[RESUME_TIME]]
// CHECK: %[[RESUME:.+]] = comb.icmp uge %[[CURRENT_TIME]], %[[RESUME_TIME_READ]]
// CHECK: scf.if %[[RESUME]]
// CHECK:   scf.execute_region
// CHECK:     %[[RESUME_BLOCK_READ:.+]] = arc.state_read %[[RESUME_BLOCK]]
// CHECK:     cf.switch %[[RESUME_BLOCK_READ]] : i16, [
// CHECK:       0: ^[[BB0:.+]],
// CHECK:       1: ^[[BB1:.+]],
// CHECK:       2: ^[[BB2:.+]],
// CHECK:       3: ^[[HALTED:.+]]
// CHECK:     ]
// CHECK:   ^[[BB0]]:
// CHECK:     arc.state_write %[[RESUME_BLOCK]] = %c1_i16
// CHECK:     arc.state_write %[[PROC_RESULT]] = %c0_i42
// CHECK:   ^[[BB1]]:
// CHECK:     arc.state_write %[[RESUME_BLOCK]] = %c2_i16
// CHECK:     arc.state_write %[[PROC_RESULT]] = %c42_i42
// CHECK:   ^[[BB2]]:
// CHECK:     arc.state_write %[[RESUME_BLOCK]] = %c3_i16
// CHECK:     arc.state_write %[[PROC_RESULT]] = %c1337_i42
// CHECK:   ^[[HALTED]]:
// CHECK:     scf.yield
```

**Test 2: JIT execution test with timing verification**
```mlir
// RUN: arcilator %s --run --jit-entry=main | FileCheck %s
// REQUIRES: arcilator-jit

hw.module @ProcessTest(out x : i42) {
  %time = llhd.constant_time <1000000000000000, 0d, 0e>  // 1ns in femtoseconds
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
  %c0_i64 = arith.constant 0 : i64
  %c1ns_i64 = arith.constant 1000000000000000 : i64  // 1ns in femtoseconds
  %c2ns_i64 = arith.constant 2000000000000000 : i64
  %c3ns_i64 = arith.constant 3000000000000000 : i64
  %c4ns_i64 = arith.constant 4000000000000000 : i64

  arc.sim.instantiate @ProcessTest as %model {
    // Time 0: should output 0
    arc.sim.set_time %model, %c0_i64 : i64, !arc.sim.instance<@ProcessTest>
    arc.sim.step %model : !arc.sim.instance<@ProcessTest>
    %x0 = arc.sim.get_port %model, "x" : i42, !arc.sim.instance<@ProcessTest>
    arc.sim.emit "x", %x0 : i42
    // CHECK: x = 00000000000

    // Time 1ns: should output 42
    arc.sim.set_time %model, %c1ns_i64 : i64, !arc.sim.instance<@ProcessTest>
    arc.sim.step %model : !arc.sim.instance<@ProcessTest>
    %x1 = arc.sim.get_port %model, "x" : i42, !arc.sim.instance<@ProcessTest>
    arc.sim.emit "x", %x1 : i42
    // CHECK: x = 0000000002a

    // Time 2ns: should output 1337
    arc.sim.set_time %model, %c2ns_i64 : i64, !arc.sim.instance<@ProcessTest>
    arc.sim.step %model : !arc.sim.instance<@ProcessTest>
    %x2 = arc.sim.get_port %model, "x" : i42, !arc.sim.instance<@ProcessTest>
    arc.sim.emit "x", %x2 : i42
    // CHECK: x = 00000000539

    // Time 3ns: should output 9001
    arc.sim.set_time %model, %c3ns_i64 : i64, !arc.sim.instance<@ProcessTest>
    arc.sim.step %model : !arc.sim.instance<@ProcessTest>
    %x3 = arc.sim.get_port %model, "x" : i42, !arc.sim.instance<@ProcessTest>
    arc.sim.emit "x", %x3 : i42
    // CHECK: x = 00000002329

    // Time 4ns: should output 0 (halted)
    arc.sim.set_time %model, %c4ns_i64 : i64, !arc.sim.instance<@ProcessTest>
    arc.sim.step %model : !arc.sim.instance<@ProcessTest>
    %x4 = arc.sim.get_port %model, "x" : i42, !arc.sim.instance<@ProcessTest>
    arc.sim.emit "x", %x4 : i42
    // CHECK: x = 00000000000
  }

  return
}
```

**Test 3: Process with fall-through blocks (optimized indexing)**
```mlir
// RUN: circt-opt %s --arc-lower-state --arc-lower-processes | FileCheck %s

hw.module @FallThrough(out x : i8) {
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c0_i8 = hw.constant 0 : i8
  %c1_i8 = hw.constant 1 : i8
  %c2_i8 = hw.constant 2 : i8

  %result = llhd.process -> i8 {
    cf.br ^compute
  ^compute:
    %val = comb.add %c0_i8, %c1_i8 : i8
    llhd.wait yield (%val : i8), delay %time, ^next
  ^next:
    llhd.halt %c2_i8 : i8
  }
  hw.output %result : i8
}

// CHECK-LABEL: arc.model @FallThrough
// CHECK: cf.switch %{{.+}} : i16, [
// CHECK:   0: ^[[ENTRY:.+]],
// CHECK:   1: ^[[NEXT:.+]],
// CHECK:   2: ^[[HALTED:.+]]
// CHECK: ]
// CHECK: ^[[ENTRY]]:
// CHECK:   cf.br ^[[COMPUTE:.+]]
// CHECK: ^[[COMPUTE]]:
// CHECK:   %[[VAL:.+]] = comb.add
// CHECK:   arc.state_write {{.+}} = %c1_i16
// CHECK: ^[[NEXT]]:
// CHECK:   arc.state_write {{.+}} = %c2_i16
// Note: ^compute block has no index, falls through naturally
```

**Testing considerations:**
- Verify output values at each time point
- Test that processes execute at correct times
- Verify halted processes don't execute
- Test fall-through blocks work correctly

## File Changes Summary

### Files to Create
1. **`lib/Dialect/Arc/Transforms/LowerProcesses.cpp`** - Main implementation of process lowering pass
2. **`test/Dialect/Arc/lower-processes.mlir`** - Transformation tests
3. **`integration_test/arcilator/JIT/process-basic.mlir`** - JIT execution tests with timing verification

### Files to Modify
1. **`lib/Dialect/Arc/Transforms/CMakeLists.txt`** - Add LowerProcesses.cpp to build
2. **`include/circt/Dialect/Arc/ArcPasses.td`** - Add LowerProcessesPass definition
3. **`lib/Tools/arcilator/pipelines.cpp`** - Integrate pass into pipeline
4. **`include/circt/Tools/arcilator/pipelines.h`** - Update pipeline interface if needed

### Files to Delete
None

## Testing Strategy

### Unit Tests
**File:** `test/Dialect/Arc/lower-processes.mlir`

Test cases:
1. Basic process with single wait
2. Process with multiple waits
3. Process with halt
4. Process with fall-through blocks (optimized indexing)
5. Process with no results
6. Process with results
7. Rejection of block operands on wait
8. Rejection of signal observation on wait
9. Rejection of unsupported LLHD ops

### Integration Tests
**File:** `integration_test/arcilator/JIT/process-basic.mlir`

Test cases:
1. Process timing verification (output changes at correct times)
2. Multiple processes in same module
3. Process that halts
4. Process with immediate resume (no delay)

### Manual Testing Steps
1. Build CIRCT with new pass: `ninja -C build check-circt`
2. Run transformation tests: `ninja -C build check-circt`
3. Run JIT tests: `ninja -C build check-circt-integration`
4. Verify output values match expected timing

## Rollback Plan

If issues are discovered:

1. **Remove pass from pipeline:**
   - Comment out `pm.addPass(arc::createLowerProcessesPass())` in `pipelines.cpp`
   - This allows the rest of the pipeline to work without process lowering

2. **Revert changes:**
   ```bash
   git revert <commit-hash>
   ```

3. **Data migration rollback:**
   - No data migration needed - this is a compiler pass
   - No persistent state to roll back

## Estimated Effort

### Complexity Assessment: **Medium-High**

**Breakdown:**
- **Step 1-2 (Pass infrastructure & block analysis):** 4-6 hours
- **Step 3-5 (State allocation & entry logic):** 6-8 hours
- **Step 6-7 (Wait/halt lowering):** 8-10 hours
- **Step 8-9 (Initialization & complete lowering):** 6-8 hours
- **Step 10-11 (Validation & pipeline integration):** 4-6 hours
- **Step 12 (Testing):** 8-12 hours

**Total estimated time:** 36-50 hours (roughly 1-1.5 weeks for one developer)

### Risk Factors
- **Medium risk:** Complexity of control flow transformation
- **Medium risk:** Correct handling of time arithmetic (femtoseconds)
- **Low risk:** Integration with existing Arc infrastructure (well-established patterns)

## Implementation Notes

### Key Design Decisions

1. **Optimized Block Indexing:**
   - Only entry block, wait targets, and halted state get indices
   - Fall-through blocks have no index and execute naturally
   - Reduces switch case count and improves performance

2. **Time Representation:**
   - Use i64 for time in femtoseconds (matches `arc.current_time`)
   - Use `llhd.time_to_int` to convert LLHD time to i64
   - Max i64 value (-1 unsigned) represents "never resume"

3. **State Storage:**
   - `resume_time`: i64 - when to resume execution
   - `resume_block`: i16 - which block to resume at (index)
   - `proc_result`: varies - current output value of process

4. **Control Flow:**
   - Use `scf.if` to check if resume time reached
   - Use `scf.execute_region` to contain the switch
   - Use `cf.switch` to jump to correct block
   - Use `scf.yield` to exit regions

### Code Patterns to Follow

**State allocation pattern (from AllocateState.cpp):**
```cpp
auto allocOp = builder.create<arc::AllocStateOp>(
    loc, StateType::get(valueType), storageArg);
```

**State read/write pattern (from LowerState.cpp):**
```cpp
auto readOp = builder.create<arc::StateReadOp>(loc, stateValue);
builder.create<arc::StateWriteOp>(loc, stateValue, newValue, condition);
```

**Current time pattern (from LowerState.cpp):**
```cpp
auto currentTime = builder.create<arc::CurrentTimeOp>(loc, storageArg);
```

### Dependencies on Other Components

**Required operations:**
- `arc.alloc_state` - allocate state storage
- `arc.state_read` - read state value
- `arc.state_write` - write state value
- `arc.current_time` - get current simulation time
- `llhd.time_to_int` - convert time to i64
- `comb.icmp`, `comb.add` - arithmetic operations
- `cf.switch` - multi-way branch
- `scf.if`, `scf.execute_region`, `scf.yield` - structured control flow

**Pass dependencies:**
- Must run after `LowerStatePass` (creates arc.model)
- Must run before `AllocateStatePass` (assigns offsets)

### Edge Cases to Handle

1. **Process with no results:** Don't allocate `proc_result` state
2. **Process with immediate wait (no delay):** Use delay of 0
3. **Process with only halt (no waits):** Still needs entry block index
4. **Empty process body:** Should be rejected or handled gracefully
5. **Process with multiple results:** Reject for now (not in scope)
6. **Nested processes:** Should not occur (processes are top-level in modules)

### Future Enhancements (Out of Scope)

1. **Block operands on wait:** Requires storing live values across waits
2. **Signal observation:** Requires integration with signal change tracking
3. **Multiple process results:** Requires tuple/struct handling
4. **Earliest event time tracking:** Requires global state coordination
5. **Automatic runner generation:** Requires additional infrastructure

## References

### Relevant Files to Study
- `lib/Dialect/Arc/Transforms/LowerState.cpp` - Pattern for lowering to arc.model
- `lib/Dialect/Arc/Transforms/AllocateState.cpp` - State allocation patterns
- `lib/Dialect/LLHD/Transforms/LowerProcesses.cpp` - Existing LLHD process lowering
- `test/Dialect/Arc/lower-state.mlir` - Examples of lowered Arc code
- `integration_test/arcilator/JIT/basic.mlir` - Examples of arc.sim.* usage

### MLIR Documentation
- [SCF Dialect](https://mlir.llvm.org/docs/Dialects/SCFDialect/)
- [ControlFlow Dialect](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/)
- [Pass Infrastructure](https://mlir.llvm.org/docs/PassManagement/)

### CIRCT Documentation
- Arc Dialect: `docs/Dialects/Arc.md`
- LLHD Dialect: `docs/Dialects/LLHD.md`
- Arcilator Tool: `docs/Tools/arcilator.md`


