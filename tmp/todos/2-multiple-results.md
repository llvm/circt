# TODO: Multiple Results - Processes with Multiple Results

## Severity: Medium

## The Problem

LLHD processes can have multiple results:

```mlir
%result1, %result2 = llhd.process -> i32, i8 {
  llhd.wait yield (%val1, %val2 : i32, i8), delay %time, ^bb1
^bb1:
  llhd.halt %final1, %final2 : i32, i8
}
```

Our current implementation only handles single-result processes.

## Current Implementation

Location: `lib/Dialect/Arc/Transforms/LowerProcesses.cpp:118-126`

```cpp
// Get result type
if (processOp.getNumResults() == 0) {
  return processOp.emitError("process must have exactly one result");
}
if (processOp.getNumResults() > 1) {
  return processOp.emitError(
      "processes with multiple results are not yet supported");
}
analysis.resultType = processOp.getResult(0).getType();
```

We explicitly reject processes with multiple results.

## What Needs to Happen

To support multiple results, we need to:

### 1. Allocate Multiple proc_result States

One for each result:

```cpp
SmallVector<arc::AllocStateOp> procResultStates;
for (Type resultType : processOp.getResultTypes()) {
  auto state = arc::AllocStateOp::create(
      allocBuilder, loc, storageArg, resultType);
  procResultStates.push_back(state);
}
```

### 2. Write All Yield Values

In wait/halt lowering:

```cpp
// In llhd.wait lowering
for (unsigned i = 0; i < waitOp.getYieldOperands().size(); ++i) {
  Value yieldValue = mapping.lookupOrDefault(waitOp.getYieldOperands()[i]);
  arc::StateWriteOp::create(bodyBuilder, loc, 
                             procResultStates[i].getResult(),
                             yieldValue, Value{});
}
```

### 3. Read All Results

At the end:

```cpp
SmallVector<Value> results;
for (auto state : procResultStates) {
  auto read = arc::StateReadOp::create(bodyBuilder, loc, state.getResult());
  results.push_back(read);
}
// Replace process op with multiple results
processOp.replaceAllUsesWith(results);
```

### 4. Update the Analysis Struct

Track multiple result types:

```cpp
struct ProcessAnalysis {
  // ...
  SmallVector<Type> resultTypes;  // Instead of single resultType
};
```

## Implementation Steps

1. **Update ProcessAnalysis struct**:
   ```cpp
   struct ProcessAnalysis {
     Block *entryBlock;
     SmallVector<Block *> waitTargetBlocks;
     DenseMap<Block *, unsigned> blockIndices;
     unsigned haltedIndex;
     SmallVector<Type> resultTypes;  // Changed from Type resultType
   };
   ```

2. **Update analyzeProcess function**:
   ```cpp
   // Remove the check for > 1 results
   if (processOp.getNumResults() == 0) {
     return processOp.emitError("process must have at least one result");
   }
   
   // Store all result types
   for (auto result : processOp.getResults()) {
     analysis.resultTypes.push_back(result.getType());
   }
   ```

3. **Update state allocation**:
   ```cpp
   SmallVector<arc::AllocStateOp> procResultStates;
   for (Type resultType : analysis.resultTypes) {
     auto procResultState = arc::AllocStateOp::create(
         allocBuilder, loc, storageArg, resultType);
     procResultStates.push_back(procResultState);
   }
   ```

4. **Update wait lowering**:
   ```cpp
   if (!waitOp.getYieldOperands().empty()) {
     for (unsigned i = 0; i < waitOp.getYieldOperands().size(); ++i) {
       Value yieldValue = mapping.lookupOrDefault(waitOp.getYieldOperands()[i]);
       arc::StateWriteOp::create(bodyBuilder, loc,
                                  procResultStates[i].getResult(),
                                  yieldValue, Value{});
     }
   }
   ```

5. **Update halt lowering**:
   ```cpp
   if (haltOp.getNumOperands() > 0) {
     for (unsigned i = 0; i < haltOp.getNumOperands(); ++i) {
       Value finalValue = mapping.lookupOrDefault(haltOp.getOperand(i));
       arc::StateWriteOp::create(bodyBuilder, loc,
                                  procResultStates[i].getResult(),
                                  finalValue, Value{});
     }
   }
   ```

6. **Update result reading**:
   ```cpp
   SmallVector<Value> results;
   for (auto procResultState : procResultStates) {
     auto resultRead = arc::StateReadOp::create(
         bodyBuilder, loc, procResultState.getResult());
     results.push_back(resultRead);
   }
   processOp.replaceAllUsesWith(results);
   ```

7. **Add tests**:
   - Test with 2 results of different types
   - Test with 3+ results
   - Test that yield operands match result count
   - Test error cases (mismatched counts, types)

## Example That Would Work After Implementation

```mlir
hw.module @MultiResult(out x : i32, out y : i8) {
  %time = llhd.constant_time <1ns, 0d, 0e>
  %c0_i32 = hw.constant 0 : i32
  %c0_i8 = hw.constant 0 : i8
  %c42_i32 = hw.constant 42 : i32
  %c99_i8 = hw.constant 99 : i8
  
  %r1, %r2 = llhd.process -> i32, i8 {
    llhd.wait yield (%c0_i32, %c0_i8 : i32, i8), delay %time, ^bb1
  ^bb1:
    llhd.halt %c42_i32, %c99_i8 : i32, i8
  }
  
  hw.output %r1, %r2 : i32, i8
}
```

This would lower to:

```mlir
arc.model @MultiResult {
^bb0(%storage: !arc.storage):
  %resume_time_state = arc.alloc_state %storage : !arc.state<i64>
  %resume_block_state = arc.alloc_state %storage : !arc.state<i16>
  %proc_result_0_state = arc.alloc_state %storage : !arc.state<i32>  // First result
  %proc_result_1_state = arc.alloc_state %storage : !arc.state<i8>   // Second result
  
  // ... control flow ...
  
  %r1 = arc.state_read %proc_result_0_state : <i32>
  %r2 = arc.state_read %proc_result_1_state : <i8>
}
```

## Why It's Not Implemented

1. **Complexity**: Adds complexity to the implementation
2. **Uncommon case**: Most processes have a single result
3. **Testing**: Would need comprehensive tests for multiple results
4. **Time constraints**: Wanted to get the basic case working first

## Impact

Without this fix:
- Processes with multiple results are rejected with an error
- Users must split into multiple single-result processes

## Workaround

Use multiple single-result processes instead of one multi-result process.

## Effort Estimate

**Low** - Straightforward extension of existing code. The infrastructure is already in place, just need to handle arrays instead of single values.

