# TODO: Observed Values - `llhd.wait` Observed Values

## Severity: Low (Architectural)

## The Problem

LLHD's `llhd.wait` can specify **observed values** - the process should resume when any of these values change, not just after a time delay:

```mlir
%sig1 = llhd.prb %signal1 : i32
%sig2 = llhd.prb %signal2 : i8

llhd.process -> i32 {
  // Wait until sig1 or sig2 changes, OR 1ns passes
  llhd.wait delay %time, (%sig1, %sig2 : i32, i8), ^bb1
^bb1:
  llhd.halt %sig1 : i32
}
```

This is like Verilog's `@(posedge clk or negedge rst)` - the process wakes up when specific signals change.

## Current Implementation

Location: `lib/Dialect/Arc/Transforms/LowerProcesses.cpp:306-337`

We completely **ignore** the observed values from `waitOp.getObserved()`. We only handle the time delay.

```cpp
if (auto waitOp = dyn_cast<llhd::WaitOp>(terminator)) {
  // Lower llhd.wait to state writes
  // 1. Write the yield value to proc_result state (if any)
  if (!waitOp.getYieldOperands().empty()) {
    // ... handle yield values ...
  }
  
  // 2. Get the target block index
  // 3. Calculate the resume time (current_time + delay)
  // 4. Write resume_time state
  // 5. Write resume_block state
  // 6. Yield from the execute region
  
  // NOTE: waitOp.getObserved() is completely ignored!
}
```

## What Needs to Happen

To properly support observed values, we need to:

### 1. Track Previous Values of Observed Signals

```cpp
// For each observed value, allocate a state to store its previous value
SmallVector<arc::AllocStateOp> observedStates;
for (Value observed : waitOp.getObserved()) {
  auto state = arc::AllocStateOp::create(
      allocBuilder, loc, storageArg, observed.getType());
  observedStates.push_back(state);
}
```

### 2. Check for Changes in the Resume Condition

```cpp
// Build condition: time_reached OR any_value_changed
Value shouldResume = timeReached;

for (unsigned i = 0; i < observedValues.size(); ++i) {
  Value currentVal = observedValues[i];
  Value prevVal = arc::StateReadOp::create(builder, loc, observedStates[i]);
  Value changed = builder.create<comb::ICmpOp>(
      loc, comb::ICmpPredicate::ne, currentVal, prevVal);
  shouldResume = builder.create<comb::OrOp>(loc, shouldResume, changed);
}
```

### 3. Update Stored Values When Resuming

```cpp
// After resuming, update the stored values
for (unsigned i = 0; i < observedValues.size(); ++i) {
  arc::StateWriteOp::create(builder, loc, observedStates[i], 
                             observedValues[i], Value{});
}
```

### 4. Handle Model-Level Logic

The observed values need to be **evaluated every simulation step**, not just when the process resumes. This means we need to:
- Store the observed values in the model
- Check them on every step
- Wake up the process if any changed

## Architectural Challenge

The big challenge is that Arc's model is **time-driven** (processes run at specific times), but observed values make it **event-driven** (processes run when values change).

We'd need to:

1. **Evaluate observed expressions** on every simulation step
2. **Compare with previous values** to detect changes
3. **Wake up the process** if any value changed, even if the time hasn't been reached

This might require:
- A separate "observed values" region in the model that runs every step
- A mechanism to trigger process execution outside the normal time-based scheduling
- Integration with Arc's existing state tracking

## Example Implementation Sketch

```mlir
arc.model @ProcessWithObserved {
^bb0(%storage: !arc.storage):
  // Allocate states for observed values
  %observed_0_prev = arc.alloc_state %storage : !arc.state<i32>
  %observed_1_prev = arc.alloc_state %storage : !arc.state<i8>
  
  // ... other state allocations ...
  
  // On every step, check if observed values changed
  %sig1_current = ... evaluate sig1 ...
  %sig2_current = ... evaluate sig2 ...
  
  %sig1_prev = arc.state_read %observed_0_prev : <i32>
  %sig2_prev = arc.state_read %observed_1_prev : <i8>
  
  %sig1_changed = comb.icmp ne %sig1_current, %sig1_prev : i32
  %sig2_changed = comb.icmp ne %sig2_current, %sig2_prev : i8
  %any_changed = comb.or %sig1_changed, %sig2_changed : i1
  
  %time_reached = comb.icmp uge %current_time, %resume_time : i64
  %should_resume = comb.or %time_reached, %any_changed : i1
  
  scf.if %should_resume {
    // Update stored values
    arc.state_write %observed_0_prev = %sig1_current : <i32>
    arc.state_write %observed_1_prev = %sig2_current : <i8>
    
    // Run the process
    scf.execute_region {
      // ... process logic ...
    }
  }
}
```

## Implementation Challenges

### Challenge 1: Where to Evaluate Observed Expressions

Observed values are SSA values from outside the process. Where do we evaluate them?

Options:
- **In the model body**: Evaluate on every step (expensive)
- **In a separate region**: Create a "combinational" region that runs every step
- **Lazy evaluation**: Only evaluate when checking if process should resume

### Challenge 2: Integration with Arc's Execution Model

Arc models have a specific execution model:
- Clock domains run at specific times
- States are updated in a specific order
- Processes would need to fit into this model

We'd need to:
- Define when observed values are checked (every step? every clock edge?)
- Define the order of execution (before/after other logic?)
- Handle interactions with other processes

### Challenge 3: Performance

Checking observed values every step could be expensive:
- Need to evaluate expressions
- Need to compare with previous values
- Need to conditionally execute process

For many observed values or complex expressions, this could significantly slow down simulation.

## Why It's Not Implemented

1. **Architectural complexity**: Requires significant changes to how Arc models work
2. **Event-driven vs time-driven**: Fundamental mismatch between LLHD's event model and Arc's time model
3. **Performance**: Checking observed values every step could be expensive
4. **Scope**: This is a major feature that goes beyond basic process lowering
5. **Uncommon in synthesis**: Most synthesizable processes use clocks, not arbitrary observed values

## Current Behavior

If you have a process with observed values:
```mlir
llhd.wait (%sig1, %sig2 : i32, i8), ^bb1
```

The observed values are **completely ignored**. The process will only resume based on time, not when the signals change. This means:
- The process might miss signal changes
- The behavior will be incorrect compared to LLHD semantics
- **No error or warning is produced**

## Impact

Without this fix:
- Processes with observed values have incorrect semantics
- Event-driven processes cannot be properly simulated
- No error is produced, leading to silent incorrectness

## Workaround

Use time-based waits only. For clock-driven processes, use explicit clock signals and time-based scheduling instead of observed values.

## Effort Estimate

**High** - This is an architectural change that requires:
- Rethinking how Arc models execute
- Adding event-driven execution support
- Significant performance optimization
- Extensive testing

This is more of a **future enhancement** than a bug fix. It would require a design document and significant discussion with the CIRCT community.

## Recommendation

For now, **document this limitation** and reject processes with observed values with a clear error message:

```cpp
if (!waitOp.getObserved().empty()) {
  return processOp.emitError(
      "processes with observed values are not yet supported. "
      "Use time-based waits only.");
}
```

This would prevent silent incorrectness and make the limitation explicit to users.

