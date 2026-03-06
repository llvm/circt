# TODO: Time Conversion - `llhd.time` to i64 femtoseconds

## Severity: High

## The Problem

LLHD uses a special time type `!llhd.time` that represents simulation time with three components:
- **Time value** (e.g., 1, 10, 100)
- **Delta cycles** (for zero-time ordering)
- **Epsilon** (for sub-delta ordering)

For example: `#llhd.time<1ns, 0d, 0e>` means "1 nanosecond, 0 delta cycles, 0 epsilon"

Arc's simulation model uses **i64 femtoseconds** as the time representation. We need to convert from LLHD's time type to femtoseconds.

## Current Implementation

Location: `lib/Dialect/Arc/Transforms/LowerProcesses.cpp:323-333`

```cpp
// 3. Calculate the resume time (current_time + delay)
auto currentTime =
    arc::CurrentTimeOp::create(bodyBuilder, loc, storageArg);
Value resumeTime = currentTime;

if (waitOp.getDelay()) {
  // TODO: Properly convert llhd.time to i64 femtoseconds
  // For now, we'll just use current time (effectively no delay)
  // In a real implementation, we'd need to extract the time value
  // and convert it to femtoseconds
}
```

As you can see, we're just using `currentTime` as the `resumeTime`, which means **the process resumes immediately** instead of waiting for the specified delay.

## What Needs to Happen

We need to:

1. **Extract the time value** from `llhd.constant_time` or the delay operand
2. **Convert to femtoseconds** based on the time unit
3. **Add it to current time** to get the resume time
4. **Handle delta and epsilon** (probably by ignoring them for now, or encoding them in the lower bits)

## Example Implementation Sketch

```cpp
if (waitOp.getDelay()) {
  Value delay = waitOp.getDelay();
  
  // If it's a constant time, we can extract the value at compile time
  if (auto constTime = delay.getDefiningOp<llhd::ConstantTimeOp>()) {
    auto timeAttr = constTime.getValue();
    
    // Extract components
    uint64_t time = timeAttr.getTime();
    uint64_t delta = timeAttr.getDelta();
    uint64_t epsilon = timeAttr.getEpsilon();
    
    // Convert to femtoseconds based on time unit
    // LLHD time units: s, ms, us, ns, ps, fs
    TimeUnit unit = timeAttr.getTimeUnit();
    uint64_t femtoseconds = convertToFemtoseconds(time, unit);
    
    // Create constant for the delay in femtoseconds
    auto delayConst = bodyBuilder.create<arith::ConstantOp>(
        loc, bodyBuilder.getI64IntegerAttr(femtoseconds));
    
    // Add to current time
    resumeTime = bodyBuilder.create<arith::AddIOp>(
        loc, currentTime, delayConst);
  } else {
    // Runtime time value - would need runtime conversion
    // This is more complex and might require helper functions
  }
}
```

## Implementation Steps

1. **Research LLHD time API**:
   - Look at `include/circt/Dialect/LLHD/IR/LLHDTypes.h`
   - Find how to extract time, delta, epsilon from `TimeAttr`
   - Find the time unit enumeration

2. **Implement conversion function**:
   ```cpp
   uint64_t convertToFemtoseconds(uint64_t time, TimeUnit unit) {
     switch (unit) {
       case TimeUnit::Second:      return time * 1'000'000'000'000'000ULL;
       case TimeUnit::Millisecond: return time * 1'000'000'000'000ULL;
       case TimeUnit::Microsecond: return time * 1'000'000'000ULL;
       case TimeUnit::Nanosecond:  return time * 1'000'000ULL;
       case TimeUnit::Picosecond:  return time * 1'000ULL;
       case TimeUnit::Femtosecond: return time;
     }
   }
   ```

3. **Handle delta and epsilon**:
   - Option 1: Ignore them (assume 0)
   - Option 2: Encode in lower bits of i64
   - Option 3: Add separate state for delta/epsilon

4. **Handle runtime time values**:
   - Create a helper function or arc operation
   - Lower to runtime conversion code

5. **Add tests**:
   - Test with various time units (ns, ps, fs, etc.)
   - Test with delta and epsilon values
   - Test with runtime time values

## Why It's Not Implemented

1. **LLHD time API complexity**: Need to understand the exact API for extracting time components
2. **Delta and epsilon handling**: Not clear how to encode these in i64 (probably need to reserve lower bits)
3. **Runtime conversion**: If the delay isn't a constant, we need runtime conversion logic
4. **Testing**: Would need tests with various time units and values

## Impact

Without this fix:
- All processes resume immediately (no actual delay)
- Integration test is marked XFAIL
- Cannot properly simulate timed behavior
- Process semantics are incorrect

## Workaround

Use processes without delays, or accept that delays are ignored.

## Effort Estimate

**Medium** - Requires understanding LLHD time API and implementing conversion logic, but the approach is straightforward.

