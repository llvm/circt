# Force/Release Test Suite for ProbesToSignals Pass

This directory contains comprehensive lit tests for the force/release functionality in the ProbesToSignals pass.

## Test Files

### 1. probes-to-signals-force-release.mlir

Basic functionality tests covering:

- **SimpleForce**: Single force operation on a forceable wire
  - Verifies state register creation
  - Checks when-block structure for state updates
  - Validates override logic generation

- **SimpleRelease**: Single release operation
  - Tests release-only scenario
  - Verifies state register for tracking forced state

- **ForceAndRelease**: Combined force and release operations
  - Tests priority mux generation for forceWins
  - Validates OR gates for combining enables
  - Checks AND gates for forceActive/releaseActive
  - Verifies when/else structure for state updates

- **MultipleForces**: Multiple concurrent force operations
  - Tests priority mux for force value selection
  - Validates OR combination of force enables
  - Ensures last-enable-wins semantics

- **ForceInitial**: force_initial operation
  - Tests clock extraction from mixed force types
  - Verifies force_initial participates in priority logic

- **ReleaseInitial**: release_initial operation
  - Tests release without explicit clock
  - Validates integration with clocked operations

- **RWProbeTarget**: Force on rwprobe target
  - Tests force on probe created via ref.rwprobe
  - Validates target resolution through probeToHWMap

- **DetailedStructure**: Complete circuit generation
  - Comprehensive checks on all generated components:
    - Constant creation for boolean values
    - Priority mux tree for forceWins
    - Invalid value for default force value
    - State register creation with proper names
    - forceActive computation logic
    - releaseActive computation logic
    - State update when/else structure
    - Override logic generation
  - This is the most complete test showing exact IR structure

- **OpsDeleted**: Verification of operation removal
  - Ensures original force/release ops are deleted
  - Confirms clean transformation

### 2. probes-to-signals-force-release-errors.mlir

Error handling tests:

- **MultipleClocks**: Error when different clocks used
  - Verifies diagnostic: "multiple different clocks on force/release operations targeting the same probe not supported"

- **NoClockError**: Error when only force_initial/release_initial
  - Verifies diagnostic: "no clock found for force/release operations"
  - Tests the requirement that at least one clocked operation must be present

### 3. probes-to-signals-force-with-probes.mlir

Integration tests with other probe operations:

- **ForceWithRefDefine**: Force with ref.define chain
  - Tests force on probe forwarded through module hierarchy
  - Validates port type conversion (rwprobe -> signal)
  - Checks ref.define -> connect transformation

- **ForceOnRegister**: Force on forceable register
  - Tests force applied to register instead of wire
  - Validates override logic connects to register
  - Ensures existing register connections preserved

- **ForceWithResolve**: Force with ref.resolve
  - Tests force on probe that is also observed via resolve
  - Validates ref.cast from rwprobe to probe
  - Checks that observations see forced value

- **MultiModule**: Multi-module hierarchy
  - Tests force at different hierarchy levels
  - Validates each module processes independently
  - Checks probe forwarding through hierarchy

- **ForceBundle**: Force with aggregate types
  - Tests force on bundle-typed probe
  - Validates state register creation for aggregate
  - Checks type preservation through transformation

- **ForceableRemoved**: Forceable attribute removal
  - Verifies forceable attribute is removed after processing
  - Confirms wire is demoted to non-forceable
  - Part of the forceable demotion logic

## Running Tests

Run all force/release tests:
```bash
./build/bin/llvm-lit test/Dialect/FIRRTL/probes-to-signals-force*.mlir
```

Run specific test file:
```bash
./build/bin/llvm-lit test/Dialect/FIRRTL/probes-to-signals-force-release.mlir
```

Run with verbose output:
```bash
./build/bin/llvm-lit -v test/Dialect/FIRRTL/probes-to-signals-force-release.mlir
```

## Test Coverage

The test suite covers:

✅ Single force/release operations
✅ Multiple concurrent forces (priority resolution)
✅ Force and release together
✅ force_initial and release_initial variants
✅ RWProbe from forceable declarations
✅ RWProbe from ref.rwprobe operations
✅ Integration with ref.define chains
✅ Integration with ref.resolve observations
✅ Force on wires and registers
✅ Aggregate types (bundles)
✅ Multi-module hierarchies
✅ Error cases (multiple clocks, missing clock)
✅ Complete generated circuit structure
✅ Operation deletion and cleanup

## Implementation Notes

These tests validate the synchronous force/release implementation that:

1. Collects all force/release operations per RWProbe
2. Generates priority mux logic for conflict resolution
3. Creates state registers (forced, forcedValue)
4. Generates enable computation logic
5. Produces state update logic (when/else)
6. Generates override connections
7. Removes original force/release operations

The implementation follows the proposal's synchronous scheme with same-clock-domain assumptions.
