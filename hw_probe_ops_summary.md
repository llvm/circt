# HW Probe Operations - Executive Summary

## Problem Statement

Currently, the FIRRTL dialect contains probe operations and the LowerXMR pass that converts these to XMRs. This creates a tight coupling between FIRRTL and XMR generation, making it difficult for other dialects to use probe semantics without depending on FIRRTL.

## Proposed Solution

Move probe operations and XMR lowering to the HW dialect, creating a clean separation:

```
FIRRTL Probes → [LowerToHW] → HW Probes → [HW LowerXMR] → SV XMRs → Verilog
```

## Key Components

### 1. New HW Types
- `!hw.probe<T>` - Read-only reference type
- `!hw.rwprobe<T>` - Read-write (forceable) reference type

### 2. New HW Operations

**Core Operations:**
- `hw.probe.send` - Create a probe reference
- `hw.probe.resolve` - Read from a probe reference  
- `hw.probe.rwprobe` - Create RW probe via inner symbol
- `hw.probe.sub` - Index into aggregate probe
- `hw.probe.cast` - Cast between compatible probe types

**Force/Release Operations:**
- `hw.probe.force` - Force a value onto RW probe target
- `hw.probe.force_initial` - Continuous force
- `hw.probe.release` - Release a forced target
- `hw.probe.release_initial` - Release initial force

**Note:** There is no intermediate HW XMR operation. The HW LowerXMR pass directly generates `sv.xmr.ref` operations.

### 3. New HW Pass
- `hw-lower-xmr` - Converts HW probe operations to `sv.xmr.ref` operations

## Benefits

1. **Modularity**: HW dialect becomes self-contained for hardware references
2. **Reusability**: Other dialects can leverage probe semantics
3. **Clarity**: Clean separation between frontend (FIRRTL) and hardware (HW) concerns
4. **Maintainability**: XMR logic centralized in one place

## Implementation Phases

### Phase 1: Core Infrastructure (Weeks 1-2)
- Define `hw::ProbeType` and `hw::RWProbeType` in HWTypes.td
- Implement basic probe operations in HWProbeOps.td
- Add type conversion logic

### Phase 2: LowerToHW Updates (Weeks 3-4)
- Add FIRRTL → HW probe operation conversions
- Update module port handling for probe types
- Comprehensive testing

### Phase 3: HW LowerXMR Pass (Weeks 5-6)
- Port dataflow analysis from FIRRTL LowerXMR
- Implement hierarchical path construction
- Generate `sv.xmr.ref` operations

### Phase 4: Integration & Testing (Weeks 7-9)
- End-to-end testing
- Regression testing with existing FIRRTL tests
- Documentation updates
- Code review

## Compatibility

- **Backward Compatible**: Existing FIRRTL code continues to work
- **Verilog Output**: Identical output after full pipeline
- **Incremental Adoption**: Can be developed and tested in phases

## Example Transformation

**FIRRTL Input:**
```mlir
firrtl.module @Child(out %probe_out: !firrtl.probe<uint<32>>) {
  %wire = firrtl.wire : !firrtl.uint<32>
  %ref = firrtl.ref.send %wire : !firrtl.uint<32>
  firrtl.ref.define %probe_out, %ref : !firrtl.probe<uint<32>>
}
```

**After LowerToHW:**
```mlir
hw.module @Child(out probe_out: !hw.probe<i32>) {
  %wire = hw.wire : i32
  %ref = hw.probe.send %wire : i32
  hw.output %ref : !hw.probe<i32>
}
```

**After HW LowerXMR:**
```mlir
hw.hierpath @xmr_path [@Parent::@child, @Child::@wire_sym]

hw.module @Child() {
  %wire = hw.wire sym @wire_sym : i32
}

hw.module @Parent() {
  hw.instance "child" sym @child @Child() -> ()
  %xmr = sv.xmr.ref @xmr_path : !hw.inout<i32>
  %value = sv.read_inout %xmr : !hw.inout<i32>
}
```

## Files to Create

1. `include/circt/Dialect/HW/HWProbeOps.td`
2. `lib/Dialect/HW/HWProbeOps.cpp`
3. `lib/Dialect/HW/Transforms/LowerXMR.cpp`
4. `include/circt/Dialect/HW/Transforms/Passes.td` (additions)
5. `test/Dialect/HW/probe-ops.mlir`
6. `test/Dialect/HW/lower-xmr.mlir`

## Files to Modify

1. `include/circt/Dialect/HW/HWTypes.td` (add probe types)
2. `lib/Dialect/HW/HWTypes.cpp` (implement probe types)
3. `include/circt/Dialect/HW/HWOps.td` (include probe ops)
4. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp` (add conversions)
5. `include/circt/Conversion/Passes.td` (update dependencies)

## Next Steps

1. Review and approve this proposal
2. Create GitHub issue for tracking
3. Begin Phase 1 implementation
4. Set up regular check-ins for progress updates

## References

- See `hw_probe_ops_proposal.md` for detailed technical specification
- See generated Mermaid diagrams for visual architecture
- FIRRTL LowerXMR: `lib/Dialect/FIRRTL/Transforms/LowerXMR.cpp`
- SV XMR operations: `include/circt/Dialect/SV/SVInOutOps.td`
