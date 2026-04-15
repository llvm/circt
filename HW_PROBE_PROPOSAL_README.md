# HW Probe Operations Proposal - Documentation Guide

This directory contains a comprehensive proposal for creating HW dialect equivalents of FIRRTL probe operations, enabling the LowerXMR pass to be moved from the FIRRTL dialect to the HW dialect.

## 📋 Document Overview

### 1. **hw_probe_ops_summary.md** ⭐ START HERE
   - **Purpose**: Quick overview and executive summary
   - **Audience**: Decision makers, reviewers, anyone wanting a high-level understanding
   - **Length**: ~5 minute read
   - **Contains**:
     - Problem statement
     - Proposed solution overview
     - Key benefits
     - Example transformations
     - Implementation timeline

### 2. **hw_probe_ops_proposal.md** 📖 DETAILED SPEC
   - **Purpose**: Complete technical specification
   - **Audience**: Implementers, detailed reviewers
   - **Length**: ~20 minute read
   - **Contains**:
     - Detailed operation definitions (TableGen specs)
     - Type system design
     - Dataflow analysis algorithms
     - Implementation strategy
     - Testing approach
     - Open questions and alternatives

### 3. **Mermaid Diagrams** 🎨 VISUAL AIDS
   Three interactive diagrams were generated showing:
   
   **a) HW Probe Operations Architecture**
   - Shows flow from FIRRTL → HW → SV → Verilog
   - Illustrates pass boundaries
   - Color-coded by dialect
   
   **b) FIRRTL to HW Operation Mapping**
   - One-to-one mapping between FIRRTL and HW probe ops
   - Highlights which operations have direct equivalents
   
   **c) HW Probe Type Hierarchy**
   - Class diagram showing type relationships
   - Shows what types can be contained in probes
   - Documents constraints (e.g., forceable types)

## 🎯 Quick Start Guide

### For Decision Makers
1. Read **hw_probe_ops_summary.md** (5 min)
2. Review the architecture diagram
3. Look at example transformations in summary

### For Implementers
1. Skim **hw_probe_ops_summary.md** (5 min)
2. Read **hw_probe_ops_proposal.md** thoroughly (20 min)
3. Study operation mapping diagram
4. Review type hierarchy diagram
5. Reference existing code:
   - `lib/Dialect/FIRRTL/Transforms/LowerXMR.cpp` - Current implementation
   - `include/circt/Dialect/FIRRTL/FIRRTLExpressions.td` - FIRRTL probe ops
   - `include/circt/Dialect/SV/SVInOutOps.td` - SV XMR operations

### For Reviewers
1. Read **hw_probe_ops_summary.md** (5 min)
2. Read relevant sections of **hw_probe_ops_proposal.md**:
   - Section 2: Proposed HW Probe Operations
   - Section 3: Migration Strategy
   - Section 4: Detailed Design Considerations
3. Check "Open Questions" section
4. Review "Alternatives Considered"

## 🔑 Key Concepts

### Probe Types
- **Probe** (`!hw.probe<T>`): Read-only reference that can cross module boundaries
- **RWProbe** (`!hw.rwprobe<T>`): Read-write reference supporting force/release

### Core Operations
- **send**: Create a reference to a value
- **resolve**: Read the value from a reference (lowered to `sv.xmr.ref`)
- **rwprobe**: Create a forceable reference via symbol
- **sub**: Index into aggregate reference
- **cast**: Convert between compatible probe types
- **force/release**: Procedurally override values (lowered to `sv.xmr.ref` + SV force/release)

### Transformation Flow
```
FIRRTL Probes 
  ↓ [LowerToHW Pass]
HW Probes
  ↓ [HW LowerXMR Pass - NEW]
SV XMR Operations
  ↓ [ExportVerilog]
Verilog XMRs
```

## 📊 Proposal Status

- **Status**: DRAFT - Awaiting Review
- **Created**: 2026-04-15
- **Author**: Based on analysis of CIRCT codebase
- **Target**: CIRCT HW Dialect Enhancement

## 🔗 Related Resources

### Existing CIRCT Code
- FIRRTL Probe Ops: `include/circt/Dialect/FIRRTL/FIRRTLExpressions.td` (lines 1510-1680)
- FIRRTL LowerXMR: `lib/Dialect/FIRRTL/Transforms/LowerXMR.cpp`
- SV XMR Ops: `include/circt/Dialect/SV/SVInOutOps.td` (lines 114-130)
- HW Dialect: `include/circt/Dialect/HW/`

### Tests to Reference
- `test/Dialect/FIRRTL/lowerXMR.mlir`
- `test/Conversion/FIRRTLToHW/*.mlir`
- `test/Dialect/SV/basic.mlir`

## ❓ Frequently Asked Questions

**Q: Why move probes to HW dialect?**
A: To create a clean separation of concerns and make probe semantics available to other dialects without FIRRTL dependency.

**Q: Will this break existing code?**
A: No, this is backward compatible. Existing FIRRTL code continues to work, just with a different internal transformation path.

**Q: How long will implementation take?**
A: Estimated 8-9 weeks for complete implementation and testing (see timeline in summary).

**Q: What about layers?**
A: Initial implementation won't include layer support. Can be added later if needed.

**Q: Will Verilog output change?**
A: No, final Verilog output should be identical after the full pipeline.

## 📝 Feedback & Questions

For questions or feedback on this proposal:
1. Review the "Open Questions" section in the detailed proposal
2. Check if alternatives were already considered
3. Add comments/suggestions as appropriate

## 🚀 Implementation Checklist

Once approved, follow this order:
- [ ] Create types (HWTypes.td)
- [ ] Implement type parsing/printing
- [ ] Define operations (HWProbeOps.td)
- [ ] Add LowerToHW conversions
- [ ] Port LowerXMR logic to HW dialect
- [ ] Write comprehensive tests
- [ ] Update documentation
- [ ] Code review and integration
