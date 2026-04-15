# HW Probe Operations - Quick Reference Card

## Types

| Type | Syntax | Description | Can Force? |
|------|--------|-------------|------------|
| ProbeType | `!hw.probe<T>` | Read-only reference | No |
| RWProbeType | `!hw.rwprobe<T>` | Read-write reference | Yes |

## Operations

### Core Probe Operations

| Operation | Syntax | Purpose | Lowered To |
|-----------|--------|---------|------------|
| `hw.probe.send` | `%ref = hw.probe.send %value : T` | Create probe reference | Removed (tracked in dataflow) |
| `hw.probe.resolve` | `%val = hw.probe.resolve %ref : !hw.probe<T>` | Read from probe | `sv.xmr.ref` + `sv.read_inout` |
| `hw.probe.rwprobe` | `%ref = hw.probe.rwprobe @Mod::@sym : !hw.rwprobe<T>` | Create RW probe via symbol | Removed (tracked in dataflow) |
| `hw.probe.sub` | `%elem = hw.probe.sub %ref[N] : !hw.probe<aggregate>` | Index into aggregate | Path suffix in XMR |
| `hw.probe.cast` | `%p = hw.probe.cast %rwp : !hw.rwprobe<T> -> !hw.probe<T>` | Cast probe types | Removed (dataflow) |

### Force/Release Operations

| Operation | Syntax | Purpose | Lowered To |
|-----------|--------|---------|------------|
| `hw.probe.force` | `hw.probe.force %clk, %pred, %ref, %val` | Force with clock | `sv.xmr.ref` + `sv.always` + `sv.force` |
| `hw.probe.force_initial` | `hw.probe.force_initial %pred, %ref, %val` | Continuous force | `sv.xmr.ref` + `sv.initial` + `sv.force` |
| `hw.probe.release` | `hw.probe.release %clk, %pred, %ref` | Release with clock | `sv.xmr.ref` + `sv.always` + `sv.release` |
| `hw.probe.release_initial` | `hw.probe.release_initial %pred, %ref` | Release initial | `sv.xmr.ref` + `sv.initial` + `sv.release` |

## FIRRTL to HW Mapping

| FIRRTL Operation | HW Operation | Notes |
|------------------|--------------|-------|
| `firrtl.ref.send` | `hw.probe.send` | Direct 1:1 |
| `firrtl.ref.resolve` | `hw.probe.resolve` | Direct 1:1 |
| `firrtl.ref.rwprobe` | `hw.probe.rwprobe` | Direct 1:1 |
| `firrtl.ref.sub` | `hw.probe.sub` | Direct 1:1 |
| `firrtl.ref.cast` | `hw.probe.cast` | Direct 1:1 |
| `firrtl.ref.define` | Module ports | Dataflow through ports |
| `firrtl.ref.force` | `hw.probe.force` | Direct 1:1 |
| `firrtl.ref.force_initial` | `hw.probe.force_initial` | Direct 1:1 |
| `firrtl.ref.release` | `hw.probe.release` | Direct 1:1 |
| `firrtl.ref.release_initial` | `hw.probe.release_initial` | Direct 1:1 |
| `firrtl::RefType` | `!hw.probe<T>` | Direct type mapping |
| `firrtl::RWProbe` | `!hw.rwprobe<T>` | Direct type mapping |

**Important:** `firrtl.xmr.ref` and `firrtl.xmr.deref` are NOT mapped to HW. They are converted directly to `sv.xmr.ref` during LowerToHW or removed during FIRRTL's LowerXMR pass.

## Type Conversions

### Valid Casts
```mlir
// RWProbe → Probe (demotion)
%probe = hw.probe.cast %rwprobe : !hw.rwprobe<i32> -> !hw.probe<i32>

// Width-compatible casts
%wider = hw.probe.cast %narrow : !hw.probe<i8> -> !hw.probe<i32>
```

### Invalid Casts
```mlir
// ❌ Cannot promote Probe → RWProbe
%rwprobe = hw.probe.cast %probe : !hw.probe<i32> -> !hw.rwprobe<i32>
```

## Transformation Pipeline

```
┌─────────────┐
│   FIRRTL    │  firrtl.ref.send, firrtl.ref.resolve, etc.
│   Probes    │  firrtl::RefType, firrtl::RWProbe
└─────┬───────┘
      │
      │ LowerToHW Pass
      ↓
┌─────────────┐
│ HW Probes   │  hw.probe.send, hw.probe.resolve, etc.
│             │  !hw.probe<T>, !hw.rwprobe<T>
└─────┬───────┘
      │
      │ HW LowerXMR Pass (NEW)
      ↓
┌─────────────┐
│  SV XMRs    │  sv.xmr.ref, sv.force, sv.release
│             │  hw.hierpath
└─────┬───────┘
      │
      │ ExportVerilog
      ↓
┌─────────────┐
│  Verilog    │  module.instance.signal
│   XMRs      │  force/release statements
└─────────────┘
```

## Key Design Principles

1. **No Intermediate HW XMR Op**: HW LowerXMR directly generates `sv.xmr.ref`
2. **Probe Ports Removed**: After LowerXMR, all probe ports are eliminated
3. **Dataflow Analysis**: Pass tracks probe dataflow using union-find
4. **Hierarchical Paths**: Built incrementally as references cross modules
5. **Symbol Management**: Symbols added during XMR resolution

## Common Patterns

### Pattern 1: Simple Cross-Module Reference
```mlir
// Child module creates probe
%ref = hw.probe.send %signal : T
// Parent module reads via probe
%value = hw.probe.resolve %ref : !hw.probe<T>
// Result: XMR in parent to child's signal
```

### Pattern 2: Forceable Register
```mlir
// Create forceable reference
%rwref = hw.probe.rwprobe @Mod::@reg : !hw.rwprobe<T>
// Force the register
hw.probe.force_initial %true, %rwref, %override_val
// Result: initial force statement in Verilog
```

### Pattern 3: Aggregate Indexing
```mlir
// Send array reference
%arr_ref = hw.probe.send %array : !hw.array<4xT>
// Index into array
%elem_ref = hw.probe.sub %arr_ref[2]
// Resolve element
%value = hw.probe.resolve %elem_ref
// Result: XMR with array index path.to.array[2]
```

## Testing Checklist

- [ ] Type parsing and printing
- [ ] Operation verification
- [ ] FIRRTL → HW conversion
- [ ] HW → SV XMR generation
- [ ] Dataflow analysis correctness
- [ ] Hierarchical path construction
- [ ] Force/release lowering
- [ ] Aggregate probe handling
- [ ] Multi-level hierarchy
- [ ] Verilog emission correctness
