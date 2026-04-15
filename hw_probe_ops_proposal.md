# Proposal: HW Dialect Probe Operations

## Executive Summary

This proposal outlines the design for creating HW dialect equivalents of FIRRTL probe operations to enable moving the LowerXMR pass from the FIRRTL dialect to the HW dialect. This refactoring will improve modularity and allow direct generation of `sv::XMRRefOp` from HW probe operations.

## Motivation

Currently, the FIRRTL dialect contains:
1. **Probe types and operations** (`RefType`, `RWProbe`, `RefSendOp`, `RefResolveOp`, `RefSubOp`, `XMRRefOp`, etc.)
2. **LowerXMR pass** in `lib/Dialect/FIRRTL/Transforms/LowerXMR.cpp` that converts probe operations to XMRs
3. **LowerToHW pass** in `lib/Conversion/FIRRTLToHW/LowerToHW.cpp` that already handles `XMRRefOp` → `sv::XMRRefOp` conversion

### Current Flow
```
FIRRTL Probe Ops → LowerXMR (FIRRTL) → FIRRTL::XMRRefOp → LowerToHW → sv::XMRRefOp
```

### Proposed Flow
```
FIRRTL Probe Ops → LowerToHW → HW Probe Ops → LowerXMR (HW) → sv::XMRRefOp
```

## Benefits

1. **Separation of Concerns**: HW dialect becomes the canonical representation for hardware references
2. **Reusability**: Other dialects can use HW probe operations without depending on FIRRTL
3. **Simpler FIRRTL Lowering**: Direct 1:1 mapping from FIRRTL probes to HW probes
4. **Unified XMR Handling**: Single pass in HW dialect handles all XMR generation

## Proposed HW Probe Operations

### 1. Types

#### `hw::ProbeType`
```tablegen
def HWProbeType : DialectType<HWDialect,
    CPred<"::circt::hw::isa<circt::hw::ProbeType>($_self)">,
    "a probe type", "::circt::hw::ProbeType"> {
  let summary = "A non-forceable reference to a hardware value";
  let description = [{
    Represents a read-only reference that can cross module boundaries.
    Syntax: `!hw.probe<T>` where T is the inner type.
  }];
}
```

#### `hw::RWProbeType`
```tablegen
def HWRWProbeType : DialectType<HWDialect,
    CPred<"::circt::hw::isa<circt::hw::RWProbeType>($_self)">,
    "a forceable probe type", "::circt::hw::RWProbeType"> {
  let summary = "A read-write (forceable) reference to a hardware value";
  let description = [{
    Represents a read-write reference that can be forced/released.
    Syntax: `!hw.rwprobe<T>` where T is the inner type.
  }];
}
```

**Type Structure**:
```cpp
class ProbeType : public Type::TypeBase<ProbeType, Type, ProbeTypeStorage> {
  Type innerType;  // The type being referenced
};

class RWProbeType : public Type::TypeBase<RWProbeType, Type, RWProbeTypeStorage> {
  Type innerType;  // The type being referenced (must be forceable)
};
```

### 2. Operations

#### `hw.probe.send`
```tablegen
def ProbeSendOp : HWOp<"probe.send", [Pure]> {
  let summary = "Create a read-only probe reference";
  let description = [{
    Creates a probe reference to a value. This marks the source of a dataflow path.
    Example: %ref = hw.probe.send %wire : i32
  }];
  let arguments = (ins HWValueType:$input);
  let results = (outs HWProbeType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

#### `hw.probe.rwprobe`
```tablegen
def ProbeRWProbeOp : HWOp<"probe.rwprobe", [
    DeclareOpInterfaceMethods<InnerRefUserOpInterface>,
    Pure
  ]> {
  let summary = "Create a read-write probe reference via inner symbol";
  let description = [{
    Creates an RWProbe reference to a local target specified by an InnerRef.
    This allows forcing/releasing the target.
    Example: %rwref = hw.probe.rwprobe @Module::@symbol : !hw.rwprobe<i32>
  }];
  let arguments = (ins InnerRefAttr:$target);
  let results = (outs HWRWProbeType:$result);
  let assemblyFormat = "$target attr-dict `:` type($result)";
}
```

#### `hw.probe.resolve`
```tablegen
def ProbeResolveOp : HWOp<"probe.resolve", [Pure]> {
  let summary = "Read the value from a probe reference";
  let description = [{
    Resolves a probe reference to read its value. This is where an XMR may be emitted.
    Example: %value = hw.probe.resolve %ref : !hw.probe<i32>
  }];
  let arguments = (ins HWProbeType:$ref);
  let results = (outs HWValueType:$result);
  let assemblyFormat = "$ref attr-dict `:` qualified(type($ref))";
}
```

#### `hw.probe.sub`
```tablegen
def ProbeSubOp : HWOp<"probe.sub", [Pure]> {
  let summary = "Extract subelement from aggregate probe";
  let description = [{
    Statically indexes into an aggregate probe type (array or struct).
    Example: %elem_ref = hw.probe.sub %array_ref[0] : !hw.probe<!hw.array<4xi32>>
  }];
  let arguments = (ins HWProbeType:$input, I32Attr:$index);
  let results = (outs HWProbeType:$result);
  let assemblyFormat = "$input `[` $index `]` attr-dict `:` qualified(type($input))";
}
```

#### `hw.probe.cast`
```tablegen
def ProbeCastOp : HWOp<"probe.cast", [Pure]> {
  let summary = "Cast between compatible probe types";
  let description = [{
    Losslessly cast between compatible probe types (e.g., RWProbe to Probe).
    Example: %probe = hw.probe.cast %rwprobe : !hw.rwprobe<i32> -> !hw.probe<i32>
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($result)";
}
```

### 3. Force/Release Operations

#### `hw.probe.force`
```tablegen
def ProbeForceOp : HWOp<"probe.force"> {
  let summary = "Force an RWProbe to a value";
  let description = [{
    Forces an RWProbe target to the specified value with clock and predicate.
    Example: hw.probe.force %clock, %pred, %rwref, %value : ...
  }];
  let arguments = (ins I1Type:$clock, I1Type:$predicate,
                       HWRWProbeType:$dest, HWValueType:$src);
  let assemblyFormat = [{
    $clock `,` $predicate `,` $dest `,` $src attr-dict
    `:` type($clock) `,` type($predicate) `,` qualified(type($dest)) `,` type($src)
  }];
}
```

#### `hw.probe.force_initial`
```tablegen
def ProbeForceInitialOp : HWOp<"probe.force_initial"> {
  let summary = "Continuously force an RWProbe";
  let description = [{
    Forces an RWProbe target continuously (initial block).
    Example: hw.probe.force_initial %pred, %rwref, %value : ...
  }];
  let arguments = (ins I1Type:$predicate, HWRWProbeType:$dest, HWValueType:$src);
  let assemblyFormat = [{
    $predicate `,` $dest `,` $src attr-dict
    `:` type($predicate) `,` qualified(type($dest)) `,` type($src)
  }];
}
```

#### `hw.probe.release` and `hw.probe.release_initial`
Similar structure to force operations but without the source value.

## Migration Strategy

### Phase 1: Add HW Probe Operations
1. Define types in `include/circt/Dialect/HW/HWTypes.td`
2. Define operations in new file `include/circt/Dialect/HW/HWProbeOps.td`
3. Implement type definitions in `lib/Dialect/HW/HWTypes.cpp`
4. Implement operations in new file `lib/Dialect/HW/HWProbeOps.cpp`
5. Add to `HWOps.td` include list

### Phase 2: Update LowerToHW Pass
Modify `lib/Conversion/FIRRTLToHW/LowerToHW.cpp` to map FIRRTL probe ops to HW:
- `firrtl::RefType` → `hw::ProbeType`
- `firrtl::RWProbe` → `hw::RWProbeType`
- `firrtl::RefSendOp` → `hw::ProbeSendOp`
- `firrtl::RefResolveOp` → `hw::ProbeResolveOp`
- `firrtl::RefSubOp` → `hw::ProbeSubOp`
- `firrtl::RWProbeOp` → `hw::ProbeRWProbeOp`
- `firrtl::RefForceOp` → `hw::ProbeForceOp`
- `firrtl::RefForceInitialOp` → `hw::ProbeForceInitialOp`
- `firrtl::RefReleaseOp` → `hw::ProbeReleaseOp`
- `firrtl::RefReleaseInitialOp` → `hw::ProbeReleaseInitialOp`
- `firrtl::RefCastOp` → `hw::ProbeCastOp`

Note: `firrtl::XMRRefOp` is NOT lowered to HW. It will be removed during FIRRTL LowerXMR or directly converted to `sv::XMRRefOp` during LowerToHW.

### Phase 3: Create HW LowerXMR Pass
1. Create new pass `lib/Dialect/HW/Transforms/LowerXMR.cpp`
2. Port logic from FIRRTL LowerXMR pass
3. Key changes:
   - Operate on `hw::ProbeType` instead of `firrtl::RefType`
   - Generate `sv::XMRRefOp` directly (no intermediate HW XMR op)
   - Handle HW-specific constructs
   - Replace `hw.probe.resolve` with `sv.xmr.ref` + `sv.read_inout`
   - Replace force/release operations with `sv.xmr.ref` + procedural SV ops

### Phase 4: Update FIRRTL LowerXMR
1. Simplify or remove FIRRTL LowerXMR pass
2. Update documentation and tests
3. Update pass pipeline in firtool

## Detailed Design Considerations

### Type System

**Inner Type Constraints**:
- `hw::ProbeType` can contain any `HWValueType`
- `hw::RWProbeType` has stricter requirements (must be forceable)
- Both types should support nested aggregates (arrays, structs)

**Type Compatibility**:
- `hw::RWProbeType<T>` can be cast to `hw::ProbeType<T>` (RW → RO demotion)
- Width-compatible casts should be supported
- Type aliases should be handled transparently

### Dataflow Analysis in HW LowerXMR

The HW LowerXMR pass will need to implement:

1. **Reaching Definitions Analysis**:
   - Track probe dataflow from `hw.probe.send` to `hw.probe.resolve`
   - Build dataflow classes using union-find
   - Propagate across module boundaries via ports

2. **Hierarchical Path Construction**:
   - Build `hw.hierpath` operations for each XMR
   - Cache paths to avoid duplication
   - Handle verbatim suffixes for external modules

3. **XMR Generation**:
   - Replace `hw.probe.resolve` with `sv.xmr.ref` + `sv.read_inout`
   - Handle force/release operations by resolving their probe arguments to `sv.xmr.ref`
   - Generate appropriate symbols and inner refs
   - Create procedural SV blocks for force/release operations

### Port Handling

**Module Ports**:
```mlir
hw.module @Child(out %probe_out: !hw.probe<i32>) {
  %wire = hw.wire : i32
  %ref = hw.probe.send %wire : i32
  hw.output %ref : !hw.probe<i32>
}

hw.module @Parent() {
  %probe = hw.instance "child" @Child() -> (probe_out: !hw.probe<i32>)
  %value = hw.probe.resolve %probe : !hw.probe<i32>
}
```

After LowerXMR:
```mlir
hw.hierpath @xmr_path [@Parent::@child, @Child::@wire_sym]

hw.module @Child() {
  %wire = hw.wire sym @wire_sym : i32
  // probe port removed
}

hw.module @Parent() {
  hw.instance "child" sym @child @Child() -> ()
  %xmr = sv.xmr.ref @xmr_path : !hw.inout<i32>
  %value = sv.read_inout %xmr : !hw.inout<i32>
}
```

### Force/Release Lowering

RWProbe force operations are lowered to SV procedural blocks:

Before:
```mlir
hw.probe.force_initial %pred, %rwprobe, %value :
  i1, !hw.rwprobe<i32>, i32
```

After:
```mlir
%xmr = sv.xmr.ref @hierpath : !hw.inout<i32>
sv.initial {
  sv.if %pred {
    sv.force %xmr, %value : i32
  }
}
```

### Interaction with Existing Passes

**Before HW LowerXMR**:
- `hw.module` operations with probe ports
- Probe operations throughout the design
- No XMR dependencies yet

**After HW LowerXMR**:
- Probe ports removed from modules
- Instance operations updated (probe results removed)
- `sv.xmr.ref` operations generated
- `hw.hierpath` operations created
- Force/release converted to SV procedural code

**Downstream Passes**:
- ExportVerilog already handles `sv.xmr.ref`
- No changes needed to emission

## Implementation Files

### New Files
1. `include/circt/Dialect/HW/HWProbeOps.td` - Operation definitions
2. `lib/Dialect/HW/HWProbeOps.cpp` - Operation implementations
3. `lib/Dialect/HW/Transforms/LowerXMR.cpp` - XMR lowering pass
4. `include/circt/Dialect/HW/Transforms/Passes.td` - Pass definition
5. `test/Dialect/HW/probe-ops.mlir` - Operation tests
6. `test/Dialect/HW/lower-xmr.mlir` - Pass tests

### Modified Files
1. `include/circt/Dialect/HW/HWTypes.td` - Add probe types
2. `lib/Dialect/HW/HWTypes.cpp` - Implement probe types
3. `include/circt/Dialect/HW/HWOps.td` - Include HWProbeOps.td
4. `lib/Conversion/FIRRTLToHW/LowerToHW.cpp` - Add probe op conversions
5. `include/circt/Conversion/Passes.td` - Update pass dependencies

## Testing Strategy

### Unit Tests
1. **Type Tests**: Verify probe type construction, printing, parsing
2. **Operation Tests**: Test each operation's assembly format and verification
3. **Canonicalization**: Test probe.cast folding, dead probe elimination

### Integration Tests
1. **Lowering Tests**: FIRRTL probes → HW probes
2. **XMR Tests**: HW probes → SV XMRs
3. **End-to-End**: FIRRTL → HW → SV → Verilog emission

### Regression Tests
- Convert existing FIRRTL XMR tests to use new flow
- Ensure identical Verilog output

## Open Questions

1. **Layer Support**: Should `hw::ProbeType` support layer attributes like FIRRTL RefType?
   - **Recommendation**: Start without layers, add later if needed

2. **Namespace**: Should operations be `hw.probe.*` or just `hw.*`?
   - **Recommendation**: `hw.probe.*` for clarity and namespace separation

3. **Type Aliases**: How should probe types interact with HW type aliases?
   - **Recommendation**: Support transparent traversal like other HW types

4. **Zero-Width Handling**: How to handle zero-width probes?
   - **Recommendation**: Remove during lowering, same as FIRRTL

5. **Memory Probes**: Should memory debug ports be supported?
   - **Recommendation**: Yes, port existing FIRRTL MemOp probe support

## Alternatives Considered

### Alternative 1: Keep XMR Lowering in FIRRTL
**Pros**: No changes needed, already works
**Cons**: HW dialect depends on FIRRTL for XMR semantics, poor modularity

### Alternative 2: Lower Directly to SV
**Pros**: Fewer intermediate operations
**Cons**: Loses ability to optimize/analyze at HW level, harder to debug

### Alternative 3: Use InOut Types
**Pros**: Reuses existing types
**Cons**: Probes have different semantics (can cross hierarchy), would be confusing

## Timeline

1. **Week 1-2**: Implement types and basic operations
2. **Week 3-4**: Update LowerToHW pass, add tests
3. **Week 5-6**: Implement HW LowerXMR pass
4. **Week 7-8**: Testing, debugging, documentation
5. **Week 9**: Code review and integration

## Conclusion

This proposal provides a clean separation of concerns by moving probe semantics to the HW dialect where they belong. The phased approach allows for incremental development and testing while maintaining backward compatibility until the transition is complete.

The key benefits are:
- **Modularity**: HW dialect is self-contained for hardware references
- **Reusability**: Other frontends can use HW probes without FIRRTL
- **Maintainability**: Clear separation between FIRRTL-specific and HW-level concerns
- **Consistency**: All XMR handling in one place (HW dialect)
