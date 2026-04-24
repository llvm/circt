//===- HWProbeOps.td - HW probe operations ----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines HW dialect probe operations for cross-module references.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWPROBEOPS_TD
#define CIRCT_DIALECT_HW_HWPROBEOPS_TD

include "circt/Dialect/HW/HWDialect.td"
include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/HW/HWAttributesNaming.td"
include "circt/Dialect/HW/HWOpInterfaces.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinTypes.td"

//===----------------------------------------------------------------------===//
// Probe Send/Resolve Operations
//===----------------------------------------------------------------------===//

def ProbeSendOp : HWOp<"probe.send", [Pure]> {
  let summary = "Create a probe reference";
  let description = [{
    Creates a probe reference to a value. This marks the source of a dataflow
    path that can be observed remotely. The probe can cross module boundaries
    and will be lowered to an XMR (cross-module reference) during lowering.

    By default, this operation produces a read-only probe (!hw.probe<T>).
    When the 'forceable' attribute is present, it produces a read-write probe
    (!hw.rwprobe<T>) that can be used with force and release operations.

    Example:
    ```mlir
    %wire = hw.wire %value : i32

    // Read-only probe (default)
    %ref = hw.probe.send %wire : i32

    // Read-write probe (forceable)
    %rwref = hw.probe.send forceable %wire : i32
    ```
  }];

  let arguments = (ins AnyType:$input,
                       UnitAttr:$forceable);
  let results = (outs AnyProbeType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;

  let builders = [
    // Default builder without forceable attribute
    OpBuilder<(ins "Value":$input), [{
      Type probeType = ProbeType::get(input.getType());
      build($_builder, $_state, probeType, input, /*forceable=*/UnitAttr());
    }]>,
    // Builder with explicit forceable control
    OpBuilder<(ins "Value":$input, "bool":$forceable), [{
      Type probeType = forceable ?
        (Type)RWProbeType::get(input.getType()) :
        (Type)ProbeType::get(input.getType());
      UnitAttr forceableAttr = forceable ?
        $_builder.getUnitAttr() : UnitAttr();
      build($_builder, $_state, probeType, input, forceableAttr);
    }]>
  ];
}

def ProbeResolveOp : HWOp<"probe.resolve", [Pure]> {
  let summary = "Read the value from a probe reference";
  let description = [{
    Resolves a probe reference to read its value. This is where an XMR will be
    emitted during lowering. The probe reference must eventually trace back to
    a probe.send operation.
    
    Example:
    ```mlir
    %value = hw.probe.resolve %ref : !hw.probe<i32>
    ```
  }];
  
  let arguments = (ins AnyProbeType:$ref);
  let results = (outs AnyType:$result);

  let hasCustomAssemblyFormat = 1;
}

//===----------------------------------------------------------------------===//
// RWProbe Operations
//===----------------------------------------------------------------------===//

def ProbeRWProbeOp : HWOp<"probe.rwprobe", [
    DeclareOpInterfaceMethods<InnerRefUserOpInterface>,
    Pure
  ]> {
  let summary = "Create a read-write probe reference via inner symbol";
  let description = [{
    Creates an RWProbe reference to a local target specified by an InnerRef.
    This allows the target to be forced and released procedurally. The target
    must be within the same module (local reference).

    Example:
    ```mlir
    %rwref = hw.probe.rwprobe @Module::@symbol : !hw.rwprobe<i32>
    ```
  }];

  let arguments = (ins InnerRefAttr:$target);
  let results = (outs RWProbeType:$result);

  let hasCustomAssemblyFormat = 1;
}

def ProbeDefineOp : HWOp<"probe.define", [SameTypeOperands]> {
  let summary = "Define a probe reference to another probe";
  let description = [{
    Defines a target probe reference to be the same as the source probe reference.
    This is used to statically route probe references from source to destination
    through the design hierarchy, one module at a time.

    Similar to a "connect" operation but with stricter semantics:
    - Cannot have multiple defines to the same destination
    - The define is never conditional, even if under a conditional block
    - Both source and destination must be statically resolvable references

    This operation is particularly useful for propagating probe references through
    module boundaries, allowing hierarchical probe routing.

    Example:
    ```mlir
    hw.probe.define %dest, %src : !hw.probe<i32>
    ```
  }];

  let arguments = (ins AnyProbeType:$dest, AnyProbeType:$src);
  let results = (outs);

  let assemblyFormat = [{
    $dest `,` $src attr-dict `:` qualified(type($dest))
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Probe Manipulation Operations
//===----------------------------------------------------------------------===//

def ProbeSubOp : HWOp<"probe.sub", [Pure]> {
  let summary = "Extract subelement from aggregate probe";
  let description = [{
    Statically indexes into an aggregate probe type (array or struct) to
    extract a reference to a subelement. The index must be a compile-time
    constant.

    Example:
    ```mlir
    %arr_ref = hw.probe.send %array : !hw.array<4xi32>
    %elem_ref = hw.probe.sub %arr_ref[2] : !hw.probe<!hw.array<4xi32>>
    ```
  }];

  let arguments = (ins AnyProbeType:$input, I32Attr:$index);
  let results = (outs AnyProbeType:$result);

  let hasCustomAssemblyFormat = 1;
}

def ProbeCastOp : HWOp<"probe.cast", [Pure]> {
  let summary = "Cast between compatible probe types";
  let description = [{
    Losslessly cast between compatible probe types. The most common use is
    demoting an RWProbe to a read-only Probe.

    Example:
    ```mlir
    %probe = hw.probe.cast %rwprobe : !hw.rwprobe<i32> -> !hw.probe<i32>
    ```
  }];

  let arguments = (ins AnyProbeType:$input);
  let results = (outs AnyProbeType:$result);

  let hasCustomAssemblyFormat = 1;
  let hasFolder = 1;
  let hasCanonicalizer = 1;
}

//===----------------------------------------------------------------------===//
// XMR Operations
//===----------------------------------------------------------------------===//

def ProbeXMRRefOp : HWOp<"probe.xmr_ref", [Pure]> {
  let summary = "Convert a probe reference to an inout reference";
  let description = [{
    Converts a probe or rwprobe reference to an inout reference. This operation
    takes a probe reference as input and returns the corresponding inout type,
    allowing the probe to be used with operations that work on inout types.

    This is typically used after probe resolution has occurred, where the probe
    reference needs to be converted to an inout for use with force/release or
    other procedural operations.

    The result type is automatically inferred from the input probe's inner type.

    Example:
    ```mlir
    %probe = hw.probe.send %wire : i32
    %inout = hw.probe.xmr_ref %probe : !hw.probe<i32>
    %value = sv.read_inout %inout : !hw.inout<i32>
    ```

    Example with rwprobe for force operations:
    ```mlir
    %rwprobe = hw.probe.send forceable %wire : i32
    %inout = hw.probe.xmr_ref %rwprobe : !hw.rwprobe<i32>
    sv.force %inout, %newval : i32
    ```
  }];

  let arguments = (ins AnyProbeType:$ref);
  let results = (outs InOutType:$result);

  let hasCustomAssemblyFormat = 1;

  let builders = [
    OpBuilder<(ins "Value":$ref), [{
      Type innerType;
      if (auto probeType = dyn_cast<ProbeType>(ref.getType()))
        innerType = probeType.getInnerType();
      else if (auto rwProbeType = dyn_cast<RWProbeType>(ref.getType()))
        innerType = rwProbeType.getInnerType();
      else
        llvm_unreachable("ref must be a probe or rwprobe type");

      Type resultType = InOutType::get(innerType);
      build($_builder, $_state, resultType, ref);
    }]>
  ];

  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_HW_HWPROBEOPS_TD
