//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PROBE_PROBEOPS_TD
#define CIRCT_DIALECT_PROBE_PROBEOPS_TD

include "circt/Dialect/Probe/ProbeDialect.td"
include "circt/Dialect/Probe/ProbeTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class ProbeOp<string mnemonic, list<Trait> traits = []> :
    Op<ProbeDialect, mnemonic, traits>;

def SendOp : ProbeOp<"send", [
  Pure,
  HasParent<"hw::HWModuleOp">,
  TypesMatchWith<"input and ref types match",
    "input", "ref", "::circt::probe::RefType::get($_self)">
]> {
  let summary = "create a read-only probe handle for a hardware value";
  let description = [{
    `probe.send` creates a read-only probe handle for an arbitrary SSA hardware
    value. The `ref` result represents an observation of that value and does not
    represent ordinary hardware dataflow. Users that also consume the input
    value should use the original SSA value directly.

    The observed value does not need to have an explicit name in the IR. A
    backend may generate a name when required. Automatically generated names
    are implementation details and are not guaranteed to remain stable across
    compiler runs or IR transformations.
  }];
  let arguments = (ins ProbeElementType:$input);
  let results = (outs RefTypeConstraint:$ref);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReadOp : ProbeOp<"read", [
  TypesMatchWith<"input and result types match",
    "input", "result",
    "cast<::circt::probe::RefType>($_self).getElementType()">,
  Pure,
  HasParent<"hw::HWModuleOp">
]> {
  let summary = "read the value observed by a probe handle";
  let description = [{
    `probe.read` returns the value observed through its input probe handle. It
    does not require the compiler to materialize ordinary hardware dataflow
    between the origin of the probe and the read site.

    The result is the hardware value observed through the read-only probe
    handle.
  }];
  let arguments = (ins RefTypeConstraint:$input);
  let results = (outs ProbeElementType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let hasFolder = true;
}

#endif // CIRCT_DIALECT_PROBE_PROBEOPS_TD
