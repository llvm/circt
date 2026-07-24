//===- ProbeOps.td - Probe dialect operations --------------*- tablegen -*-===//
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
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class ProbeOp<string mnemonic, list<Trait> traits = []> :
    Op<ProbeDialect, mnemonic, traits>;

class ProbeInferTypeOp<string mnemonic, list<Trait> traits = []> :
    ProbeOp<mnemonic, traits # [
      Pure, DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>
    ]>;

def SendOp : ProbeOp<"send", [
  TypesMatchWith<"input and forwarded types match",
    "input", "forwarded", "$_self">,
  TypesMatchWith<"input and ref types match",
    "input", "ref", "::circt::probe::RefType::get($_self)">
]> {
  let summary = "create a read-only probe handle for a hardware value";
  let description = [{
    `probe.send` creates a read-only probe handle for an arbitrary SSA hardware
    value. The `forwarded` result carries the input value through an explicit
    tap, while the `ref` result identifies the value for by-name observation.

    Users of `forwarded` must continue to observe the value through this
    operation; optimizations must not bypass the tap and replace those uses with
    the input. Leaving `forwarded` unused expresses a probe-only tap. A backend
    that requires the observed value to be nameable may materialize an anchor
    for this operation.
  }];
  let arguments = (ins ProbeElementType:$input);
  let results = (outs ProbeElementType:$forwarded,
                      RefTypeConstraint:$ref);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReadOp : ProbeInferTypeOp<"read", [
  TypesMatchWith<"input and result types match",
    "input", "result",
    "cast<::circt::probe::RefType>($_self).getElementType()">
]> {
  let summary = "read the value observed by a probe handle";
  let description = [{
    `probe.read` provides a by-name, read-only observation of the value referred
    to by its input. It does not require the compiler to materialize ordinary
    hardware dataflow between the origin of the probe and the read site.

    The result is the value of the observed hardware, not writable access to the
    hardware object itself. It cannot be used for assignment, force, or release.
  }];
  let arguments = (ins RefTypeConstraint:$input);
  let results = (outs ProbeElementType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // CIRCT_DIALECT_PROBE_PROBEOPS_TD
