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
  Pure,
  TypesMatchWith<"input and result types match",
    "input", "result", "::circt::probe::RefType::get($_self)">
]> {
  let summary = "create a read-only probe handle for a hardware value";
  let arguments = (ins ProbeElementType:$input);
  let results = (outs RefTypeConstraint:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def ReadOp : ProbeInferTypeOp<"read", [
  TypesMatchWith<"input and result types match",
    "input", "result",
    "cast<::circt::probe::RefType>($_self).getElementType()">
]> {
  let summary = "read the value observed by a probe handle";
  let arguments = (ins RefTypeConstraint:$input);
  let results = (outs ProbeElementType:$result);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

#endif // CIRCT_DIALECT_PROBE_PROBEOPS_TD
