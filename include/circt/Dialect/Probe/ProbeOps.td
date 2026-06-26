//===- ProbeOps.td - Probe dialect operations --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_PROBE_PROBEOPS_TD
#define CIRCT_DIALECT_PROBE_PROBEOPS_TD

include "circt/Dialect/HW/HWTypes.td"
include "circt/Dialect/Probe/ProbeDialect.td"
include "circt/Dialect/Probe/ProbeTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/OpBase.td"

class ProbeOp<string mnemonic, list<Trait> traits = []> :
    Op<ProbeDialect, mnemonic, traits>;

def SendOp : ProbeOp<"send", [Pure]> {
  let summary = "create a read-only probe handle for a hardware value";
  let arguments = (ins HWNonInOutType:$input);
  let results = (outs RefTypeConstraint:$result);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def ReadOp : ProbeOp<"read", [Pure]> {
  let summary = "read the value observed by a probe handle";
  let arguments = (ins RefTypeConstraint:$input);
  let results = (outs HWNonInOutType:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` type($result)";
  let hasVerifier = 1;
}

def SubfieldOp : ProbeOp<"subfield", [Pure]> {
  let summary = "extract a field probe from a struct probe";
  let arguments = (ins RefTypeConstraint:$input, StrAttr:$field);
  let results = (outs RefTypeConstraint:$result);
  let assemblyFormat = "$input `[` $field `]` attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def SubindexOp : ProbeOp<"subindex", [Pure]> {
  let summary = "extract an element probe from an array probe";
  let arguments = (ins RefTypeConstraint:$input, I32Attr:$index);
  let results = (outs RefTypeConstraint:$result);
  let assemblyFormat = "$input `[` $index `]` attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

def CastOp : ProbeOp<"cast", [Pure]> {
  let summary = "cast between compatible probe handle types";
  let arguments = (ins RefTypeConstraint:$input);
  let results = (outs RefTypeConstraint:$result);
  let assemblyFormat = "$input attr-dict `:` qualified(type($input)) `->` qualified(type($result))";
  let hasVerifier = 1;
}

#endif // CIRCT_DIALECT_PROBE_PROBEOPS_TD
