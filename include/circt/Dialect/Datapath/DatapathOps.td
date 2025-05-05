//===- DatapathOps.td - datapath ops ----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for datapath logic.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_DATAPATH_OPS_TD
#define CIRCT_DIALECT_DATAPATH_OPS_TD

include "circt/Dialect/Datapath/DatapathDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/IR/SymbolInterfaces.td"

// Base class for the operation in this dialect.
class DatapathOp<string mnemonic, list<Trait> traits = []> :
    Op<DatapathDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//

def CompressOp : DatapathOp<"compress", [Pure]> {
  let summary = "Apply a compressor tree to reduce a set of input bitvectors, "
                "to be added to targetRows output bitvectors.";

  let arguments = (ins Variadic<HWIntegerType>:$inputs, I32Attr:$targetRows);
  let results = (outs Variadic<HWIntegerType>:$result);

  let assemblyFormat =
    "$inputs `to` $targetRows attr-dict `:` type($inputs)";

  let hasFolder = true;
  let hasVerifier = 1;
  let hasCanonicalizeMethod = true;

  let builders = [
    OpBuilder<(ins "Value":$lhs, "int32_t":$lowBit, "int32_t":$bitWidth), [{
      auto resultType = $_builder.getIntegerType(bitWidth);
      return build($_builder, $_state, resultType, lhs, lowBit);
    }]>
  ];
}

#endif // CIRCT_DIALECT_DATAPATH_OPS_TD