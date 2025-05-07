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
include "mlir/IR/EnumAttr.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

include "circt/Dialect/HW/HWTypes.td"

// Base class for the operation in this dialect.
class DatapathOp<string mnemonic, list<Trait> traits = []> :
    Op<DatapathDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//

def CompressOp : DatapathOp<"compress", [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Apply a compressor tree to reduce a set of input bitvectors, "
                "to be added to targetRows output bitvectors.";

  let arguments = (ins Variadic<HWIntegerType>:$inputs);
  let results   = (outs Variadic<HWIntegerType>:$results);

  let assemblyFormat =
    // "$inputs `to` $targetRows attr-dict `:` qualified(type($inputs))";
    // "$inputs `to` $targetRows attr-dict `:` functional-type($inputs, $results)";
    "$inputs attr-dict `:` functional-type($inputs, $results)";
    // "$inputs attr-dict `:`  qualified(type($results))";

  // let hasFolder = true;
  let hasVerifier = 1;
  // let hasCanonicalizeMethod = true;

  // let builders = [
  //   OpBuilder<(ins "ValueRange":$lhs, "int32_t":$targetRows), [{
  //     auto inputType = lhs.front().getType();
  //     SmallVector<Type> resultTypes(targetRows, inputType);
  //     // for (auto i=0; i< targetRows; ++i) 
  //     //   resultTypes.push_back(inputType);
  //     return build($_builder, $_state, resultTypes, lhs, targetRows);
  //   }]>
  // ];
}

#endif // CIRCT_DIALECT_DATAPATH_OPS_TD