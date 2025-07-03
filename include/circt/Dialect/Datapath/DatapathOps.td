//===----------------------------------------------------------------------===//
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
include "mlir/Interfaces/SideEffectInterfaces.td"

include "circt/Dialect/HW/HWTypes.td"

// Base class for the operation in this dialect.
class DatapathOp<string mnemonic, list<Trait> traits = []> :
    Op<DatapathDialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//

// Compress an array of bitvectors to a smaller set of bitvectors (at least 2).
def CompressOp : DatapathOp<"compress", 
                                   [Pure, SameTypeOperands, 
                                    SameOperandsAndResultType, Commutative]> {
  let summary = "Reduce a set of bitvectors to a carry-save representation";
  let description = [{
    A compressor tree sums multiple bitvectors (often partial products in
    multipliers or adders). Instead of adding all bitvectors sequentially, a 
    compressor tree reduces the number of operands in parallel stages. The 
    result is stored in a redundant (carry-save) representation, deferring the 
    compressor tree implementation to a later stage.

    Example:
    ```mlir
    %0:2 = datapath.compress %a, %b, %c : 3 x i16 -> (i16, i16)
    ```
  }];

  let arguments = (ins Variadic<HWIntegerType>:$inputs);
  let results   = (outs Variadic<HWIntegerType>:$results);

  let hasCustomAssemblyFormat = true;
  let hasVerifier = 1;
  let hasCanonicalizer = true;

  let builders = [
    OpBuilder<(ins "ValueRange":$lhs, "int32_t":$targetRows), [{
      auto inputType = lhs.front().getType();
      SmallVector<Type> resultTypes(targetRows, inputType);
      return build($_builder, $_state, resultTypes, lhs);
    }]>
  ];
}

// Construct partial product array from two operands
def PartialProductOp : DatapathOp<"partial_product", 
                                    [Pure, SameTypeOperands, 
                                     SameOperandsAndResultType, Commutative]> {
  let summary = "Generate partial products from multiplying the operands";
  let description = [{
    The first step in a multiplication is to generate partial products, which 
    when summed, yield the product of the two operands. The partial
    product operator does not specify an implementation, only that summing the 
    results will yield the product of the two operands. 

    Example:
    ```mlir
    %0:8 = datapath.partial_product %a, %b : 8 x i8
    ```
  }];
  let arguments = (ins HWIntegerType:$multiplicand, 
                       HWIntegerType:$multiplier);
  let results   = (outs Variadic<HWIntegerType>:$results);

  let hasCustomAssemblyFormat = true;
  let hasCanonicalizer = true;

  let builders = [
    OpBuilder<(ins "ValueRange":$lhs, "int32_t":$targetRows), [{
      auto inputType = lhs.front().getType();
      SmallVector<Type> resultTypes(targetRows, inputType);
      return build($_builder, $_state, resultTypes, lhs);
    }]>
  ];
}

#endif // CIRCT_DIALECT_DATAPATH_OPS_TD
