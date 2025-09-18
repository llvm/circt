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

def CompressOp : DatapathOp<"compress", 
            [Pure, SameTypeOperands, SameOperandsAndResultType, Commutative]> {
  let summary = "Reduce a set of bitvectors to a carry-save representation";
  let description = [{
    Reduce an array of bitvectors to a smaller set of bitvectors (at least 2).
    A compressor tree sums multiple bitvectors (often partial products in
    multipliers or adders). Instead of adding all bitvectors sequentially, a 
    compressor tree reduces the number of operands in parallel stages. The 
    result is stored in a redundant (carry-save) representation, deferring the 
    compressor tree implementation to a later stage.

    Example:
    ```mlir
    %0:2 = datapath.compress %a, %b, %c : i16 [3 -> 2]
    ```
  }];

  let arguments = (ins Variadic<HWIntegerType>:$inputs);
  let results   = (outs Variadic<HWIntegerType>:$results);

  let assemblyFormat = [{
    $inputs attr-dict `:` custom<CompressFormat>(type($inputs), type($results))
  }];
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

def PartialProductOp : DatapathOp<"partial_product", 
            [Pure, SameTypeOperands, SameOperandsAndResultType, Commutative]> {
  let summary = "Generate partial products from multiplying the operands";
  let description = [{
    The first step in a multiplication is to generate partial products, which 
    when summed, yield the product of the two operands. The partial
    product operator does not specify an implementation, only that summing the 
    results will yield the product of the two operands. The number of results 
    corresponds to the rows of a partial product array, which by default is
    equal to the width of the inputs. 
    
    Verilog Example 4-bit multiplication:
    ```verilog
    partial_product[0][3:0] = {4{a[0]}} & b
    ...
    partial_product[3][3:0] = {4{a[3]}} & b
    ab[3:0] = partial_product[0] + ... + partial_product[3] // = a*b
    ```

    Example using `datapath` dialect:
    ```mlir
    %0:4 = datapath.partial_product %a, %b : (i4, i4) -> (i4, i4, i4, i4)
    ```
  }];
  let arguments = (ins HWIntegerType:$lhs, 
                       HWIntegerType:$rhs);
  let results   = (outs Variadic<HWIntegerType>:$results);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` functional-type(operands, results)
  }];
  let hasCanonicalizer = true;

  let builders = [
    OpBuilder<(ins "ValueRange":$lhs, "int32_t":$targetRows), [{
      auto inputType = lhs.front().getType();
      SmallVector<Type> resultTypes(targetRows, inputType);
      return build($_builder, $_state, resultTypes, lhs);
    }]>
  ];
}

def PosPartialProductOp : DatapathOp<"pos_partial_product", 
            [Pure, SameTypeOperands, SameOperandsAndResultType]> {
  let summary = "Generate partial products for (a+b)*c";
  let description = [{
    A partial product array which when summed produces the result (a+b)*c. 
    An efficient circuit can be constructed that encodes a carry-save input
    without performing a full carry-propagation. The number of results 
    corresponds to the rows of a partial product array, which by default is 
    equal to the width of the inputs. 
    
    Example using `datapath` dialect:
    ```mlir
    %0:3 = datapath.pos_partial_product %a, %b, %c : (i3, i3, i3) 
                                                  -> (i3, i3, i3)
    ```
  }];
  let arguments = (ins HWIntegerType:$addend0, HWIntegerType:$addend1,
                       HWIntegerType:$multiplicand);
  let results   = (outs Variadic<HWIntegerType>:$results);

  let assemblyFormat = [{
    $addend0  `,` $addend1 `,` $multiplicand attr-dict `:` functional-type(operands, results)
  }];
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
