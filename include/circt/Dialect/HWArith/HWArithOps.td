//===- HWArithOps.td - HW arithmetic operations ------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines HW arithmetic operations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HWARITH_HWARITHOPS_TD
#define CIRCT_DIALECT_HWARITH_HWARITHOPS_TD

include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : HWArithOp<"constant",
    [ConstantLike, NoSideEffect, FirstAttrDerivedResultType]> {
  let summary = "Produce a constant value";
  let description = [{
    The constant operation produces a sign-aware constant value.
    ```
      %result = hwarith.constant 42 : t1
    ```
    }];

  let arguments = (ins APIntAttr:$rawValue);
  let results = (outs HWArithIntegerType:$result);

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    APSInt getConstantValue();
  }];

  // Folder is required for constant like operations
  let hasFolder = true;
}

// Base class for binary operators.
class BinOp<string mnemonic, list<Trait> traits = []> :
      HWArithOp<mnemonic, traits # [NoSideEffect, InferTypeOpInterface]> {
  let arguments = (ins Variadic<HWArithIntegerType>:$inputs);
  let results = (outs HWArithIntegerType:$result);

  let assemblyFormat =
      "$inputs attr-dict `:` functional-type($inputs, $result)";

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    /// Infer the return types of this operation.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          Optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results);
  }];

  let extraClassDefinition = [{
    LogicalResult $cppClass::verify() { return hwarith::verifyBinOp(*this); }
  }];
}

// Arithmetic Operations.
def AddOp : BinOp<"add", [Commutative]> {
  let summary = "Bitwidth-aware integer addition.";
  let description = [{
    The `add` operation takes two operands and returns one result. The result
    type is inferred from the operand types, which may be signed or unsigned
    scalar integer types of arbitrary bitwidth.
    
    | LHS type | RHS type | Result type                              |
    | :------- | :------- | :--------------------------------------- |
    | `ui<a>`  | `ui<b>`  | `ui<r>`, *r* = max(*a*, *b*) + 1         |
    | `si<a>`  | `si<b>`  | `si<r>`, *r* = max(*a*, *b*) + 1         |
    | `ui<a>`  | `si<b>`  | `si<r>`, *r* = *a* + 2 **if** *a* ≥ *b*  |
    |          |          | `si<r>`, *r* = *b* + 1 **if** *a* < *b*  |
    | `si<a>`  | `ui<b>`  | Same as `ui<b> + si<a>`                  |

    Examples:
    ```mlir
    %0 = hwarith.add %10, %11 : (ui3, ui4) -> ui5
    %1 = hwarith.add %12, %13 : (si3, si3) -> si4
    %2 = hwarith.add %14, %15 : (ui3, si4) -> si5
    %3 = hwarith.add %16, %17 : (si4, ui6) -> si8
    ```
  }];
}

def SubOp : BinOp<"sub"> {
  let summary = "Bitwidth-aware integer subtraction.";
  let description = [{
    The `sub` operation takes two operands and returns one result. The result
    type is inferred from the operand types, which may be signed or unsigned
    scalar integer types of arbitrary bitwidth.
    
    | LHS type | RHS type | Result type                              |
    | :------- | :------- | :--------------------------------------- |
    | `ui<a>`  | `ui<b>`  | `si<r>`, *r* = max(*a*, *b*) + 1         |
    | `si<a>`  | `si<b>`  | `si<r>`, *r* = max(*a*, *b*) + 1         |
    | `ui<a>`  | `si<b>`  | `si<r>`, *r* = *a* + 2 **if** *a* ≥ *b*  |
    |          |          | `si<r>`, *r* = *b* + 1 **if** *a* < *b*  |
    | `si<a>`  | `ui<b>`  | Same as `ui<b> - si<a>`                  |

    Examples:
    ```mlir
    %0 = hwarith.sub %10, %11 : (ui3, ui4) -> si5
    %1 = hwarith.sub %12, %13 : (si3, si3) -> si4
    %2 = hwarith.sub %14, %15 : (ui3, si4) -> si5
    %3 = hwarith.sub %16, %17 : (si4, ui6) -> si8
    ```
  }];
}

def MulOp : BinOp<"mul", [Commutative]> {
  let summary = "Bitwidth-aware integer multiplication.";
  let description = [{
    The `mul` operation takes two operands and returns one result. The result
    type is inferred from the operand types, which may be signed or unsigned
    scalar integer types of arbitrary bitwidth.
    
    | LHS type | RHS type | Result type                              |
    | :------- | :------- | :--------------------------------------- |
    | `ui<a>`  | `ui<b>`  | `ui<r>`, *r* = *a* + *b*                 |
    | `si<a>`  | `si<b>`  | `si<r>`, *r* = *a* + *b*                 |
    | `ui<a>`  | `si<b>`  | `si<r>`, *r* = *a* + *b*                 |
    | `si<a>`  | `ui<b>`  | `si<r>`, *r* = *a* + *b*                 |

    Examples:
    ```mlir
    %0 = hwarith.mul %10, %11 : (ui3, ui4) -> ui7
    %1 = hwarith.mul %12, %13 : (si3, si3) -> si6
    %2 = hwarith.mul %14, %15 : (si3, ui5) -> si8
    ```
  }];
}

def DivOp : BinOp<"div"> {
  let summary = "Bitwidth-aware integer division.";
  let description = [{
    The `div` operation takes two operands and returns one result. The result
    type is inferred from the operand types, which may be signed or unsigned
    scalar integer types of arbitrary bitwidth.
    
    | LHS type | RHS type | Result type                              |
    | :------- | :------- | :--------------------------------------- |
    | `ui<a>`  | `ui<b>`  | `ui<r>`, *r* = *a*                       |
    | `si<a>`  | `si<b>`  | `si<r>`, *r* = *a* + 1                   |
    | `ui<a>`  | `si<b>`  | `si<r>`, *r* = *a* + 1                   |
    | `si<a>`  | `ui<b>`  | `si<r>`, *r* = *a*                       |

    Examples:
    ```mlir
    %0 = hwarith.div %10, %11 : (ui3, ui4) -> ui3
    %1 = hwarith.div %12, %13 : (si3, si3) -> si4
    %2 = hwarith.div %14, %15 : (ui3, si4) -> si4
    %3 = hwarith.div %16, %17 : (si4, ui6) -> si4
    ```
  }];
}

#endif // CIRCT_DIALECT_HWARITH_HWARITHOPS_TD
