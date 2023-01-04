//===- HWMiscOps.td - Miscellaneous HW ops -----------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This defines miscellaneous generic HW ops, like ConstantOp and BitcastOp.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_HWMISCOPS_TD
#define CIRCT_DIALECT_HW_HWMISCOPS_TD

include "circt/Dialect/HW/HWAttributes.td"
include "circt/Dialect/HW/HWDialect.td"
include "circt/Dialect/HW/HWTypes.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp
 : HWOp<"constant", [Pure, ConstantLike, FirstAttrDerivedResultType,
         DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Produce a constant value";
  let description = [{
    The constant operation produces a constant value of standard integer type
    without a sign.
    ```
      %result = hw.constant 42 : t1
    ```
    }];

  let arguments = (ins APIntAttr:$value);
  let results = (outs HWIntegerType:$result);
  let hasCustomAssemblyFormat = 1;

  let builders = [
    /// Build a ConstantOp from an APInt, infering the result type from the
    /// width of the APInt.
    OpBuilder<(ins "const APInt &":$value)>,

    /// This builder allows construction of small signed integers like 0, 1, -1
    /// matching a specified MLIR IntegerType.  This shouldn't be used for
    /// general constant folding because it only works with values that can be
    /// expressed in an int64_t.  Use APInt's instead.
    OpBuilder<(ins "Type":$type, "int64_t":$value)>,

    /// Build a ConstantOp from a prebuilt attribute.
    OpBuilder<(ins "IntegerAttr":$value)>
  ];
  let hasFolder = true;
  let hasVerifier = 1;
}

def KnownBitWidthType : Type<CPred<[{getBitWidth($_self) != -1}]>,
  "Type wherein the bitwidth in hardware is known">;

def BitcastOp: HWOp<"bitcast", [Pure]> {
  let summary = [{
    Reinterpret one value to another value of the same size and
    potentially different type.  See the `hw` dialect rationale document for
    more details.
  }];

  let arguments = (ins KnownBitWidthType:$input);
  let results = (outs KnownBitWidthType:$result);
  let hasCanonicalizeMethod = true;
  let hasFolder = true;
  let hasVerifier = 1;

  let assemblyFormat = "$input attr-dict `:` functional-type($input, $result)";
}

def ParamValueOp : HWOp<"param.value",
                        [FirstAttrDerivedResultType, Pure,
                         ConstantLike]> {
  let summary = [{
    Return the value of a parameter expression as an SSA value that may be used
    by other ops.
  }];

  let arguments = (ins AnyAttr:$value);
  let results = (outs HWValueType:$result);
  let assemblyFormat = "custom<ParamValue>($value, qualified(type($result))) attr-dict";
  let hasVerifier = 1;
  let hasFolder = true;
}

def EnumConstantOp : HWOp<"enum.constant", [Pure, ConstantLike,
         DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>]> {
  let summary = "Produce a constant enumeration value.";
  let description = [{
    The enum.constant operation produces an enumeration value of the specified
    enum value attribute.
    ```
      %0 = hw.enum.constant A : !hw.enum<A, B, C>
    ```
    }];

  let arguments = (ins EnumFieldAttr:$field);
  let results = (outs EnumType:$result);
  let hasCustomAssemblyFormat = 1;
  let hasFolder = true;

  let builders = [
    OpBuilder<(ins "hw::EnumFieldAttr":$field)>,
  ];
}

#endif // CIRCT_DIALECT_HW_HWMISCOPS_TD
