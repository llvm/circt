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

def ConstantOp
 : HWOp<"constant",
          [NoSideEffect, ConstantLike, FirstAttrDerivedResultType,
           DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
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

  let parser = "return parse$cppClass(parser, result);";
  let printer = "print$cppClass(p, *this);";

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
  let extraClassDeclaration = [{
    APInt getValue() {
      return (*this)->getAttrOfType<IntegerAttr>("value").getValue();
    }
  }];
  let hasFolder = true;
  let verifier = "return ::verifyConstantOp(*this);";
}

def KnownBitWidthType : Type<CPred<[{getBitWidth($_self) != -1}]>,
  "Type wherein the bitwidth in hardware is known">;

def BitcastOp: HWOp<"bitcast", [NoSideEffect]> {
  let summary = [{
    Reinterpret one value to another value of the same size and
    potentially different type.
  }];

  let description = [{
    See the HW-SV rationale document for a longer description, including an
    example.
  }];

  let arguments = (ins KnownBitWidthType:$input);
  let results = (outs KnownBitWidthType:$result);
  let verifier = [{
    if (getBitWidth(input().getType()) !=
        getBitWidth(result().getType()))
      return this->emitOpError("Bitwidth of input must match result");
    return success();
  }];

  let assemblyFormat = [{
    $input attr-dict `:` functional-type($input, $result)
  }];
}
