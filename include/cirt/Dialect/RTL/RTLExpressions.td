//===- OpExpressions.td - RTL expression ops ---------------*- tablegen -*-===//
//
// This describes the MLIR ops for RTL expressions.
//
//===----------------------------------------------------------------------===//

def ConstantOp : RTLOp<"constant", [NoSideEffect, ConstantLike,
                                    FirstAttrDerivedResultType]> {
  let summary = "Produce a constant value";
  let description = [{
    The constant operation produces a constant value of standard integer type
    without a sign.
    ```
      %result = rtl.constant 42 : t1
    ```
    }];

  let arguments = (ins APIntAttr:$value);
  let results = (outs AnySignlessInteger:$result);
  
  let assemblyFormat = [{
     `(` $value `)` attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, const APInt &value">
  ];
  let hasFolder = 1;
  let verifier = [{ return ::verify(*this); }];
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

// Base class for binary operators.
class BinOp<string mnemonic, list<OpTrait> traits = []> :
    RTLOp<mnemonic,
          !listconcat(traits, [SameTypeOperands, SameOperandsAndResultType])> {
  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs);
  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:` type($result)
  }];
}

def AddOp : BinOp<"add", [Commutative]>;
def SubOp : BinOp<"sub", []>;
def MulOp : BinOp<"mul", [Commutative]>;
def DivOp : BinOp<"div", []>;
