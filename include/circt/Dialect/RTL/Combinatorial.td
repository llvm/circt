//===- Combinatorial.td - RTL combinatorial logic ops ------*- tablegen -*-===//
//
// This describes the MLIR ops for combinatorial logic.
//
//===----------------------------------------------------------------------===//

def ConstantOp
 : RTLOp<"constant",
          [NoSideEffect, ConstantLike, FirstAttrDerivedResultType,
           DeclareOpInterfaceMethods<OpAsmOpInterface>]> {
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
  
  // FIXME(QoI): Instead of requiring "rtl.constant (42: i8) : i8", we should
  // just use "rtl.constant 42: i8".  This can be done with a custom printer and
  // parser, but would be better to be autoderived from the
  // FirstAttrDerivedResultType trait.
  let assemblyFormat = [{
     `(` $value `)` attr-dict `:` type($result)
  }];

  let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &result, const APInt &value">
  ];
  let extraClassDeclaration = [{
    APInt getValue() { return getAttrOfType<IntegerAttr>("value").getValue(); }
  }];
  let hasFolder = 1;
  let verifier = [{ return ::verifyConstantOp(*this); }];
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

// Base class for binary operators.
class BinRTLOp<string mnemonic, list<OpTrait> traits = []> :
      RTLOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
  let arguments = (ins AnySignlessInteger:$lhs, AnySignlessInteger:$rhs);
  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:` functional-type($args, results)
  }];
}

// Binary operator with uniform input/result types.
class UTBinRTLOp<string mnemonic, list<OpTrait> traits = []> :
      BinRTLOp<mnemonic,
               !listconcat(traits,
                           [SameTypeOperands, SameOperandsAndResultType])> {
   let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:` type($result)
  }];
}

// Arithmetic and Logical Binary Operations.
def AddOp : UTBinRTLOp<"add", [Commutative]>;
def SubOp : UTBinRTLOp<"sub">;
def MulOp : UTBinRTLOp<"mul", [Commutative]>;
def DivOp : UTBinRTLOp<"div">;
def ModOp : UTBinRTLOp<"mod">;
def AndOp : UTBinRTLOp<"and", [Commutative]>;
def OrOp  : UTBinRTLOp<"or",  [Commutative]>;
def XorOp : UTBinRTLOp<"xor", [Commutative]>;

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

def SExtOp : RTLOp<"sext", [NoSideEffect]> {
  let summary = "Sign extend an integer to a larger integer";

  let arguments = (ins AnySignlessInteger:$input);
  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `,` type($result)
  }];

  let verifier = [{ return ::verifyExtOp(*this); }];
}

// TODO: Should zext just be a concat with zeros?
def ZExtOp : RTLOp<"zext", [NoSideEffect]> {
  let summary = "Zero extend an integer to a larger integer";

  let arguments = (ins AnySignlessInteger:$input);
  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `,` type($result)
  }];

  let verifier = [{ return ::verifyExtOp(*this); }];
}

// TODO: Bit extract range, use it for trunc.

def ConcatOp : RTLOp<"concat", [NoSideEffect]> {
  let summary = "Concatenate a variadic list of operands together.";

  let arguments = (ins Variadic<AnySignlessInteger>:$inputs);
  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = [{
    $inputs attr-dict `:` functional-type($inputs, results)
  }];
}
