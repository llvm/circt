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
  // FirstAttrDerivedResultType trait.  This should be an MLIR core improvement.
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

// Base class for variadic operators.
class VariadicRTLOp<string mnemonic, list<OpTrait> traits = []> :
      RTLOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
  let arguments = (ins Variadic<AnySignlessInteger>:$operands);
  let results = (outs AnySignlessInteger:$result);

  let assemblyFormat = [{
    $operands  attr-dict `:` functional-type($operands, results)
  }];
}

class UTVariadicRTLOp<string mnemonic, list<OpTrait> traits = []> :
      VariadicRTLOp<mnemonic,
               !listconcat(traits,
                           [SameTypeOperands, SameOperandsAndResultType])> {
   let assemblyFormat = [{
    $operands  attr-dict `:` type($result)
  }];
}

// Arithmetic and Logical Binary Operations.
def AddOp : UTVariadicRTLOp<"add", [Commutative]>;
def SubOp : UTBinRTLOp<"sub">;
def MulOp : UTVariadicRTLOp<"mul", [Commutative]>;
def DivOp : UTBinRTLOp<"div">;
def ModOp : UTBinRTLOp<"mod">;

def AndOp : UTVariadicRTLOp<"and", [Commutative]>;
def OrOp  : UTVariadicRTLOp<"or", [Commutative]>;
def XorOp : UTVariadicRTLOp<"xor", [Commutative]>;

// Concatenate a variadic list of operands together.
def ConcatOp : VariadicRTLOp<"concat">;

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Base class for unary reduction operations that produce an i1.
class UnaryI1ReductionRTLOp<string mnemonic, list<OpTrait> traits = []> :
      RTLOp<mnemonic, !listconcat(traits, [NoSideEffect])> {
  let arguments = (ins AnySignlessInteger:$input);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    $input attr-dict `:` type($input)
  }];
}

def AndROp : UnaryI1ReductionRTLOp<"andr">;
def OrROp  : UnaryI1ReductionRTLOp<"orr">;
def XorROp : UnaryI1ReductionRTLOp<"xorr">;

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
