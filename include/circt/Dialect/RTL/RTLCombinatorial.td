//===- RTLCombinatorial.td - RTL combinatorial logic ops ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
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
  let results = (outs RTLIntegerType:$result);

  // FIXME(QoI): Instead of requiring "rtl.constant (42: i8) : i8", we should
  // just use "rtl.constant 42: i8".  This can be done with a custom printer and
  // parser, but would be better to be autoderived from the
  // FirstAttrDerivedResultType trait.  This should be an MLIR core improvement.
  let assemblyFormat = "`(` $value `)` attr-dict `:` type($result)";

  let builders = [
    /// Build a ConstantOp from an APInt, infering the result type from the
    /// width of the APInt.
    OpBuilderDAG<(ins "const APInt &":$value)>,

    /// This builder allows construction of small signed integers like 0, 1, -1
    /// matching a specified MLIR IntegerType.  This shouldn't be used for
    /// general constant folding because it only works with values that can be
    /// expressed in an int64_t.  Use APInt's instead.
    OpBuilderDAG<(ins "Type":$type, "int64_t":$value)>
  ];
  let extraClassDeclaration = [{
    APInt getValue() {
      return (*this)->getAttrOfType<IntegerAttr>("value").getValue();
    }
  }];
  let hasFolder = true;
  let verifier = "return ::verifyConstantOp(*this);";
}

//===----------------------------------------------------------------------===//
// Arithmetic and Logical Operations
//===----------------------------------------------------------------------===//

// Base class for binary operators.
class BinRTLOp<string mnemonic, list<OpTrait> traits = []> :
      RTLOp<mnemonic, traits # [NoSideEffect]> {
  let arguments = (ins RTLIntegerType:$lhs, RTLIntegerType:$rhs);
  let results = (outs RTLIntegerType:$result);

  let assemblyFormat =
    "$lhs `,` $rhs  attr-dict `:` functional-type($args, $results)";
}

// Binary operator with uniform input/result types.
class UTBinRTLOp<string mnemonic, list<OpTrait> traits = []> :
      BinRTLOp<mnemonic,
               traits # [SameTypeOperands, SameOperandsAndResultType]> {
  let assemblyFormat = "$lhs `,` $rhs  attr-dict `:` type($result)";
}

// Base class for variadic operators.
class VariadicRTLOp<string mnemonic, list<OpTrait> traits = []> :
      RTLOp<mnemonic, traits # [NoSideEffect]> {
  let arguments = (ins Variadic<RTLIntegerType>:$inputs);
  let results = (outs RTLIntegerType:$result);

  let assemblyFormat =
    "$inputs attr-dict `:` functional-type($inputs, results)";
}

class UTVariadicRTLOp<string mnemonic, list<OpTrait> traits = []> :
      VariadicRTLOp<mnemonic,
                    traits # [SameTypeOperands, SameOperandsAndResultType]> {

  let hasCanonicalizer = true;
  let hasFolder = true;
  let verifier =  "return ::verifyUTVariadicRTLOp(*this);";

  let assemblyFormat = "$inputs attr-dict `:` type($result)";

  let builders = [
    OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs), [{
      return build($_builder, $_state, lhs.getType(),
                   ValueRange{lhs, rhs}, ArrayRef<NamedAttribute>{});
    }]>
  ];
}

// Arithmetic and Logical Operations.
def AddOp : UTVariadicRTLOp<"add", [Commutative]>;
def SubOp : UTBinRTLOp<"sub">;
def MulOp : UTVariadicRTLOp<"mul", [Commutative]>;
def DivUOp : UTBinRTLOp<"divu">;
def DivSOp : UTBinRTLOp<"divs">;
def ModUOp : UTBinRTLOp<"modu">;
def ModSOp : UTBinRTLOp<"mods">;
let hasFolder = true in {
  def ShlOp : UTBinRTLOp<"shl">;
  def ShrUOp : UTBinRTLOp<"shru">;
  def ShrSOp : UTBinRTLOp<"shrs">;
}

def AndOp : UTVariadicRTLOp<"and", [Commutative]>;
def OrOp  : UTVariadicRTLOp<"or", [Commutative]>;
def XorOp : UTVariadicRTLOp<"xor", [Commutative]>;

def MergeOp : UTVariadicRTLOp<"merge", [Commutative]> {
  let summary ="Electrically merge signals together as if connected by a wire.";
  let description = [{
    This operation merges multiple signals together as if they were connected to
    a single wire.  Presumably at most one of them should be a Z state at any
    given time otherwise you'd get a short circuit.

    This operation allows modeling multiconnect semantics in SSA.
    ```
      %result = rtl.merge %a, %b, %c : t1
    ```
    }];
  let hasCanonicalizer = false;
}

//===----------------------------------------------------------------------===//
// Comparisons
//===----------------------------------------------------------------------===//

def ICmpPredicateEQ  : I64EnumAttrCase<"eq", 0>;
def ICmpPredicateNE  : I64EnumAttrCase<"ne", 1>;
def ICmpPredicateSLT : I64EnumAttrCase<"slt", 2>;
def ICmpPredicateSLE : I64EnumAttrCase<"sle", 3>;
def ICmpPredicateSGT : I64EnumAttrCase<"sgt", 4>;
def ICmpPredicateSGE : I64EnumAttrCase<"sge", 5>;
def ICmpPredicateULT : I64EnumAttrCase<"ult", 6>;
def ICmpPredicateULE : I64EnumAttrCase<"ule", 7>;
def ICmpPredicateUGT : I64EnumAttrCase<"ugt", 8>;
def ICmpPredicateUGE : I64EnumAttrCase<"uge", 9>;
def ICmpPredicate : I64EnumAttr<
    "ICmpPredicate",
    "rtl.icmp comparison predicate",
    [ICmpPredicateEQ, ICmpPredicateNE, ICmpPredicateSLT, ICmpPredicateSLE,
     ICmpPredicateSGT, ICmpPredicateSGE, ICmpPredicateULT, ICmpPredicateULE,
     ICmpPredicateUGT, ICmpPredicateUGE]>;

def ICmpOp : RTLOp<"icmp", [NoSideEffect, SameTypeOperands]> {
  let summary = "Compare two integer values";
  let description = [{
    This operation compares two integers using a predicate.  If the predicate is
    true, returns 1, otherwise returns 0. This operation always returns a one
    bit wide result.

    ```
        %r = rtl.icmp eq %a, %b : i4
    ```
  }];

  let arguments = (ins ICmpPredicate:$predicate, 
                   RTLIntegerType:$lhs, RTLIntegerType:$rhs);
  let results = (outs I1:$result);

  let assemblyFormat = "$predicate $lhs `,` $rhs  attr-dict `:` type($lhs)";
  
  let hasFolder = true;
  let hasCanonicalizer = true;

  let builders = [
    OpBuilderDAG<(ins "ICmpPredicate":$pred, "Value":$lhs, "Value":$rhs), [{
      return build($_builder, $_state, $_builder.getI1Type(), pred, lhs, rhs);
    }]>
  ];
  let extraClassDeclaration = [{
    /// Returns the flipped predicate, reversing the LHS and RHS operands.  The
    /// lhs and rhs operands should be flipped to match the new predicate.
    static ICmpPredicate getFlippedPredicate(ICmpPredicate predicate);
  }];
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Base class for unary reduction operations that produce an i1.
class UnaryI1ReductionRTLOp<string mnemonic, list<OpTrait> traits = []> :
      RTLOp<mnemonic, traits # [NoSideEffect]> {
  let arguments = (ins RTLIntegerType:$input);
  let results = (outs I1:$result);
  let hasFolder = 1;

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def AndROp : UnaryI1ReductionRTLOp<"andr">;
def OrROp  : UnaryI1ReductionRTLOp<"orr">;
def XorROp : UnaryI1ReductionRTLOp<"xorr">;

//===----------------------------------------------------------------------===//
// Integer width modifying operations.
//===----------------------------------------------------------------------===//

// Extract a range of bits from the specified input.
def ExtractOp : RTLOp<"extract", [NoSideEffect]> {
  let summary = "Extract a range of bits into a smaller value, lowBit "
                "specifies the lowest bit included.";

  let arguments = (ins RTLIntegerType:$input, I32Attr:$lowBit);
  let results = (outs RTLIntegerType:$result);

  let assemblyFormat =
    "$input `from` $lowBit attr-dict `:` functional-type($input, $result)";

  let hasFolder = true;
  let verifier = "return ::verifyExtractOp(*this);";
}

def SExtOp : RTLOp<"sext", [NoSideEffect]> {
  let summary = "Sign extend an integer to a larger integer";

  let arguments = (ins RTLIntegerType:$input);
  let results = (outs RTLIntegerType:$result);

  let assemblyFormat =
    "$input attr-dict `:` functional-type($input, $result)";

  let hasFolder = true;
  let verifier = "return ::verifySExtOp(*this);";
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

def ConcatOp : VariadicRTLOp<"concat"> {
  let summary = "Concatenate a variadic list of operands together.";
  let description = [{
    See the RTL-SV rationale document for details on operand ordering.
  }];

  let hasFolder = true;
  let hasCanonicalizer = true;

  let builders = [
    OpBuilderDAG<(ins "ValueRange":$inputs)>,
    OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs), [{
      return build($_builder, $_state, ValueRange{lhs, rhs});
    }]>
  ];
}

// Select one of two values based on a condition.
def MuxOp : RTLOp<"mux",
 [NoSideEffect, AllTypesMatch<["trueValue", "falseValue", "result"]>]> {
  let summary = "Return one or the other operand depending on a selector bit";
  let description = [{
    ```
      %0 = mux %pred, %tvalue, %fvalue : i4
    ```
  }];

  let arguments = (ins I1:$cond, RTLNonInOutType:$trueValue,
                       RTLNonInOutType:$falseValue);
  let results = (outs RTLNonInOutType:$result);

  let assemblyFormat =
    "$cond `,` $trueValue `,` $falseValue attr-dict `:` type($result)";
  
  let hasFolder = true;

  let builders = [
    OpBuilderDAG<(ins "Value":$cond, "Value":$trueValue, "Value":$falseValue),
    [{
      assert(trueValue.getType() == falseValue.getType() &&
             "mux requires matching true/false value types");
      return build($_builder, $_state, trueValue.getType(),
                   cond, trueValue, falseValue);
    }]>
  ];
}

def KnownBitWidthType : Type<CPred<[{getBitWidth($_self) != -1}]>,
  "Type wherein the bitwidth in hardware is known">;

def BitcastOp: RTLOp<"bitcast", [NoSideEffect]> {
  let summary = [{
    Reinterpret one value to another value of the same size and
    potentially different type.
  }];

  let description = [{
    See the RTL-SV rationale document for a longer description, including an
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
