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

// Extract a range of bits from the specified input.
def StructExtractOp : RTLOp<"struct_extract", [NoSideEffect]> {
  let summary = "Extract a named field from a struct.";

  let arguments = (ins StructType:$input, StrAttr:$field);
  let results = (outs RTLNonInOutType:$result);

  let parser = "return ::parse$cppClass(parser, result);";
  let printer = "return ::print(p, *this);";
}

// Extract a range of bits from the specified input.
def StructInjectOp : RTLOp<"struct_inject", [NoSideEffect, 
                           AllTypesMatch<["input", "result"]>]> {
  let summary = "Inject a value into a named field of a struct.";

  let arguments = (ins StructType:$input, StrAttr:$field,
                   RTLNonInOutType:$newValue);
  let results = (outs StructType:$result);

  let parser = "return ::parse$cppClass(parser, result);";
  let printer = "return ::print(p, *this);";
}

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
  let hasFolder = true;
  let hasCanonicalizer = true;

  let builders = [
    OpBuilderDAG<(ins "ValueRange":$inputs)>,
    OpBuilderDAG<(ins "Value":$lhs, "Value":$rhs), [{
      return build($_builder, $_state, ValueRange{lhs, rhs});
    }]>
  ];
}

def ArraySliceOp : RTLOp<"array_slice", [NoSideEffect]> {
  let summary = "Get a range of values from an array";
  let description = [{
    Extracts a sub-range from an array. The range is from `lowIndex` to
    `lowIndex` + the number of elements in the return type, non-inclusive on
    the high end. For instance,

    ```
    // Slices 16 elements starting at '%offset'.
    %subArray = rtl.slice %largerArray at %offset :
        (!rtl.array<1024xi8>) -> !rtl.array<16xi8>
    ```

    Would translate to the following SystemVerilog:

    ```
    logic [7:0][15:0] subArray = largerArray[offset +: 16];
    ```

    Width of 'idx' is defined to be the precise number of bits required to
    index the 'input' array. More precisely: for an input array of size M,
    the width of 'idx' is ceil(log2(M)). Lower and upper bound indexes which
    are larger than the size of the 'input' array results in undefined
    behavior.
  }];

  let arguments = (ins ArrayType:$input, RTLIntegerType:$lowIndex);
  let results = (outs ArrayType:$dst);
  let verifier = [{
    unsigned inputSize = input().getType().cast<ArrayType>().getSize();
    if (llvm::Log2_64_Ceil(inputSize) !=
          lowIndex().getType().getIntOrFloatBitWidth())
      return failure();
    return success();
  }];

  let assemblyFormat = [{
    $input `at` $lowIndex attr-dict `:`
      `(` custom<SliceTypes>(type($input), type($lowIndex)) `)` `->` type($dst)
  }];

  let builders = [
    OpBuilderDAG<(ins "Value":$input, "size_t":$lowBit, "size_t":$size)>
  ];
}

def ArrayCreateOp : RTLOp<"array_create", [NoSideEffect, SameTypeOperands]> {
  let summary = "Create an array from values";
  let description = [{
    Creates an array from a variable set of values. One or more values must be
    listed.

    ```
    // %a, %b, %c are all i4
    %array = rtl.array_create %a, %b, %c : (i4)
    ```
  }];

  let arguments = (ins Variadic<RTLNonInOutType>:$inputs);
  let results = (outs ArrayType:$result);

  let verifier = [{
    unsigned returnSize = getType().cast<ArrayType>().getSize();
    if (inputs().size() != returnSize)
      return failure();
    return success();
  }];

  let printer = "return ::print(p, *this);";
  let parser = "return ::parse$cppClass(parser, result);";
  let builders = [
    // ArrayRef needs to contain at least one element.
    OpBuilderDAG<(ins "ArrayRef<::mlir::Value>":$input)>
  ];
}

// Extract a range of bits from the specified input.
def MuxOp : RTLOp<"mux",
 [NoSideEffect, AllTypesMatch<["trueValue", "falseValue", "result"]>]> {
  let summary = "Return one or the other operand depending on a selector bit";
  let description = [{
    ```
      %0 = mux %pred, %tvalue, %fvalue : i4
    ```
  }];

  let arguments = (ins I1:$cond, RTLIntegerType:$trueValue,
                       RTLIntegerType:$falseValue);
  let results = (outs RTLIntegerType:$result);

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
    The bitcast operation represents a bitwise reinerpretation (cast) of a
    value. This always synthesizes away in hardware, though if may or may not
    be syntactically represented in lowering or export language.

    Since bitcasting requires information on the bitwise layout of the types
    on which it operates, we discuss that here.

    ### Integer bit vectors

    MLIR's `IntegerType` with `Signless` semantics are used to represent bit
    vectors. They are never padded or aligned.

    ### Arrays

    The RTL dialect defines a custom `ArrayType`.

    - **Bit layout**: High index of array starts at the MSB. Array's 0th
      element's LSB located at array LSB.
    - **Padding between elements**: None
    - **Alignment of elements**: None

    ### Structs

    The RTL dialect defines a custom `StructType`.

    - **Bit layout**: The first listed member's MSB corresponds to the
      struct's MSB. The last member in the list shares its LSB with the
      struct.
    - **Padding between elements**: None
    - **Alignment of elements**: None

    ### Example figure

    ```
    15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0 
    -------------------------------------------------
    | MSB                                       LSB | 16 bit integer vector
    -------------------------------------------------
                             | MSB              LSB | 8 bit integer vector
    -------------------------------------------------
    | MSB      [1]       LSB | MSB     [0]      LSB | 2 element array of 8 bit integer vectors
    -------------------------------------------------

          13 12 11 10  9  8  7  6  5  4  3  2  1  0 
                                ---------------------
                                | MSB           LSB | 7 bit integer vector
          -------------------------------------------
          | MSB     [1]     LSB | MSB    [0]    LSB | 2 element array of 7 bit integer vectors
          -------------------------------------------
          | MSB a LSB | MSB b[1] LSB | MSB b[0] LSB | struct
          -------------------------------------------  a: 4 bit integral
                                                       b: 2 element array of 5 bit integer vectors
    ```
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
