//===- FIRRTLExpressions.td - FIRRTL expression ops --------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This describes the MLIR ops for FIRRTL expressions.
//
//===----------------------------------------------------------------------===//

include "circt/Dialect/HW/HWTypes.td"
include "circt/Types.td"

def SameOperandsIntTypeKind : NativeOpTrait<"SameOperandsIntTypeKind"> {
  let cppNamespace = "::circt::firrtl";
}

// A common base class for operations that implement type inference and parsed
// argument validation.
class FIRRTLExprOp<string mnemonic, list<OpTrait> traits = []> :
    FIRRTLOp<mnemonic, traits # [InferTypeOpInterface, NoSideEffect]> {

  // The narrow operation-specific type inference method. Operations can
  // override this with an inline declaration for the class header, or just
  // leave it as is and implement the function in a cpp file.
  code inferTypeDecl = [{
    /// Infer the return type of this operation.
    static FIRRTLType inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      Optional<Location> loc);
  }];

  // The operation-specific validator for a parsed list of operands and
  // constants. Operations can override this with an inline declaration for the
  // class header, or just leave it as is and implement the function in a cpp
  // file.
  code parseValidator = "";
  code parseValidatorDecl = [{
    /// Check that the parser has consumed the correct number of operands and
    /// constants.
    static LogicalResult validateArguments(ValueRange operands,
                                           ArrayRef<NamedAttribute> attrs,
                                           Location loc)
  }] # !if(!empty(parseValidator), ";", !subst("$_impl", parseValidator, [{ {
    return $_impl(operands, attrs, loc);
  } }]));

  // Additional class declarations to emit alongside the type inference.
  code firrtlExtraClassDeclaration = "";

  let extraClassDeclaration = firrtlExtraClassDeclaration # inferTypeDecl #
      parseValidatorDecl # [{
    /// Infer the return types of this operation. This is called by the
    /// `InferTypeOpInterface`. We simply forward to a narrower
    /// operation-specific implementation which is sufficient for FIRRTL ops.
    static LogicalResult inferReturnTypes(MLIRContext *context,
                                          Optional<Location> loc,
                                          ValueRange operands,
                                          DictionaryAttr attrs,
                                          mlir::RegionRange regions,
                                          SmallVectorImpl<Type> &results) {
      return impl::inferReturnTypes(context, loc, operands, attrs, regions,
        results, &inferReturnType);
    }

    /// Check that the parser has consumed the correct number of operands and
    /// constants, and infer the appropriate return type for the operation.
    static FIRRTLType validateAndInferReturnType(ValueRange operands,
                                                 ArrayRef<NamedAttribute> attrs,
                                                 Location loc) {
      if (failed(validateArguments(operands, attrs, loc)))
        return {};
      return inferReturnType(operands, attrs, loc);
    }
  }];
}

def ConstantOp : FIRRTLOp<"constant", [NoSideEffect, ConstantLike,
                                       FirstAttrDerivedResultType]> {
  let summary = "Produce a constant value";
  let description = [{
    The constant operation produces a constant value of SInt or UInt type, it
    never produces a zero bit wide result.
    ```
      %result = firrtl.constant 42 : t1
    ```
    }];

  let arguments = (ins APSIntAttr:$value);
  let results = (outs IntType:$result);

  // Need a custom parser/printer to avoid redundant type for the attribute and
  // the op itself.
  let parser = "return parse$cppClass(parser, result);";
  let printer = "print$cppClass(p, *this);";

  let builders = [
    OpBuilder<(ins "IntType":$type, "const APInt &":$value)>,
    OpBuilder<(ins "const APSInt &":$value)>
  ];
  let hasFolder = 1;
  let verifier = "return ::verifyConstantOp(*this);";
}

def SpecialConstantOp : FIRRTLOp<"specialconstant", [NoSideEffect,
                                                     ConstantLike]> {
  let summary = "Produce a constant Reset or Clock value";
  let description = [{
    The constant operation produces a constant value of Reset, AsyncReset, or
    Clock type. The value can only be 0 or 1.
    ```
      %result = firrtl.specialconstant 1 : !firrtl.clock
    ```
    }];

  let arguments = (ins BoolAttr:$value);
  let results = (outs
    AnyTypeOf<[ClockType, ResetType, AsyncResetType]>:$result);

  // Need a custom parser/printer to avoid redundant type for the attribute and
  // the op itself.
  let parser = "return parse$cppClass(parser, result);";
  let printer = "print$cppClass(p, *this);";
  let hasFolder = 1;
}

def InvalidValueOp : FIRRTLOp<"invalidvalue", [NoSideEffect, ConstantLike]> {
  let summary = "InvalidValue primitive";
  let description = [{
    The InvalidValue operation returns an invalid value of a specified type:
    ```
      %result = firrtl.invalid : !firrtl.uint<1>
    ```

    This corresponds to the FIRRTL invalidate operation without the implicit
    connect semantics.
  }];

  let arguments = (ins);
  let results = (outs FIRRTLType:$result);

  let hasFolder = 1;
  let assemblyFormat = "attr-dict `:` qualified(type($result))";
}

def SubfieldOp : FIRRTLExprOp<"subfield"> {
  let summary = "Extract a subfield of another value";
  let description = [{
    The subfield expression refers to a subelement of an expression with a
    bundle type.
    ```
      %result = firrtl.subfield %input "fieldIndex" : t1, t2
    ```
    }];

  let arguments = (ins BundleType:$input, I32Attr:$fieldIndex);
  let results = (outs FIRRTLType:$result);

  // TODO: Could drop the result type, inferring it from the source.
  let assemblyFormat =
    "$input `(` $fieldIndex `)` attr-dict `:` functional-type($input, $result)";

  let builders = [
    OpBuilder<(ins "Value":$input, "StringRef":$fieldName), [{
      auto bundleType = input.getType().cast<BundleType>();
      auto fieldIndex = bundleType.getElementIndex(fieldName);
      assert(fieldIndex.hasValue() && "subfield operation to unknown field");
      return build($_builder, $_state, input, *fieldIndex);
    }]>
  ];

  let firrtlExtraClassDeclaration = [{
    /// Return true if the specified field is flipped.
    bool isFieldFlipped();

    /// Return a `FieldRef` to the accessed field.
    FieldRef getAccessedField() {
      return FieldRef(input(), input().getType().cast<BundleType>()
                                      .getFieldID(fieldIndex()));
    }
  }];

  let verifier = "return ::verifySubfieldOp(*this);";
}

class IndexConstraint<string value, string resultValue>
  : TypesMatchWith<"type should be element of vector type",
                   value, resultValue,
                   "firrtl::getVectorElementType($_self)">;

def SubindexOp : FIRRTLExprOp<"subindex", [
    IndexConstraint<"input", "result">
  ]> {
  let summary = "Extract an element of a vector value";
  let description = [{
    The subindex expression statically refers, by index, to a subelement
    of an expression with a vector type. The index must be a non-negative
    integer and cannot be equal to or exceed the length of the vector it
    indexes.
    ```
      %result = firrtl.subindex %input[index] : t1
    ```
    }];

  let arguments = (ins FIRRTLType:$input, I32Attr:$index);
  let results = (outs FIRRTLType:$result);

  // TODO: Could drop the result type, inferring it from the source.
  let assemblyFormat =
     "$input `[` $index `]` attr-dict `:` qualified(type($input))";

  let firrtlExtraClassDeclaration = [{
    /// Return a `FieldRef` to the accessed field.
    FieldRef getAccessedField() {
      return FieldRef(input(), input().getType().cast<FVectorType>()
                                      .getFieldID(index()));
    }
  }];
}

def SubaccessOp : FIRRTLExprOp<"subaccess", [
    IndexConstraint<"input", "result">
  ]> {
  let summary = "Extract a dynamic element of a vector value";
  let description = [{
    The subaccess expression dynamically refers to a subelement of a
    vector-typed expression using a calculated index. The index must be an
    expression with an unsigned integer type.
    ```
      %result = firrtl.subaccess %input[%idx] : t1, t2
    ```
    }];

  let arguments = (ins FIRRTLType:$input, UIntType:$index);
  let results = (outs FIRRTLType:$result);

  let assemblyFormat =
     "$input `[` $index `]` attr-dict `:` qualified(type($input)) `,` qualified(type($index))";

  let hasCanonicalizeMethod = true;
}

def MultibitMuxOp : FIRRTLExprOp<"multibit_mux"> {
  let summary = "Multibit multiplexer";
  let description = [{
    The multibit mux expression dynamically selects operands. The
    index must be an expression with an unsigned integer type.
    ```
      %result = firrtl.multibit_mux %index,
                %v_{n-1}, ..., %v_2, %v_1, %v_0  : t1, t2
    ```
    The order of operands is defined in the same way as hw dialect.
    For the example above, if `%index` is 0, then the value is `%v_0`.
    }];

  let arguments = (ins FIRRTLType:$index, Variadic<FIRRTLType>:$inputs);
  let results = (outs FIRRTLType:$result);

  // Need a custom parser/printer to emit operands type only once.
  let parser = "return parse$cppClass(parser, result);";
  let printer = "print$cppClass(p, *this);";
  let hasFolder = true;
  let hasCanonicalizeMethod = true;
}

//===----------------------------------------------------------------------===//
// Primitive Operations
//===----------------------------------------------------------------------===//

/// PrimOp is a FIRRTLOp without side effects.
class PrimOp<string mnemonic, list<OpTrait> traits = []> :
    FIRRTLExprOp<mnemonic, traits> {
  let hasFolder = 1;
}

//===----------------------------------------------------------------------===//
// Binary Operations
//===----------------------------------------------------------------------===//

// Base class for binary primitive operators.
class BinaryPrimOp<string mnemonic, Type lhsType, Type rhsType, Type resultType,
                   list<OpTrait> traits = []> : PrimOp<mnemonic, traits> {
  let arguments = (ins lhsType:$lhs, rhsType:$rhs);
  let results = (outs resultType:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs  attr-dict `:`
       `(` qualified(type($lhs)) `,` qualified(type($rhs)) `)` `->` qualified(type($result))
  }];

  // Give concrete operations a chance to set a type inference callback. If left
  // empty, a declaration for `inferBinaryReturnType` will be emitted that the
  // operation is expected to implement.
  code inferType = "";
  let inferTypeDecl = !if(!empty(inferType), [{
    /// Infer the return type of this binary operation.
    static FIRRTLType inferBinaryReturnType(FIRRTLType lhs, FIRRTLType rhs,
                                            Optional<Location> loc);
    }], "") # !subst("$_infer", !if(!empty(inferType), "inferBinaryReturnType",
      inferType), [{
    /// Infer the return type of this operation.
    static FIRRTLType inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      Optional<Location> loc) {
      return $_infer(operands[0].getType().cast<FIRRTLType>(),
                     operands[1].getType().cast<FIRRTLType>(),
                     loc);
    }
  }]);
  let parseValidator = "impl::validateBinaryOpArguments";
}

// A binary operation on two integer-typed arguments of the same kind.
class IntBinaryPrimOp<string mnemonic, Type resultType,
                      list<OpTrait> traits = []> :
   BinaryPrimOp<mnemonic, IntType, IntType, resultType,
                traits # [SameOperandsIntTypeKind]>;

let inferType = "impl::inferAddSubResult" in {
  let hasCanonicalizeMethod = true in {
    def AddPrimOp : IntBinaryPrimOp<"add", IntType, [Commutative]>;
    def SubPrimOp : IntBinaryPrimOp<"sub", IntType>;
  }
}

def MulPrimOp : IntBinaryPrimOp<"mul", IntType, [Commutative]>;
def DivPrimOp : IntBinaryPrimOp<"div", IntType> {
  let description = [{
    Divides the first argument (the numerator) by the second argument
    (the denominator) truncating the result (rounding towards zero).

    **If the denominator is zero, the result is undefined.**

    The compiler may optimize this undefined behavior in any way it
    wants. Notably `div(a, a)` will be optimized to `1`. This may cause
    erroneous formal equivalence mismatches between unoptimized and
    optimized FIRRTL dialects that are separately converted to Verilog.
  }];
}
def RemPrimOp : IntBinaryPrimOp<"rem", IntType>;
let inferType = "impl::inferBitwiseResult" in {
  def AndPrimOp : IntBinaryPrimOp<"and", UIntType, [Commutative]>;
  def OrPrimOp  : IntBinaryPrimOp<"or",  UIntType, [Commutative]>;
  def XorPrimOp : IntBinaryPrimOp<"xor", UIntType, [Commutative]>;
}

// Comparison Operations
let inferType = "impl::inferComparisonResult" in {
  let hasCanonicalizer = true in {
    def LEQPrimOp : IntBinaryPrimOp<"leq", UInt1Type>;
    def LTPrimOp  : IntBinaryPrimOp<"lt",  UInt1Type>;
    def GEQPrimOp : IntBinaryPrimOp<"geq", UInt1Type>;
    def GTPrimOp  : IntBinaryPrimOp<"gt",  UInt1Type>;
  }
  let hasCanonicalizeMethod = true in {
    def EQPrimOp  : IntBinaryPrimOp<"eq",  UInt1Type, [Commutative]>;
    def NEQPrimOp : IntBinaryPrimOp<"neq", UInt1Type, [Commutative]>;
  }
}

def CatPrimOp : IntBinaryPrimOp<"cat", UIntType> {
  let hasCanonicalizeMethod = true;
}
def DShlPrimOp  : BinaryPrimOp<"dshl", IntType, UIntType, IntType>  {
  let description = [{
    A dynamic shift left operation. The width of `$result` is expanded to
    `width($lhs) + 1 << width($rhs) - 1`.
  }];
  let hasCanonicalizeMethod = true;
}
def DShlwPrimOp : BinaryPrimOp<"dshlw", IntType, UIntType, IntType> {
  let description = [{
    A dynamic shift left operation same as 'dshl' but with different width rule.
    The width of `$result` is equal to `$lhs`.
  }];
}
def DShrPrimOp  : BinaryPrimOp<"dshr", IntType, UIntType, IntType> {
  let hasCanonicalizeMethod = true;
}

//===----------------------------------------------------------------------===//
// Unary Operations
//===----------------------------------------------------------------------===//

// Base class for binary primitive operators.
class UnaryPrimOp<string mnemonic, Type srcType, Type resultType,
                  list<OpTrait> traits = []> : PrimOp<mnemonic, traits> {
  let arguments = (ins srcType:$input);
  let results = (outs resultType:$result);

  let assemblyFormat =
    "$input attr-dict `:` functional-type($input, $result)";

  // Give concrete operations a chance to set a type inference callback. If left
  // empty, a declaration for `inferUnaryReturnType` will be emitted that the
  // operation is expected to implement.
  code inferType = "";
  let inferTypeDecl = !if(!empty(inferType), [{
    /// Infer the return type of this unary operation.
    static FIRRTLType inferUnaryReturnType(FIRRTLType arg,
                                           Optional<Location> loc);
    }], "") # !subst("$_infer", !if(!empty(inferType), "inferUnaryReturnType",
      inferType), [{
    /// Infer the return type of this operation.
    static FIRRTLType inferReturnType(ValueRange operands,
                                      ArrayRef<NamedAttribute> attrs,
                                      Optional<Location> loc) {
      return $_infer(operands[0].getType().cast<FIRRTLType>(), loc);
    }
  }]);

  let parseValidator = "impl::validateUnaryOpArguments";
}

def AsSIntPrimOp : UnaryPrimOp<"asSInt", FIRRTLType, SIntType>;
def AsUIntPrimOp : UnaryPrimOp<"asUInt", FIRRTLType, UIntType>;
def AsAsyncResetPrimOp
  : UnaryPrimOp<"asAsyncReset", OneBitCastableType, AsyncResetType>;
def AsClockPrimOp : UnaryPrimOp<"asClock", OneBitCastableType, ClockType>;
def CvtPrimOp : UnaryPrimOp<"cvt", IntType, SIntType>;
def NegPrimOp : UnaryPrimOp<"neg", IntType, SIntType>;
def NotPrimOp : UnaryPrimOp<"not", IntType, UIntType>;

let inferType = "impl::inferReductionResult" in {
  def AndRPrimOp : UnaryPrimOp<"andr", IntType, UInt1Type> {
    let description = [{
      Horizontally reduce a value to one bit, using the 'and' operation to merge
      bits.  `andr(x)` is equivalent to `concat(x, 1b1) == ~0`.  As such, it
      returns 1 for zero-bit-wide operands.
    }];
  }
  def OrRPrimOp : UnaryPrimOp<"orr", IntType, UInt1Type> {
    let description = [{
      Horizontally reduce a value to one bit, using the 'or' operation to merge
      bits.  `orr(x)` is equivalent to `concat(x, 1b0) != 0`.  As such, it
      returns 0 for zero-bit-wide operands.
    }];
  }
  def XorRPrimOp : UnaryPrimOp<"xorr", IntType, UInt1Type> {
    let description = [{
      Horizontally reduce a value to one bit, using the 'xor' operation to merge
      bits.  `xorr(x)` is equivalent to `popcount(concat(x, 1b0)) & 1`.  As
      such, it returns 0 for zero-bit-wide operands.
    }];
  }
}

//===----------------------------------------------------------------------===//
// Other Operations
//===----------------------------------------------------------------------===//

def BitsPrimOp : PrimOp<"bits"> {
  let arguments = (ins IntType:$input, I32Attr:$hi, I32Attr:$lo);
  let results = (outs UIntType:$result);

  let assemblyFormat =
    "$input $hi `to` $lo attr-dict `:` functional-type($input, $result)";

  let description = [{
    The `bits` operation extracts the bits between `hi` (inclusive) and `lo`
    (inclusive) from `input`.  `hi` must be greater than or equal to `lo`. Both
    `hi` and `lo` must be non-negative and less than the bit width of `input`.
    The result is `hi - lo + 1` bits wide.
  }];

  let hasCanonicalizeMethod = true;
}

def HeadPrimOp : PrimOp<"head"> {
  let arguments = (ins IntType:$input, I32Attr:$amount);
  let results = (outs UIntType:$result);

  let assemblyFormat =
    "$input `,` $amount attr-dict `:` functional-type($input, $result)";

  let hasCanonicalizeMethod = true;
  let parseValidator = "impl::validateOneOperandOneConst";
}

def MuxPrimOp : PrimOp<"mux"> {
  let arguments = (ins UInt1Type:$sel, PassiveType:$high, PassiveType:$low);
  let results = (outs PassiveType:$result);

  let assemblyFormat =
    "`(` operands `)` attr-dict `:` functional-type(operands, $result)";

  let hasCanonicalizer = true;
}

def PadPrimOp : PrimOp<"pad"> {
  let arguments = (ins IntType:$input, I32Attr:$amount);
  let results = (outs IntType:$result);

  let assemblyFormat =
    "$input `,` $amount attr-dict `:` functional-type($input, $result)";

  let description = [{
    Pad the input out to an `amount` wide integer, sign extending or zero
    extending according to `input`s type.  If `amount` is less than the existing
    width of `input`, then input is unmodified.
  }];

  let parseValidator = "impl::validateOneOperandOneConst";
}

class ShiftPrimOp<string mnemonic> : PrimOp<mnemonic> {
  let arguments = (ins IntType:$input, I32Attr:$amount);
  let results = (outs IntType:$result);

  let assemblyFormat =
    "$input `,` $amount attr-dict `:` functional-type($input, $result)";

  let parseValidator = "impl::validateOneOperandOneConst";
}

def ShlPrimOp : ShiftPrimOp<"shl"> {
  let description = [{
    The `shl` operation concatenates `amount` zero bits to the least significant
    end of `input`. `amount` must be non-negative.
  }];
}

def ShrPrimOp : ShiftPrimOp<"shr"> {
  let description = [{
    The `shr` operation truncates least significant `amount` bits from `input`.
    If `amount` is greater than of equal to `width(input)`, the value will be
    zero for unsigned types and the sign bit for signed types. `amount` must be
    non-negative.
  }];

  let hasCanonicalizeMethod = true;
}

def TailPrimOp : PrimOp<"tail"> {
  let arguments = (ins IntType:$input, I32Attr:$amount);
  let results = (outs UIntType:$result);

  let assemblyFormat =
    "$input `,` $amount attr-dict `:` functional-type($input, $result)";

  let description = [{
    The `tail` operation truncates the `amount` most significant bits from
    `input`.  `amount` must be non-negative and less than or equal to the bit
    width of e. The result is `width(input)-amount` bits wide.
  }];

  let hasCanonicalizeMethod = true;
  let parseValidator = "impl::validateOneOperandOneConst";
}

//===----------------------------------------------------------------------===//
// Verbatim
//===----------------------------------------------------------------------===//

def HasCustomSSAName : DeclareOpInterfaceMethods<OpAsmOpInterface,
                         ["getAsmResultNames"]>;

def VerbatimExprOp : FIRRTLOp<"verbatim.expr",
                              [NoSideEffect, HasCustomSSAName]> {
  let summary = "Expression that expands to a value given SystemVerilog text";
  let description = [{
    This operation produces a typed value expressed by a string of
    SystemVerilog.  This can be used to access macros and other values that are
    only sensible as Verilog text.

    The text string is expected to have the highest precedence, so you should
    include parentheses in the string if it isn't a single token.  This is also
    assumed to not have side effects (use `sv.verbatim` if you need them).

    `firrtl.verbatim.expr` allows operand substitutions with `{{0}}` syntax.
  }];

  let arguments = (ins StrAttr:$text, Variadic<AnyType>:$operands,
                       DefaultValuedAttr<NameRefArrayAttr,"{}">:$symbols);
  let results = (outs FIRRTLType:$result);
  let assemblyFormat = [{
    $text (`(` $operands^ `)`)?
    `:` functional-type($operands, $result) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Twine":$text,
                   CArg<"ValueRange", "{}">:$operands,
                   CArg<"ArrayRef<Attribute>", "{}">:$symbols), [{
      build(odsBuilder, odsState, resultType, odsBuilder.getStringAttr(text),
            operands, odsBuilder.getArrayAttr(symbols));
    }]>
  ];
}

def VerbatimWireOp : FIRRTLOp<"verbatim.wire",
                              [NoSideEffect, HasCustomSSAName]> {
  let summary = "Expression with wire semantics that expands to a value given "
                "SystemVerilog text";
  let description = [{
    This operation produces a typed value with wire semantics, expressed by a
    string of SystemVerilog. This can be used to access macros and other values
    that are only sensible as Verilog text.

    The text string is expected to have the highest precedence, so you should
    include parentheses in the string if it isn't a single token. This is also
    assumed to not have side effects (use `sv.verbatim` if you need them).

    `firrtl.verbatim.wire` allows operand substitutions with `{{0}}` syntax.
  }];

  let arguments = (ins StrAttr:$text, Variadic<AnyType>:$operands,
                       DefaultValuedAttr<NameRefArrayAttr,"{}">:$symbols);
  let results = (outs FIRRTLType:$result);
  let assemblyFormat = [{
    $text (`(` $operands^ `)`)?
    `:` functional-type($operands, $result) attr-dict
  }];

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Twine":$text,
                   CArg<"ValueRange", "{}">:$operands,
                   CArg<"ArrayRef<Attribute>", "{}">:$symbols), [{
      build(odsBuilder, odsState, resultType, odsBuilder.getStringAttr(text),
            operands, odsBuilder.getArrayAttr(symbols));
    }]>
  ];
}

//===----------------------------------------------------------------------===//
// Conversions to/from fixed-width signless integer types in standard dialect.
//===----------------------------------------------------------------------===//

// This operation converts from a struct to a bundle
// type, or visa-versa.  FIRRTL source/destination types must be passive.
def HWStructCastOp : FIRRTLOp<"hwStructCast", [NoSideEffect]> {
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$result);

  let assemblyFormat =
    "$input attr-dict `:` functional-type($input, $result)";

  let verifier = "return ::verifyHWStructCastOp(*this);";
}

def BitCastOp: FIRRTLOp<"bitcast", [NoSideEffect]> {
  let summary = [{
    Reinterpret one value to another value of the same size and
    potentially different type. This op is lowered to hw::BitCastOp.
  }];

  let arguments = (ins FIRRTLType:$input);
  let results = (outs FIRRTLType:$result);
  let verifier = "return ::verifyBitCastOp(*this);";
  let hasFolder = 1;

  let assemblyFormat = "$input attr-dict `:` functional-type($input, $result)";
}
