//===- SMTBitVectorOps.td - SMT bit-vector dialect ops -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_SMT_SMTBITVECTOROPS_TD
#define CIRCT_DIALECT_SMT_SMTBITVECTOROPS_TD

include "circt/Dialect/SMT/SMTDialect.td"
include "circt/Dialect/SMT/SMTAttributes.td"
include "circt/Dialect/SMT/SMTTypes.td"
include "mlir/IR/EnumAttr.td"
include "mlir/IR/OpAsmInterface.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

class SMTBVOp<string mnemonic, list<Trait> traits = []> :
  Op<SMTDialect, "bv." # mnemonic, traits>;

def BVConstantOp : SMTBVOp<"constant", [
  Pure,
  ConstantLike,
  FirstAttrDerivedResultType,
  DeclareOpInterfaceMethods<InferTypeOpInterface, ["inferReturnTypes"]>,
  DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
]> {
  let summary = "produce a constant bit-vector";
  let description = [{
    This operation produces an SSA value equal to the bit-vector constant
    specified by the 'value' attribute.
    Refer to the `BitVectorAttr` documentation for more information about
    the semantics of bit-vector constants, their format, and associated sort.
    The result type always matches the attribute's type.

    Examples:
    ```mlir
    %c92_bv8 = smt.bv.constant #smt.bv<92> : !smt.bv<8>
    %c5_bv4 = smt.bv.constant #smt.bv<5> : !smt.bv<4>
    ```
  }];

  let arguments = (ins BitVectorAttr:$value);
  let results = (outs BitVectorType:$result);

  let assemblyFormat = "qualified($value) attr-dict";

  let builders = [
    OpBuilder<(ins "const llvm::APInt &":$value), [{
      build($_builder, $_state,
            BitVectorAttr::get($_builder.getContext(), value));
    }]>,
    OpBuilder<(ins "unsigned":$value, "unsigned":$width), [{
      build($_builder, $_state,
            BitVectorAttr::get($_builder.getContext(), value, width));
    }]>,
  ];

  let hasFolder = true;
}

class BVArithmeticOrBitwiseOp<string mnemonic, string desc> :
    SMTBVOp<mnemonic, [Pure, SameOperandsAndResultType]> {
  let summary = "equivalent to bv" # mnemonic # " in SMT-LIB";
  let description = "This operation performs " # desc # [{. The semantics are
    equivalent to the `bv}] # mnemonic # [{` operator defined in the SMT-LIB 2.6
    standard. More precisely in the [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
  }];

  let results = (outs BitVectorType:$result);
}

class BinaryBVOp<string mnemonic, string desc> :
     BVArithmeticOrBitwiseOp<mnemonic, desc> {
  let arguments = (ins BitVectorType:$lhs, BitVectorType:$rhs);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` qualified(type($result))";
}

class UnaryBVOp<string mnemonic, string desc> :
    BVArithmeticOrBitwiseOp<mnemonic, desc> {
  let arguments = (ins BitVectorType:$input);
  let assemblyFormat = "$input attr-dict `:` qualified(type($result))";
}

def BVNotOp  : UnaryBVOp<"not", "bitwise negation">;
def BVNegOp  : UnaryBVOp<"neg", "two's complement unary minus">;

def BVAndOp  : BinaryBVOp<"and", "bitwise AND">;
def BVOrOp   : BinaryBVOp<"or", "bitwise OR">;
def BVXOrOp  : BinaryBVOp<"xor", "bitwise exclusive OR">;

def BVAddOp  : BinaryBVOp<"add", "addition">;
def BVMulOp  : BinaryBVOp<"mul", "multiplication">;
def BVUDivOp : BinaryBVOp<"udiv", "unsigned division (rounded towards zero)">;
def BVSDivOp : BinaryBVOp<"sdiv", "two's complement signed division">;
def BVURemOp : BinaryBVOp<"urem", "unsigned remainder">;
def BVSRemOp : BinaryBVOp<"srem",
  "two's complement signed remainder (sign follows dividend)">;
def BVSModOp : BinaryBVOp<"smod",
  "two's complement signed remainder (sign follows divisor)">;
def BVShlOp  : BinaryBVOp<"shl", "shift left">;
def BVLShrOp : BinaryBVOp<"lshr", "logical shift right">;
def BVAShrOp : BinaryBVOp<"ashr", "arithmetic shift right">;

def PredicateSLT : I64EnumAttrCase<"slt", 0>;
def PredicateSLE : I64EnumAttrCase<"sle", 1>;
def PredicateSGT : I64EnumAttrCase<"sgt", 2>;
def PredicateSGE : I64EnumAttrCase<"sge", 3>;
def PredicateULT : I64EnumAttrCase<"ult", 4>;
def PredicateULE : I64EnumAttrCase<"ule", 5>;
def PredicateUGT : I64EnumAttrCase<"ugt", 6>;
def PredicateUGE : I64EnumAttrCase<"uge", 7>;
let cppNamespace = "circt::smt" in
def BVCmpPredicate : I64EnumAttr<
    "BVCmpPredicate",
    "smt bit-vector comparison predicate",
    [PredicateSLT, PredicateSLE, PredicateSGT, PredicateSGE,
     PredicateULT, PredicateULE, PredicateUGT, PredicateUGE]>;

def BVCmpOp : SMTBVOp<"cmp", [Pure, SameTypeOperands]> {
  let summary = "compare bit-vectors interpreted as signed or unsigned";
  let description = [{
    This operation compares bit-vector values, interpreting them as signed or
    unsigned values depending on the predicate. The semantics are equivalent to
    the `bvslt`, `bvsle`, `bvsgt`, `bvsge`, `bvult`, `bvule`, `bvugt`, or
    `bvuge` operator defined in the SMT-LIB 2.6 standard depending on the
    specified predicate. More precisely in the
    [theory of FixedSizeBitVectors](https://smtlib.cs.uiowa.edu/Theories/FixedSizeBitVectors.smt2)
    and the [QF_BV logic](https://smtlib.cs.uiowa.edu/Logics/QF_BV.smt2)
    describing closed quantifier-free formulas over the theory of fixed-size
    bit-vectors.
  }];

  let arguments = (ins BVCmpPredicate:$pred,
                       BitVectorType:$lhs,
                       BitVectorType:$rhs);
  let results = (outs BoolType:$result);

  let assemblyFormat = [{
    $pred $lhs `,` $rhs attr-dict `:` qualified(type($lhs))
  }];
}

#endif // CIRCT_DIALECT_SMT_SMTBITVECTOROPS_TD
