//===- FIRRTLCanonicalization.td - FIRRTL canonicalization -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// The canonicalization patterns for the FIRRTL dialect.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_FIRRTLCANONICALIZATION_TD
#define CIRCT_DIALECT_FIRRTL_FIRRTLCANONICALIZATION_TD

include "FIRRTLDeclarations.td"
include "FIRRTLExpressions.td"
include "FIRRTLStatements.td"
include "FIRRTLIntrinsics.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/PatternBase.td"

/// Constraint that matches a ConstantOp, SpecialConstantOp, or AggregateConstantOp.
def AnyConstantOp : Constraint<CPred<[{
  isa_and_nonnull<ConstantOp, SpecialConstantOp, AggregateConstantOp>
    ($0.getDefiningOp())
  }]>>;

/// Constraint that matches non-constant operations. Used to ensure that the
/// const-on-LHS rewriting patterns converge in case both operands are constant.
def NonConstantOp : Constraint<CPred<[{
  !isa_and_nonnull<ConstantOp, SpecialConstantOp, AggregateConstantOp>
    ($0.getDefiningOp())
  }]>>;

// Constraint that matches empty attribute dictionaries.  Used to ensure that
// there are no FIRRTL annotation on an operation.
def EmptyAttrDict : Constraint<CPred<"$0.empty()">>;

def GetDroppableNameAttr : NativeCodeCall<
  "NameKindEnumAttr::get($_builder.getContext(), NameKindEnum::DroppableName) ">;
def NonEmptyAttr : Constraint<CPred<"!$0.getValue().empty()">>;
def NullAttr : Constraint<CPred<"!$0">>;

// Constraint that enforces equal types
def EqualTypes : Constraint<CPred<"areAnonymousTypesEquivalent($0.getType(), $1.getType())">>;

// Constraint that enforces equal type sizes
def EqualIntSize : Constraint<CPred<"type_cast<IntType>($0.getType()).getWidth() == type_cast<IntType>($1.getType()).getWidth()">>;

// Constraint that enforces equal type signedness
def EqualSigns : Constraint<CPred<"type_cast<IntType>($0.getType()).isSigned() == type_cast<IntType>($1.getType()).isSigned()">>;

// sizeof(0) >= sizeof(1)
def IntTypeWidthGEQ32 : Constraint<CPred<
  "type_cast<IntType>($0.getType()).getBitWidthOrSentinel() >= type_cast<IntType>($1.getType()).getBitWidthOrSentinel()">>;

// sizeof(0) > sizeof(1)
def IntTypeWidthGT32 : Constraint<CPred<
  "type_cast<IntType>($0.getType()).getBitWidthOrSentinel() > type_cast<IntType>($1.getType()).getBitWidthOrSentinel()">>;

// sizeof(0) < X
class IntTypeWidthLTX<int X> : Constraint<CPred<
  "type_cast<IntType>($0.getType()).getBitWidthOrSentinel() >= 0 &&"
  "type_cast<IntType>($0.getType()).getBitWidthOrSentinel()< " # X
>>;

// Constraint that enforces int types
def IntTypes : Constraint<CPred<"type_isa<IntType>($0.getType())">>;

// Constraint that enforces sint types
def SIntTypes : Constraint<CPred<"type_isa<SIntType>($0.getType())">>;

// Constraint that enforces uint types
def UIntTypes : Constraint<CPred<"type_isa<UIntType>($0.getType())">>;

// Constraint that enforces types mismatches
def mismatchTypes : Constraint<CPred<"!areAnonymousTypesEquivalent($0.getType(), $1.getType())">>;

// Constraint that enforces types of known width
def KnownWidth : Constraint<CPred<[{
  type_isa<FIRRTLBaseType>($0.getType()) &&
    !type_cast<FIRRTLBaseType>($0.getType()).getRecursiveTypeProperties().hasUninferredWidth
  }]>>;

/// Constraint that matches a false SpecialConstantOp
def FalseSpecialConstantOp : Constraint<CPred<
  "$0.getDefiningOp<SpecialConstantOp>() &&"
  "$0.getDefiningOp<SpecialConstantOp>().getValue() == false">>;

/// Constraint that matches a true SpecialConstantOp
def TrueSpecialConstantOp : Constraint<CPred<
  "$0.getDefiningOp<SpecialConstantOp>() &&"
  "$0.getDefiningOp<SpecialConstantOp>().getValue() == true">>;

/// Constraint that matches a zero ConstantOp or SpecialConstantOp.
def ZeroConstantOp : Constraint<Or<[
  CPred<"$0.getDefiningOp<ConstantOp>() &&"
        "$0.getDefiningOp<ConstantOp>().getValue().isZero()">,
  FalseSpecialConstantOp.predicate
]>>;

/// Constraint that matches a one ConstantOp or SpecialConstantOp.
def OneConstantOp : Constraint<Or<[
  CPred<"$0.getDefiningOp<ConstantOp>() &&"
        "$0.getDefiningOp<ConstantOp>().getValue().isOne()">,
  TrueSpecialConstantOp.predicate
]>>;

/// Constraint that matches a not-all-ones ConstantOp or SpecialConstantOp.
def NotAllOneConstantOp : Constraint<Or<[
  CPred<"$0.getDefiningOp<ConstantOp>() &&"
        "!$0.getDefiningOp<ConstantOp>().getValue().isAllOnes()">,
  FalseSpecialConstantOp.predicate
]>>;

/// Constraint that matches an all ones ConstantOp.
def AllOneConstantOp : Constraint<CPred<"$0.getDefiningOp<ConstantOp>() && $0.getDefiningOp<ConstantOp>().getValue().isAllOnes()">>;

// Delta between type widths
def TypeWidthAdjust32 : NativeCodeCall<
  "$_builder.getI32IntegerAttr(type_cast<FIRRTLBaseType>($0.getType()).getBitWidthOrSentinel() - type_cast<FIRRTLBaseType>($1.getType()).getBitWidthOrSentinel())">;

// Int 1 is one higher than int 2
def AdjInt : Constraint<CPred<"$0.getValue() + 1 == $1.getValue()">>;

/// Drop the writer to the first argument and passthrough the second
def DropWrite : NativeCodeCall<"dropWrite($_builder, $0, $1)">;

def MoveNameHint : NativeCodeCall<"moveNameHint($0, $1)">;

def GetEmptyString : NativeCodeCall<
  "StringAttr::get($_builder.getContext(), {}) ">;

// leq(const, x) -> geq(x, const)
def LEQWithConstLHS : Pat<
  (LEQPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (GEQPrimOp $rhs, $lhs)),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// lt(const, x) -> gt(x, const)
def LTWithConstLHS : Pat<
  (LTPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (GTPrimOp $rhs, $lhs)),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// geq(const, x) -> leq(x, const)
def GEQWithConstLHS : Pat<
  (GEQPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (LEQPrimOp $rhs, $lhs)),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// gt(const, x) -> lt(x, const)
def GTWithConstLHS : Pat<
  (GTPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (LTPrimOp $rhs, $lhs)),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// not(not(x)) -> x
def NotNot : Pat <
  (NotPrimOp:$old (NotPrimOp $x)),
  (MoveNameHint $old, (AsUIntPrimOp $x)),
  []>;

// not(eq(x,y)) -> neq(x,y)
def NotEq : Pat <
  (NotPrimOp:$old (EQPrimOp $x, $y)),
  (MoveNameHint $old, (NEQPrimOp $x, $y)),
  []>;

// not(neq(x,y)) -> eq(x,y)
def NotNeq : Pat <
  (NotPrimOp:$old (NEQPrimOp $x, $y)),
  (MoveNameHint $old, (EQPrimOp $x, $y)),
  []>;

// not(leq(x,y)) -> gt(x,y)
def NotLeq : Pat <
  (NotPrimOp:$old (LEQPrimOp $x, $y)),
  (MoveNameHint $old, (GTPrimOp $x, $y)),
  []>;

// not(lt(x,y)) -> geq(x,y)
def NotLt : Pat <
  (NotPrimOp:$old (LTPrimOp $x, $y)),
  (MoveNameHint $old, (GEQPrimOp $x, $y)),
  []>;

// not(geq(x,y)) -> lt(x,y)
def NotGeq : Pat <
  (NotPrimOp:$old (GEQPrimOp $x, $y)),
  (MoveNameHint $old, (LTPrimOp $x, $y)),
  []>;

// not(gt(x,y)) -> leq(x,y)
def NotGt : Pat <
  (NotPrimOp:$old (GTPrimOp $x, $y)),
  (MoveNameHint $old, (LEQPrimOp $x, $y)),
  []>;

// connect(a:t,b:t) -> matchingconnect(a,b)
def ConnectSameType : Pat<
  (ConnectOp $dst, $src),
  (MatchingConnectOp $dst, $src),
  [(EqualTypes $dst, $src), (KnownWidth $dst)]>;

// connect(a:t1,b:(t2<t1) -> matchingconnect(a,(pad b))
def ConnectExtension : Pat<
  (ConnectOp $dst, $src),
  (MatchingConnectOp $dst, (PadPrimOp $src,
    (NativeCodeCall<"type_cast<FIRRTLBaseType>($0.getType()).getBitWidthOrSentinel()"> $dst))),
  [(IntType $dst), (IntType $src), (KnownWidth $dst), (KnownWidth $src), (mismatchTypes $src, $dst)]>;

def LimitConstant32 : NativeCodeCall<
  "$_builder.getI32IntegerAttr($0.getValue().getLimitedValue(1ULL << 31))">;

def TypeWidth32 : NativeCodeCall<
  "$_builder.getI32IntegerAttr(type_cast<FIRRTLBaseType>($0.getType()).getBitWidthOrSentinel())">;

// dshl(a, const) -> shl(a, const)
def DShlOfConstant : Pat<
  (DShlPrimOp:$old $x, (ConstantOp $cst)),
  (PadPrimOp (MoveNameHint $old, (ShlPrimOp $x, (LimitConstant32 $cst))), (TypeWidth32 $old)),
  [(KnownWidth $x)]>;

// dshr(a, const) -> shr(a, const)
def DShrOfConstant : Pat<
  (DShrPrimOp:$old $x, (ConstantOp $cst)),
  (PadPrimOp (MoveNameHint $old, (ShrPrimOp $x, (LimitConstant32 $cst))), (TypeWidth32 $old)),
  [(KnownWidth $x)]>;

////////////////////////////////////////////////////////////////////////////////
// Extend arguments to make types match
////////////////////////////////////////////////////////////////////////////////

def extendAnd : Pat <
  (AndPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (AndPrimOp (PadPrimOp $lhs, (TypeWidth32 $old)),
                                 (PadPrimOp $rhs, (TypeWidth32 $old)))),
  [(KnownWidth $lhs), (KnownWidth $rhs), (mismatchTypes $lhs, $rhs)]>;

def extendOr : Pat <
  (OrPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (OrPrimOp (PadPrimOp $lhs, (TypeWidth32 $old)),
                                (PadPrimOp $rhs, (TypeWidth32 $old)))),
  [(KnownWidth $lhs), (KnownWidth $rhs), (mismatchTypes $lhs, $rhs)]>;

def extendXor : Pat <
  (XorPrimOp:$old $lhs, $rhs),
  (MoveNameHint $old, (XorPrimOp (PadPrimOp $lhs, (TypeWidth32 $old)),
                                 (PadPrimOp $rhs, (TypeWidth32 $old)))),
  [(KnownWidth $lhs), (KnownWidth $rhs), (mismatchTypes $lhs, $rhs)]>;

////////////////////////////////////////////////////////////////////////////////
// Move constants to the right
////////////////////////////////////////////////////////////////////////////////

def moveConstAdd : Pat <
  (AddPrimOp:$old $x, $y),
  (MoveNameHint $old, (AddPrimOp $y, $x)),
  [(AnyConstantOp $x), (NonConstantOp $y)]>;

def moveConstAnd : Pat <
  (AndPrimOp:$old $x, $y),
  (MoveNameHint $old, (AndPrimOp $y, $x)),
  [(AnyConstantOp $x), (NonConstantOp $y)]>;

def moveConstOr : Pat <
  (OrPrimOp:$old $x, $y),
  (MoveNameHint $old, (OrPrimOp $y, $x)),
  [(AnyConstantOp $x), (NonConstantOp $y)]>;

def moveConstXor : Pat <
  (XorPrimOp:$old $x, $y),
  (MoveNameHint $old, (XorPrimOp $y, $x)),
  [(AnyConstantOp $x), (NonConstantOp $y)]>;

////////////////////////////////////////////////////////////////////////////////
// Binary ops
////////////////////////////////////////////////////////////////////////////////

// add(a, 0) -> a
def AddOfZero : Pat <
  (AddPrimOp:$old $x, (ConstantOp:$zcst $_)),
  (PadPrimOp $x, (TypeWidth32 $old)),
  [(KnownWidth $x), (ZeroConstantOp $zcst)]>;

// add(a, a) -> a << 2
def AddOfSelf : Pat <
  (AddPrimOp:$old $x, $x),
  (MoveNameHint $old, (ShlPrimOp $x, (NativeCodeCall<"1">))),
  [(KnownWidth $x)]>;

// add((pad a, n), b) -> pad(add(a, b), n)
def AddOfPad : Pat <
  (AddPrimOp:$old (either (PadPrimOp $a, $m), $b)),
  (PadPrimOp (MoveNameHint $old, (AddPrimOp $a, $b)), (TypeWidth32 $old)),
  [(KnownWidth $a), (KnownWidth $b)]>;

// sub(a, 0) -> a
def SubOfZero : Pat <
  (SubPrimOp:$old $x, (ConstantOp:$zcst $cst)),
  (MoveNameHint $old, (PadPrimOp $x, (TypeWidth32 $old))),
  [(KnownWidth $x), (ZeroConstantOp $zcst)]>;

// sub(0, a) -> -a
def SubFromZeroSigned : Pat <
  (SubPrimOp:$old (ConstantOp:$zcst $cst), $x),
  (PadPrimOp (MoveNameHint $old, (NegPrimOp $x)), (TypeWidth32 $old)),
  [(KnownWidth $x), (ZeroConstantOp $zcst), (SIntType $x)]>;

//sub(0, a) -> -a
def SubFromZeroUnsigned : Pat <
  (SubPrimOp:$old (ConstantOp:$zcst $cst), $x),
  (AsUIntPrimOp (PadPrimOp (MoveNameHint $old, (NegPrimOp $x)), (TypeWidth32 $old))),
  [(KnownWidth $x), (ZeroConstantOp $zcst), (UIntType $x)]>;

//sub(a,a) -> 0
def SubOfSelf : Pat <
  (SubPrimOp:$old $x, $x),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntZerosAttr($0.getType()))"> $old),
  [(KnownWidth $x)]>;

// sub((pad a, n), b) -> pad(sub(a, b), n)
def SubOfPadL : Pat <
  (SubPrimOp:$old (PadPrimOp $a, $m), $b),
  (MoveNameHint $old, (PadPrimOp (SubPrimOp $a, $b), (TypeWidth32 $old))),
  [(KnownWidth $a), (KnownWidth $b)]>;

// sub(b, (pad a, n)) -> pad(sub(b, a), n)
def SubOfPadR : Pat <
  (SubPrimOp:$old $b, (PadPrimOp $a, $m)),
  (MoveNameHint $old, (PadPrimOp (SubPrimOp $b, $a), (TypeWidth32 $old))),
  [(KnownWidth $a), (KnownWidth $b)]>;

// and(x, 0) -> 0, fold can't handle all cases
def AndOfZero : Pat <
  (AndPrimOp:$old $x, (ConstantOp:$zcst $cst)),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntZerosAttr($0.getType()))"> $old),
  [(KnownWidth $x), (ZeroConstantOp $zcst)]>;

// and(x, -1) -> x, fold can't handle all cases
def AndOfAllOne : Pat <
  (AndPrimOp:$old $x, (ConstantOp:$ocst $cst)),
  (MoveNameHint $old, (AsUIntPrimOp (PadPrimOp $x, (TypeWidth32 $old)))),
  [(KnownWidth $x), (EqualTypes $x, $ocst), (AllOneConstantOp $ocst)]>;

/// and(x,x) -> x, fold can't handle all cases
def AndOfSelf : Pat <
  (AndPrimOp:$old $x, $x),
  (MoveNameHint $old, (AsUIntPrimOp $x)),
  [(KnownWidth $x)]>;

/// and(x, pad(y, n)) -> pad(and(tail(x), y), n), x is unsigned
def AndOfPad : Pat <
  (AndPrimOp:$old (either $x, (PadPrimOp:$pad $y, $n))),
  (MoveNameHint $old, (PadPrimOp (AndPrimOp (TailPrimOp $x, (TypeWidthAdjust32 $x, $y)), $y), $n)),
  [(KnownWidth $x), (UIntType $x), (EqualTypes $x, $pad)]>;



def AndOfAsSIntL : Pat<
  (AndPrimOp:$old (AsSIntPrimOp $x), $y),
  (MoveNameHint $old, (AndPrimOp $x, (AsUIntPrimOp $y))),
  [(KnownWidth $x), (EqualIntSize $x, $y)]>;

def AndOfAsSIntR : Pat<
  (AndPrimOp:$old $x, (AsSIntPrimOp $y)),
  (MoveNameHint $old, (AndPrimOp (AsUIntPrimOp $x), $y)),
  [(KnownWidth $x), (EqualIntSize $x, $y)]>;

// or(x, 0) -> x, fold can't handle all cases
def OrOfZero : Pat <
  (OrPrimOp:$old $x, (ConstantOp:$zcst $cst)),
  (MoveNameHint $old, (AsUIntPrimOp (PadPrimOp $x, (TypeWidth32 $old)))),
  [(KnownWidth $x), (ZeroConstantOp $zcst)]>;

// or(x, -1) -> -1, fold can't handle all cases
def OrOfAllOne : Pat <
  (OrPrimOp:$old $x, (ConstantOp:$ocst $cst)),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntOnesAttr($0.getType()))"> $old),
  [(KnownWidth $x), (EqualTypes $x, $ocst), (AllOneConstantOp $ocst)]>;

/// or(x,x) -> x, fold can't handle all cases
def OrOfSelf : Pat <
  (OrPrimOp:$old $x, $x),
  (MoveNameHint $old, (AsUIntPrimOp $x)),
  [(KnownWidth $x)]>;

/// or(x, pad(y, n)) -> cat(head(x), or(tail(x), y))
def OrOfPad : Pat <
  (OrPrimOp:$old $x, (PadPrimOp:$pad $y, $n)),
  (MoveNameHint $old, (CatPrimOp (variadic (HeadPrimOp $x, (TypeWidthAdjust32 $x, $y)),
                                                   (OrPrimOp (TailPrimOp $x, (TypeWidthAdjust32 $x, $y)), $y)))),
  [(KnownWidth $x), (UIntType $x), (EqualTypes $x, $pad)]>;

/// or(a, orr(x)) -> orr(cat(a, x))
def OrOrr : Pat <
  (OrPrimOp:$old $a, (OrRPrimOp $x)),
  (MoveNameHint $old, (OrRPrimOp (CatPrimOp (variadic (AsUIntPrimOp $a), (AsUIntPrimOp $x))))),
  // No type extension in the Or
  [(IntTypeWidthLTX<2> $a)]>;

// xor(x, 0) -> x, fold can't handle all cases
def XorOfZero : Pat <
  (XorPrimOp:$old $x, (ConstantOp:$zcst $_)),
  (MoveNameHint $old, (AsUIntPrimOp (PadPrimOp $x, (TypeWidth32 $old)))),
  [(KnownWidth $x), (ZeroConstantOp $zcst)]>;

/// xor(x,x) -> 0, fold can't handle all cases
def XorOfSelf : Pat <
  (XorPrimOp:$old $x, $x),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntZerosAttr($0.getType()))"> $old),
  [(KnownWidth $x)]>;

/// xor(x, pad(y, n)) -> cat(head(x), xor(tail(x), y))
def XorOfPad : Pat <
  (XorPrimOp:$old $x, (PadPrimOp:$pad $y, $n)),
  (MoveNameHint $old, (CatPrimOp (variadic (HeadPrimOp $x, (TypeWidthAdjust32 $x, $y)),
                                                   (XorPrimOp (TailPrimOp $x, (TypeWidthAdjust32 $x, $y)), $y)))),
  [(KnownWidth $x), (UIntType $x), (EqualTypes $x, $pad)]>;

/// orr(asSInt(x)) -> orr(x)
def OrRasSInt : Pat <
  (OrRPrimOp:$old (AsSIntPrimOp $x)),
  (MoveNameHint $old, (OrRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// orr(asUInt(x)) -> orr(x)
def OrRasUInt : Pat <
  (OrRPrimOp:$old (AsUIntPrimOp $x)),
  (MoveNameHint $old, (OrRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// orr(pad(x,n)) -> orr(x)
def OrRPadU : Pat <
  (OrRPrimOp:$old (PadPrimOp:$pad $x, $n)),
  (MoveNameHint $old, (OrRPrimOp $x)),
  [(KnownWidth $x), (UIntTypes $x), (IntTypeWidthGEQ32 $pad, $x)]>;

/// orr(cat(y, orr(z))) -> orr(cat(y,z))
def OrRCatOrR_left : Pat <
  (OrRPrimOp:$old (CatPrimOp (variadic $y, (OrRPrimOp $z)))),
  (MoveNameHint $old, (OrRPrimOp (CatPrimOp (variadic $y, (AsUIntPrimOp $z))))),
  []>;

/// orr(cat(orr(z),y)) -> orr(cat(z,y))
def OrRCatOrR_right : Pat <
  (OrRPrimOp:$old (CatPrimOp (variadic (OrRPrimOp $z), $y))),
  (MoveNameHint $old, (OrRPrimOp (CatPrimOp (variadic (AsUIntPrimOp $z), $y)))),
  []>;

/// xorr(asSInt(x)) -> xorr(x)
def XorRasSInt : Pat <
  (XorRPrimOp:$old (AsSIntPrimOp $x)),
  (MoveNameHint $old, (XorRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// xorr(asUInt(x)) -> xorr(x)
def XorRasUInt : Pat <
  (XorRPrimOp:$old (AsUIntPrimOp $x)),
  (MoveNameHint $old, (XorRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// xorr(pad(x,n)) -> xorr(x)
def XorRPadU : Pat <
  (XorRPrimOp:$old (PadPrimOp:$pad $x, $n)),
  (MoveNameHint $old, (XorRPrimOp $x)),
  [(KnownWidth $x), (UIntTypes $x), (IntTypeWidthGEQ32 $pad, $x)]>;

/// xorr(cat(y, xorr(z))) -> xorr(cat(y,z))
def XorRCatXorR_left : Pat <
  (XorRPrimOp:$old (CatPrimOp (variadic $y, (XorRPrimOp $z)))),
  (MoveNameHint $old, (XorRPrimOp (CatPrimOp (variadic $y, (AsUIntPrimOp $z))))),
  []>;

/// xorr(cat(xorr(z),y)) -> xorr(cat(z,y))
def XorRCatXorR_right : Pat <
  (XorRPrimOp:$old (CatPrimOp (variadic (XorRPrimOp $z), $y))),
  (MoveNameHint $old, (XorRPrimOp (CatPrimOp (variadic (AsUIntPrimOp $z), $y)))),
  []>;


/// andr(asSInt(x)) -> andr(x)
def AndRasSInt : Pat <
  (AndRPrimOp:$old (AsSIntPrimOp $x)),
  (MoveNameHint $old, (AndRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// andr(asUInt(x)) -> andr(x)
def AndRasUInt : Pat <
  (AndRPrimOp:$old (AsUIntPrimOp $x)),
  (MoveNameHint $old, (AndRPrimOp $x)),
  [(KnownWidth $x), (IntTypes $x)]>;

/// andr(pad(x:U,n)) -> 0 (where pad is doing something)
def AndRPadU : Pat <
  (AndRPrimOp:$old (PadPrimOp:$pad $x, $n)),
  (NativeCodeCall<"$_builder.create<ConstantOp>($0.getLoc(), type_cast<IntType>($0.getType()), getIntZerosAttr($0.getType()))"> $old),
  [(KnownWidth $x), (UIntTypes $x), (IntTypeWidthGT32 $pad, $x)]>;

/// andr(pad(x:S,n)) -> andr(x)
def AndRPadS : Pat <
  (AndRPrimOp:$old (PadPrimOp:$pad $x, $n)),
  (MoveNameHint $old, (AndRPrimOp $x)),
  [(KnownWidth $x), (SIntTypes $x)]>;

/// andr(cat(y, andr(z))) -> andr(cat(y,z))
def AndRCatAndR_left : Pat <
  (AndRPrimOp:$old (CatPrimOp (variadic $y, (AndRPrimOp $z)))),
  (MoveNameHint $old, (AndRPrimOp (CatPrimOp (variadic $y, (AsUIntPrimOp $z))))),
  []>;

/// andr(cat(andr(z),y)) -> andr(cat(z,y))
def AndRCatAndR_right : Pat <
  (AndRPrimOp:$old (CatPrimOp (variadic (AndRPrimOp $z), $y))),
  (MoveNameHint $old, (AndRPrimOp (CatPrimOp (variadic (AsUIntPrimOp $z), $y)))),
  []>;

// bits(bits(x, ...), ...) -> bits(x, ...)
def BitsOfBits : Pat<
  (BitsPrimOp:$old (BitsPrimOp $x, I32Attr:$iHigh, I32Attr:$iLow), I32Attr:$oHigh, I32Attr:$oLow),
  (MoveNameHint $old, (BitsPrimOp $x,
    (NativeCodeCall<"$_builder.getI32IntegerAttr($0.getValue().getZExtValue() + $1.getValue().getZExtValue())"> $iLow, $oHigh),
    (NativeCodeCall<"$_builder.getI32IntegerAttr($0.getValue().getZExtValue() + $1.getValue().getZExtValue())"> $oLow, $iLow))),
  []>;

// bits(mux(c,x,y)) -> mux(c, bits(x), bits(y))
// This isn't obviously good if the mux has multiple uses, so limit it to constants.
def BitsOfMux : Pat<
  (BitsPrimOp:$old (MuxPrimOp $c, $x, $y), I32Attr:$oHigh, I32Attr:$oLow),
  (MoveNameHint $old, (MuxPrimOp $c, (BitsPrimOp $x, $oHigh, $oLow), (BitsPrimOp $y, $oHigh, $oLow))),
  [(AnyConstantOp $x), (AnyConstantOp $y)]>;

// bits(asUInt) -> bits
def BitsOfAsUInt : Pat<
  (BitsPrimOp:$old (AsUIntPrimOp $x), I32Attr:$oHigh, I32Attr:$oLow),
  (MoveNameHint $old, (BitsPrimOp $x, $oHigh, $oLow)),
  [(KnownWidth $x)]>;

// bits(asUInt) -> bits
def BitsOfAnd : Pat<
  (BitsPrimOp:$old (AndPrimOp $x, $y), I32Attr:$oHigh, I32Attr:$oLow),
  (MoveNameHint $old, (AndPrimOp (BitsPrimOp $x, $oHigh, $oLow), (BitsPrimOp $y, $oHigh, $oLow))),
  [(KnownWidth $x), (EqualTypes $x, $y)]>;

// bits(pad(x)) -> x when they cancel out
def AttrIsZero : Constraint<CPred<"$0.getValue().isZero()">>;
def BitsOfPad : Pat<
  (BitsPrimOp:$old (PadPrimOp $x, I32Attr:$n), I32Attr:$oHigh, I32Attr:$oLow),
  (AsUIntPrimOp $x),
  [(KnownWidth $x), (EqualIntSize $old, $x), (AttrIsZero $oLow)]>;

def IndexInBound : Constraint<CPred<"cast<IntegerAttr>($0).getAPSInt().getExtValue() < static_cast<int64_t>(type_cast<FVectorType>($1.getType()).getNumElements())">>;

// subaccess a, cst -> subindex a, cst
def SubaccessOfConstant : Pat<
  (SubaccessOp:$old $array, (ConstantOp $cst)),
  (MoveNameHint $old, (SubindexOp $array, (NativeCodeCall<"$_builder.getI32IntegerAttr(cast<IntegerAttr>($0).getAPSInt().getExtValue())"> $cst))),
  [(IndexInBound $cst, $array)]>;

// mux(cond, 0, 1) -> ~cond
def MuxNot : Pat<
  (MuxPrimOp:$old $cond, (ConstantOp:$zcst $_), (ConstantOp:$ocst $_)),
  (MoveNameHint $old, (NotPrimOp $cond)), [
    (EqualTypes $cond, $zcst), (EqualTypes $cond, $ocst),
    (ZeroConstantOp $zcst) , (OneConstantOp $ocst)
  ]>;

// mux(cond1, t, mux(cond2, t ,f)) -> mux(or(cond1, cond2), t, f)
def MuxSameTrue : Pat <
  (MuxPrimOp:$old $cond1, $t, (MuxPrimOp $cond2, $t, $f)),
  (MoveNameHint $old, (MuxPrimOp (OrPrimOp $cond1, $cond2), $t, $f)),
  [(EqualTypes $t, $f), (KnownWidth $t)]>;

// mux(cond1, mux(cond2, t, f), f) -> mux(and(cond1, cond2), t, f)
def MuxSameFalse : Pat <
  (MuxPrimOp:$old $cond1, (MuxPrimOp $cond2, $t, $f), $f),
  (MoveNameHint $old, (MuxPrimOp (AndPrimOp $cond1, $cond2), $t, $f)),
  [(EqualTypes $t, $f), (KnownWidth $f)]>;

// mux(cond, (Concat a, b), (Concat a, c)) -> Concat(a, mux(cond, b, c))
def NarrowMuxLHS : Pat <
  (MuxPrimOp:$old $cond, (CatPrimOp:$lhs (variadic $a, $b)), (CatPrimOp:$rhs (variadic $a, $c))),
  (MoveNameHint $old, (CatPrimOp (variadic $a, (MuxPrimOp $cond, $b, $c)))),
  [(EqualTypes $lhs, $rhs), (KnownWidth $lhs)]>;

// mux(cond, (Concat a, b), (Concat c, b)) -> Concat(mux(cond, a, c), b)
def NarrowMuxRHS : Pat <
  (MuxPrimOp:$old $cond, (CatPrimOp:$lhs (variadic $a, $b)), (CatPrimOp:$rhs (variadic $c, $b))),
  (MoveNameHint $old, (CatPrimOp (variadic (MuxPrimOp $cond, $a, $c), $b))),
  [(EqualTypes $lhs, $rhs), (KnownWidth $lhs)]>;

// mux(eq(a, b), a, b) -> b
def MuxEQOperands : Pat<
  (MuxPrimOp (EQPrimOp $a, $b), $a, $b),
  (replaceWithValue $b),
  [(EqualTypes $a, $b), (KnownWidth $a)]>;

// mux(eq(b, a), a, b) -> b
def MuxEQOperandsSwapped : Pat<
  (MuxPrimOp (EQPrimOp $b, $a), $a, $b),
  (replaceWithValue $b),
  [(EqualTypes $a, $b), (KnownWidth $a)]>;

// mux(neq(a, b), x, y) -> mux(eq(a, b), y, x)
def MuxNEQ : Pat<
  (MuxPrimOp:$old (NEQPrimOp $a, $b), $x, $y),
  (MoveNameHint $old, (MuxPrimOp (EQPrimOp $a, $b), $y, $x)),
  [(EqualTypes $x, $y), (KnownWidth $x)]>;

// mux(cond : u0, a, b) -> mux(0 : u1, a, b)
def MuxPadSel : Pat<
  (MuxPrimOp:$old $cond, $a, $b),
  (MoveNameHint $old, (MuxPrimOp
    (ConstantOp
      (NativeCodeCall<"getIntZerosAttr($_builder.getType<UIntType>(1))">),
      (returnType "$_builder.getType<UIntType>(1)")),
    $a, $b)),
  [(IntTypeWidthLTX<1> $cond)]>;

// mux2(cond : u0, a, b) -> mux2(0 : u1, a, b)
def Mux2PadSel : Pat<
  (Mux2CellIntrinsicOp:$old $cond, $a, $b),
  (MoveNameHint $old, (Mux2CellIntrinsicOp
    (ConstantOp
      (NativeCodeCall<"getIntZerosAttr($_builder.getType<UIntType>(1))">),
      (returnType "$_builder.getType<UIntType>(1)")),
    $a, $b)),
  [(IntTypeWidthLTX<1> $cond)]>;

// mux4(cond : u0/u1, a, b) -> mux4(pad(cond -> u2), a, b)
def Mux4PadSel : Pat<
  (Mux4CellIntrinsicOp:$old $cond, $a, $b, $c, $d),
  (MoveNameHint $old, (Mux4CellIntrinsicOp
    (PadPrimOp $cond,
      (NativeCodeCall<"$_builder.getI32IntegerAttr(2)">)),
    $a, $b, $c, $d)),
  [(IntTypeWidthLTX<2> $cond)]>;

def CatDoubleConst : Pat <
  (CatPrimOp:$old (variadic $cst1, (CatPrimOp (variadic $cst2, $v)))),
  (MoveNameHint $old, (CatPrimOp (variadic (CatPrimOp (variadic $cst1, (AsUIntPrimOp $cst2))), (AsUIntPrimOp $v)))),
  [(KnownWidth $v), (AnyConstantOp $cst1), (AnyConstantOp $cst2)]>;

// cat(asUint(x), asUint(y)) -> cat(x,y)
def CatCast : Pat <
  (CatPrimOp:$old (variadic (AsUIntPrimOp $x), (AsUIntPrimOp $y))),
  (MoveNameHint $old, (CatPrimOp (variadic $x, $y))),
  [(EqualSigns $x, $y)]>;

// cat(bits a:b x, bits b+1:c x) -> bits( a:c x)
def CatBitsBits : Pat <
  (CatPrimOp:$old (variadic (BitsPrimOp $x, I32Attr:$hi, I32Attr:$mid), (BitsPrimOp $x, I32Attr:$mid2, I32Attr:$low))),
  (MoveNameHint $old, (BitsPrimOp $x, $hi, $low)),
  [(AdjInt $mid2, $mid)]>;

// regreset(clock, constant_zero, resetValue) -> reg(clock)
def RegResetWithZeroReset : Pat<
  (RegResetOp $clock, $reset, $_, $name, $nameKind, $annotations, $inner_sym, $forceable),
  (RegOp $clock, $name, $nameKind, $annotations, $inner_sym, $forceable), [(ZeroConstantOp $reset)]>;

// ref.resolve(ref.send(a)) -> a
def RefResolveOfRefSend : Pat<
  (RefResolveOp:$resolve (RefSendOp $base)),
  (replaceWithValue $base), [(EqualTypes $base, $resolve)]
>;

def UtoStoU : Pat<
  (AsSIntPrimOp (AsUIntPrimOp $x)),
  (replaceWithValue $x),
  []>;

def StoUtoS : Pat<
  (AsUIntPrimOp (AsSIntPrimOp $x)),
  (replaceWithValue $x),
  []>;

def CVTSigned : Pat<
  (CvtPrimOp $x),
  (replaceWithValue $x),
  [(SIntType $x)]>;

def CVTUnSigned : Pat<
  (CvtPrimOp:$old $x),
  (MoveNameHint $old,
    (AsSIntPrimOp
      (PadPrimOp $x,
        (NativeCodeCall<"$_builder.getI32IntegerAttr(type_cast<SIntType>($0.getType()).getBitWidthOrSentinel())"> $old)))),
  [(UIntType $x), (KnownWidth $old)]>;

def AssertXWhenX : Pattern<(AssertOp $_, $x, $x, $_, $_, $_, $_, $_), []>;

def AssumeXWhenX : Pattern<(AssumeOp $_, $x, $x, $_, $_, $_, $_, $_), []>;

def UnclockedAssumeIntrinsicXWhenX
    : Pattern<(UnclockedAssumeIntrinsicOp $x, $x, $_, $_, $_), []>;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLCANONICALIZATION_TD
