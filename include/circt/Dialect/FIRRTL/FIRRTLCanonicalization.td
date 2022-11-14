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

include "circt/Dialect/FIRRTL/FIRRTL.td"

/// Constraint that matches a ConstantOp, SpecialConstantOp, AggregateConstantOp, or InvalidValueOp.
def AnyConstantOp : Constraint<CPred<[{
  isa_and_nonnull<ConstantOp, SpecialConstantOp, InvalidValueOp, AggregateConstantOp>
    ($0.getDefiningOp())
  }]>>;

/// Constraint that matches non-constant operations. Used to ensure that the
/// const-on-LHS rewriting patterns converge in case both operands are constant.
def NonConstantOp : Constraint<CPred<[{
  !isa_and_nonnull<ConstantOp, SpecialConstantOp, InvalidValueOp, AggregateConstantOp>
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
def EqualTypes : Constraint<CPred<"$0.getType() == $1.getType()">>;

// Constraint that enforces types of known width
def KnownWidth : Constraint<CPred<[{
  $0.getType().isa<FIRRTLBaseType>() &&
    $0.getType().cast<FIRRTLBaseType>().getBitWidthOrSentinel() >= 0
  }]>>;

/// Constraint that matches a zero ConstantOp or SpecialConstantOp.
def ZeroConstantOp : Constraint<Or<[
  CPred<"$0.getDefiningOp<ConstantOp>() &&"
        "$0.getDefiningOp<ConstantOp>().getValue().isZero()">,
  CPred<"$0.getDefiningOp<SpecialConstantOp>() &&"
        "$0.getDefiningOp<SpecialConstantOp>().getValue() == false">
]>>;

/// Constraint that matches a one ConstantOp or SpecialConstantOp.
def OneConstantOp : Constraint<Or<[
  CPred<"$0.getDefiningOp<ConstantOp>() &&"
        "$0.getDefiningOp<ConstantOp>().getValue().isOne()">,
  CPred<"$0.getDefiningOp<SpecialConstantOp>() &&"
        "$0.getDefiningOp<SpecialConstantOp>().getValue() == true">
]>>;

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

// mux(cond, x, mux(cond, y, z)) -> mux(cond, x, z)
def MuxSameCondLow : Pat<
  (MuxPrimOp:$old $cond, $x, (MuxPrimOp $cond, $y, $z)),
  (MoveNameHint $old, (MuxPrimOp $cond, $x, $z)), [
    (EqualTypes $x, $y), (EqualTypes $x, $z), (KnownWidth $x)
  ]>;

// mux(cond, mux(cond, y, z), x) -> mux(cond, y, x)
def MuxSameCondHigh : Pat<
  (MuxPrimOp:$old $cond, (MuxPrimOp $cond, $y, $z), $x),
  (MoveNameHint $old, (MuxPrimOp $cond, $y, $x)), [
    (EqualTypes $x, $y), (EqualTypes $x, $z), (KnownWidth $x)
  ]>;

// regreset(clock, constant_zero, resetValue) -> reg(clock)
def RegResetWithZeroReset : Pat<
  (RegResetOp $clock, $reset, $_, $name, $nameKind, $annotations, $inner_sym),
  (RegOp $clock, $name, $nameKind, $annotations, $inner_sym), [(ZeroConstantOp $reset)]>;

// regreset(clock, constant_one, resetValue) -> node(resetValue)
def RegResetWithOneReset : Pat<
  (RegResetOp:$reg $clock, $reset, $resetVal, $name, $nameKind, $annotations, $inner_sym),
  (DropWrite $reg, (NodeOp $resetVal, $name, $nameKind, $annotations, $inner_sym)),
  [(OneConstantOp $reset)]>;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLCANONICALIZATION_TD
