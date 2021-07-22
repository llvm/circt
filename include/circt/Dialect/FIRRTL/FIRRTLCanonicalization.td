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

/// Constraint that matches non-constant operations. Used to ensure that the
/// const-on-LHS rewriting patterns converge in case both operands are constant.
def NonConstantOp : Constraint<CPred<"!$0.getDefiningOp<ConstantOp>()">>;

// Constraint that matches empty attribute dictionaries.  Used to ensure that
// there are no FIRRTL annotation on an operation.
def EmptyAttrDict : Constraint<CPred<"$0.empty()">>;

// Constraint that enforces equal types
def EqualTypes : Constraint<CPred<"$0.getType() == $1.getType()">>;

// Constraint that enforces types of known width
def KnownWidth : Constraint<CPred<[{
  $0.getType().isa<FIRRTLType>() &&
    $0.getType().cast<FIRRTLType>().getBitWidthOrSentinel() >= 0
  }]>>;

// leq(const, x) -> geq(x, const)
def LEQWithConstLHS : Pat<
  (LEQPrimOp (ConstantOp:$lhs $_), $rhs),
  (GEQPrimOp $rhs, $lhs),
  [(NonConstantOp $rhs)]>;

// lt(const, x) -> gt(x, const)
def LTWithConstLHS : Pat<
  (LTPrimOp (ConstantOp:$lhs $_), $rhs),
  (GTPrimOp $rhs, $lhs),
  [(NonConstantOp $rhs)]>;

// geq(const, x) -> leq(x, const)
def GEQWithConstLHS : Pat<
  (GEQPrimOp (ConstantOp:$lhs $_), $rhs),
  (LEQPrimOp $rhs, $lhs),
  [(NonConstantOp $rhs)]>;

// gt(const, x) -> lt(x, const)
def GTWithConstLHS : Pat<
  (GTPrimOp (ConstantOp:$lhs $_), $rhs),
  (LTPrimOp $rhs, $lhs),
  [(NonConstantOp $rhs)]>;

// mux(cond, x, mux(cond, y, z)) -> mux(cond, x, z)
def MuxSameCondLow : Pat<
  (MuxPrimOp $cond, $x, (MuxPrimOp $cond, $y, $z)),
  (MuxPrimOp $cond, $x, $z), [
    (EqualTypes $x, $y), (EqualTypes $x, $z), (KnownWidth $x)
  ]>;

// mux(cond, mux(cond, y, z), x) -> mux(cond, y, x)
def MuxSameCondHigh : Pat<
  (MuxPrimOp $cond, (MuxPrimOp $cond, $y, $z), $x),
  (MuxPrimOp $cond, $y, $x), [
    (EqualTypes $x, $y), (EqualTypes $x, $z), (KnownWidth $x)
  ]>;

// node(x) -> x
def EmptyNode : Pat<
  (NodeOp $x, $name, $annotations),
  (replaceWithValue $x),
  [(EmptyAttrDict $annotations)]>;

// regreset(clock, invalidvalue, resetValue) -> reg(clock)
def RegresetWithInvalidReset : Pat<
  (RegResetOp $clock, (InvalidValueOp), $resetValue, $name, $annotations),
  (RegOp $clock, $name, $annotations),
  []>;

// regreset(clock, reset, invalidvalue) -> reg(clock)
// This is handled by the `RemoveReset` pass in the original Scala code.
def RegresetWithInvalidResetValue : Pat<
  (RegResetOp $clock, $reset, (InvalidValueOp), $name, $annotations),
  (RegOp $clock, $name, $annotations),
  []>;

// regreset(clock, constant_zero, resetValue) -> reg(clock)
def RegresetWithZeroReset : Pat<
  (RegResetOp $clock, (SpecialConstantOp ConstBoolAttrFalse), $resetValue,
   $name, $annotations),
  (RegOp $clock, $name, $annotations)>;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLCANONICALIZATION_TD
