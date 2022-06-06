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

/// Constraint that matches a ConstantOp, SpecialConstantOp, or InvalidValueOp.
def AnyConstantOp : Constraint<CPred<[{
  isa_and_nonnull<ConstantOp, SpecialConstantOp, InvalidValueOp>
    ($0.getDefiningOp())
  }]>>;

/// Constraint that matches non-constant operations. Used to ensure that the
/// const-on-LHS rewriting patterns converge in case both operands are constant.
def NonConstantOp : Constraint<CPred<[{
  !isa_and_nonnull<ConstantOp, SpecialConstantOp, InvalidValueOp>
    ($0.getDefiningOp())
  }]>>;

// Constraint that matches empty attribute dictionaries.  Used to ensure that
// there are no FIRRTL annotation on an operation.
def EmptyAttrDict : Constraint<CPred<"$0.empty()">>;

def GetDroppableNameAttr : NativeCodeCall<
  "NameKindEnumAttr::get($_builder.getContext(), NameKindEnum::DroppableName) ">;
def NonEmptyAttr : Constraint<CPred<"!$0.getValue().empty()">>;
def uselessNameAttr : Constraint<CPred<"isUselessName($0.getValue())">>;
def NullAttr : Constraint<CPred<"!$0">>;

// Constraint that enforces equal types
def EqualTypes : Constraint<CPred<"$0.getType() == $1.getType()">>;

// Constraint that enforces types of known width
def KnownWidth : Constraint<CPred<[{
  $0.getType().isa<FIRRTLType>() &&
    $0.getType().cast<FIRRTLType>().getBitWidthOrSentinel() >= 0
  }]>>;

/// Constraint that matches a zero ConstantOp or SpecialConstantOp.
def ZeroConstantOp : Constraint<Or<[
  CPred<"$0.getDefiningOp<ConstantOp>() &&"
        "$0.getDefiningOp<ConstantOp>().value().isZero()">,
  CPred<"$0.getDefiningOp<SpecialConstantOp>() &&"
        "$0.getDefiningOp<SpecialConstantOp>().value() == false">
]>>;

def GetEmptyString : NativeCodeCall<
  "StringAttr::get($_builder.getContext(), {}) ">;

// leq(const, x) -> geq(x, const)
def LEQWithConstLHS : Pat<
  (LEQPrimOp $lhs, $rhs),
  (GEQPrimOp $rhs, $lhs),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// lt(const, x) -> gt(x, const)
def LTWithConstLHS : Pat<
  (LTPrimOp $lhs, $rhs),
  (GTPrimOp $rhs, $lhs),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// geq(const, x) -> leq(x, const)
def GEQWithConstLHS : Pat<
  (GEQPrimOp $lhs, $rhs),
  (LEQPrimOp $rhs, $lhs),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

// gt(const, x) -> lt(x, const)
def GTWithConstLHS : Pat<
  (GTPrimOp $lhs, $rhs),
  (LTPrimOp $rhs, $lhs),
  [(AnyConstantOp $lhs), (NonConstantOp $rhs)]>;

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

// regreset(clock, invalidvalue, resetValue) -> reg(clock)
def RegResetWithInvalidReset : Pat<
  (RegResetOp $clock, (InvalidValueOp), $_, $name, $annotations, $inner_sym, $nameKind),
  (RegOp $clock, $name, $annotations, $inner_sym, $nameKind),
  []>;

// regreset(clock, reset, invalidvalue) -> reg(clock)
// This is handled by the `RemoveReset` pass in the original Scala code.
def RegResetWithInvalidResetValue : Pat<
  (RegResetOp $clock, $_, (InvalidValueOp), $name, $annotations, $inner_sym, $nameKind),
  (RegOp $clock, $name, $annotations, $inner_sym, $nameKind),
  []>;

// regreset(clock, constant_zero, resetValue) -> reg(clock)
def RegResetWithZeroReset : Pat<
  (RegResetOp $clock, $reset, $_, $name, $annotations, $inner_sym, $nameKind),
  (RegOp $clock, $name, $annotations, $inner_sym, $nameKind), [(ZeroConstantOp $reset)]>;

// Return the width of an operation result as an integer attribute.  This is
// useful to pad another operation up to the width of the original operation.
def GetWidthAsIntAttr : NativeCodeCall<
  "IntegerAttr::get(IntegerType::get($_builder.getContext(), 32, IntegerType::Signless), "
                    "$0.getType().cast<FIRRTLType>().getBitWidthOrSentinel())">;

////////////////////////////////////////////////////////////////////////////////
// Drop name if no symbol (DontTouch application) and name is a stylized
// temporary name
////////////////////////////////////////////////////////////////////////////////

// node(x,name),!dnt -> node(x, "")
def DropNameNode : Pat<
  (NodeOp $x, $name, $annotations, $inner_sym, $nameKind),
  (NodeOp $x, (GetEmptyString),  $annotations, $inner_sym, (GetDroppableNameAttr)),
  [(uselessNameAttr $name), (NonEmptyAttr $name), (NullAttr $inner_sym)]>;

// wire(name),!dnt -> wire("")
def DropNameWire : Pat<
  (WireOp $name,  $annotations, $inner_sym, $nameKind),
  (WireOp (GetEmptyString), $annotations, $inner_sym, (GetDroppableNameAttr)),
  [(uselessNameAttr $name), (NonEmptyAttr $name), (NullAttr $inner_sym)]>;

// reg(clk, name),!dnt -> reg(clk, "")
def DropNameReg : Pat<
  (RegOp $clk, $name, $annotations, $inner_sym, $nameKind),
  (RegOp $clk, (GetEmptyString), $annotations, $inner_sym, (GetDroppableNameAttr)),
  [(uselessNameAttr $name), (NonEmptyAttr $name), (NullAttr $inner_sym)]>;

// regreset(clk, sig, val, name),!dnt -> regreset(clk, sig, val, "")
def DropNameRegReset : Pat<
  (RegResetOp $clk, $sig, $val, $name, $annotations, $inner_sym, $nameKind),
  (RegResetOp $clk, $sig, $val, (GetEmptyString),
                    $annotations, $inner_sym, (GetDroppableNameAttr)),
  [(uselessNameAttr $name), (NonEmptyAttr $name), (NullAttr $inner_sym)]>;

// Mem(...,name),!dnt -> Mem(...x, "")
def DropNameMem : Pat<
  (MemOp $rd, $rw, $depth, $ruw, $portNames, $name, $annotations, $portAnno,
                                             $inner_sym, $groupID, $nameKind),
  (MemOp $rd, $rw, $depth, $ruw, $portNames, (GetEmptyString), $annotations,
                     $portAnno, $inner_sym, $groupID, (GetDroppableNameAttr)),
  [(uselessNameAttr $name), (NonEmptyAttr $name), (NullAttr $inner_sym)]>;

#endif // CIRCT_DIALECT_FIRRTL_FIRRTLCANONICALIZATION_TD
