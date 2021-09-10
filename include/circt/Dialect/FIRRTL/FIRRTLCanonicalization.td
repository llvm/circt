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

// Constraint that matches annotation sets without dontTouch.
def NoDontTouch : Constraint<CPred<"!AnnotationSet($0).hasDontTouch()">>;

// Constraint that matches annotation sets without dontTouch.
def NoPortsDontTouch : Constraint<CPred<"llvm::all_of($0, [](Attribute a) { if (auto b = a.dyn_cast<ArrayAttr>()) return !AnnotationSet(b).hasDontTouch(); return true; })">>;

def NonEmptyStringAttr : Constraint<CPred<"!$0.getValue().empty()">>;
def uselessNameAttr : Constraint<CPred<"isUselessName($0.getValue())">>;

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
        "$0.getDefiningOp<ConstantOp>().value().isNullValue()">,
  CPred<"$0.getDefiningOp<SpecialConstantOp>() &&"
        "$0.getDefiningOp<SpecialConstantOp>().value() == false">
]>>;

def IsInvalid : Constraint<
  CPred<"$0.getDefiningOp<InvalidValueOp>()">
>;

def GetEmptyString : NativeCodeCall<
  "StringAttr::get($_builder.getContext(), {}) ">;

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
  [(NonEmptyStringAttr $name), (EmptyAttrDict $annotations)]>;

// regreset(clock, invalidvalue, resetValue) -> reg(clock)
def RegresetWithInvalidReset : Pat<
  (RegResetOp $clock, (InvalidValueOp), $_, $name, $annotations),
  (RegOp $clock, $name, $annotations),
  []>;

// regreset(clock, reset, invalidvalue) -> reg(clock)
// This is handled by the `RemoveReset` pass in the original Scala code.
def RegresetWithInvalidResetValue : Pat<
  (RegResetOp $clock, $_, (InvalidValueOp), $name, $annotations),
  (RegOp $clock, $name, $annotations),
  []>;

// regreset(clock, constant_zero, resetValue) -> reg(clock)
def RegresetWithZeroReset : Pat<
  (RegResetOp $clock, $reset, $_, $name, $annotations),
  (RegOp $clock, $name, $annotations), [(ZeroConstantOp $reset)]>;

// Return the width of an operation result as an integer attribute.  This is
// useful to pad another operation up to the width of the original operation.
def GetWidthAsIntAttr : NativeCodeCall<
  "IntegerAttr::get(IntegerType::get($_builder.getContext(), 32, IntegerType::Signless), "
                    "$0.getType().cast<FIRRTLType>().getBitWidthOrSentinel())">;

// add(x, invalid) -> pad(x, width)
//
// This is legal because it aligns with the Scala FIRRTL Compiler
// interpretation of lowering invalid to constant zero before constant
// propagation.
def AddWithInvalidOp : Pat<
  (AddPrimOp:$result $x, $y),
  (PadPrimOp $x, (GetWidthAsIntAttr $result)), [
    (KnownWidth $x), (IsInvalid $y)
  ]>;

// sub(x, invalid) -> pad(x, width)
//
// This is legal because it aligns with the Scala FIRRTL Compiler
// interpretation of lowering invalid to constant zero before constant
// propagation.
def SubWithInvalidOp : Pat<
  (SubPrimOp:$result $x, $y),
  (PadPrimOp $x, (GetWidthAsIntAttr $result)), [
    (KnownWidth $x), (IsInvalid $y)
  ]>;

////////////////////////////////////////////////////////////////////////////////
// DontTouch application
////////////////////////////////////////////////////////////////////////////////

// node(x,name),!dnt -> node(x, "")
def DropNameNode : Pat<
  (NodeOp $x, $name, $annotations),
  (NodeOp $x, (GetEmptyString), $annotations),
  [(uselessNameAttr $name), (NoDontTouch $annotations)]>;

// wire(name),!dnt -> wire("")
def DropNameWire : Pat<
  (WireOp $name, $annotations),
  (WireOp (GetEmptyString), $annotations),
  [(uselessNameAttr $name), (NoDontTouch $annotations)]>;

// reg(clk, name),!dnt -> reg(clk, "")
def DropNameReg : Pat<
  (RegOp $clk, $name, $annotations),
  (RegOp $clk, (GetEmptyString), $annotations),
  [(uselessNameAttr $name), (NoDontTouch $annotations)]>;

// regreset(clk, sig, val, name),!dnt -> regreset(clk, sig, val, "")
def DropNameRegReset : Pat<
  (RegResetOp $clk, $sig, $val, $name, $annotations),
  (RegResetOp $clk, $sig, $val, (GetEmptyString), $annotations),
  [(uselessNameAttr $name), (NoDontTouch $annotations)]>;

// Mem(...,name),!dnt -> Mem(...x, "")
def DropNameMem : Pat<
  (MemOp $rd, $rw, $depth, $ruw, $portNames, $name, $annotations, $portAnno),
  (MemOp $rd, $rw, $depth, $ruw, $portNames, (GetEmptyString), $annotations, $portAnno),
  [(uselessNameAttr $name), (NoDontTouch $annotations), (NoPortsDontTouch $portAnno)]>;

// CombMem(name),!dnt -> CombMem("")
def DropNameCombMem : Pat<
  (CombMemOp $name, $annotations),
  (CombMemOp (GetEmptyString), $annotations),
  [(uselessNameAttr $name), (NoDontTouch $annotations)]>;

// SeqMem(ruw, name),!dnt -> SeqMem(ruw, "")
def DropNameSeqMem : Pat<
  (SeqMemOp $ruw, $name, $annotations),
  (SeqMemOp $ruw, (GetEmptyString), $annotations),
  [(uselessNameAttr $name), (NoDontTouch $annotations)]>;

// memoryport(..., name),!dnt -> memoryport(..., "")
def DropNameMemoryPort : Pat<
  (MemoryPortOp  $memory, $direction, $name, $annotations),
  (MemoryPortOp  $memory, $direction, (GetEmptyString), $annotations),
  [(uselessNameAttr $name), (NoDontTouch $annotations)]>;


#endif // CIRCT_DIALECT_FIRRTL_FIRRTLCANONICALIZATION_TD
