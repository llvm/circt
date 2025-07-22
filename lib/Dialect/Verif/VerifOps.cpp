//===- VerifOps.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/LTL/LTLTypes.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Support/CustomDirectiveImpl.h"
#include "circt/Support/FoldUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"

using namespace circt;
using namespace verif;
using namespace mlir;

static ClockEdge ltlToVerifClockEdge(ltl::ClockEdge ce) {
  switch (ce) {
  case ltl::ClockEdge::Pos:
    return ClockEdge::Pos;
  case ltl::ClockEdge::Neg:
    return ClockEdge::Neg;
  case ltl::ClockEdge::Both:
    return ClockEdge::Both;
  }
  llvm_unreachable("Unknown event control kind");
}

//===----------------------------------------------------------------------===//
// HasBeenResetOp
//===----------------------------------------------------------------------===//

OpFoldResult HasBeenResetOp::fold(FoldAdaptor adaptor) {
  // Fold to zero if the reset is a constant. In this case the op is either
  // permanently in reset or never resets. Both mean that the reset never
  // finishes, so this op never returns true.
  if (adaptor.getReset())
    return BoolAttr::get(getContext(), false);

  // Fold to zero if the clock is a constant and the reset is synchronous. In
  // that case the reset will never be started.
  if (!adaptor.getAsync() && adaptor.getClock())
    return BoolAttr::get(getContext(), false);

  return {};
}

//===----------------------------------------------------------------------===//
// AssertLikeOps Canonicalizations
//===----------------------------------------------------------------------===//

namespace AssertLikeOp {
// assertlike(ltl.clock(prop, clk), en) -> clocked_assertlike(prop, en, clk)
template <typename TargetOp, typename Op>
static LogicalResult canonicalize(Op op, PatternRewriter &rewriter) {
  // If the property is a block argument, then no canonicalization is possible
  Value property = op.getProperty();
  auto clockOp = property.getDefiningOp<ltl::ClockOp>();
  if (!clockOp)
    return failure();

  // Check for clock operand
  // If it exists, fold it into a clocked assertlike
  rewriter.replaceOpWithNewOp<TargetOp>(
      op, clockOp.getInput(), ltlToVerifClockEdge(clockOp.getEdge()),
      clockOp.getClock(), op.getEnable(), op.getLabelAttr());

  return success();
}

} // namespace AssertLikeOp

LogicalResult AssertOp::canonicalize(AssertOp op, PatternRewriter &rewriter) {
  return AssertLikeOp::canonicalize<ClockedAssertOp>(op, rewriter);
}

LogicalResult AssumeOp::canonicalize(AssumeOp op, PatternRewriter &rewriter) {
  return AssertLikeOp::canonicalize<ClockedAssumeOp>(op, rewriter);
}

LogicalResult CoverOp::canonicalize(CoverOp op, PatternRewriter &rewriter) {
  return AssertLikeOp::canonicalize<ClockedCoverOp>(op, rewriter);
}

//===----------------------------------------------------------------------===//
// CircuitRelationCheckOp
//===----------------------------------------------------------------------===//

template <typename OpTy>
static LogicalResult verifyCircuitRelationCheckOpRegions(OpTy &op) {
  if (op.getFirstCircuit().getArgumentTypes() !=
      op.getSecondCircuit().getArgumentTypes())
    return op.emitOpError()
           << "block argument types of both regions must match";
  if (op.getFirstCircuit().front().getTerminator()->getOperandTypes() !=
      op.getSecondCircuit().front().getTerminator()->getOperandTypes())
    return op.emitOpError()
           << "types of the yielded values of both regions must match";

  return success();
}

//===----------------------------------------------------------------------===//
// LogicalEquivalenceCheckingOp
//===----------------------------------------------------------------------===//

LogicalResult LogicEquivalenceCheckingOp::verifyRegions() {
  return verifyCircuitRelationCheckOpRegions(*this);
}

//===----------------------------------------------------------------------===//
// RefinementCheckingOp
//===----------------------------------------------------------------------===//

LogicalResult RefinementCheckingOp::verifyRegions() {
  return verifyCircuitRelationCheckOpRegions(*this);
}

//===----------------------------------------------------------------------===//
// BoundedModelCheckingOp
//===----------------------------------------------------------------------===//

LogicalResult BoundedModelCheckingOp::verifyRegions() {
  if (!getInit().getArgumentTypes().empty())
    return emitOpError() << "init region must have no arguments";
  auto *initYieldOp = getInit().front().getTerminator();
  auto *loopYieldOp = getLoop().front().getTerminator();
  if (initYieldOp->getOperandTypes() != loopYieldOp->getOperandTypes())
    return emitOpError()
           << "init and loop regions must yield the same types of values";
  if (initYieldOp->getOperandTypes() != getLoop().front().getArgumentTypes())
    return emitOpError()
           << "loop region arguments must match the types of the values "
              "yielded by the init and loop regions";
  size_t totalClocks = 0;
  auto circuitArgTy = getCircuit().getArgumentTypes();
  for (auto input : circuitArgTy)
    if (isa<seq::ClockType>(input))
      totalClocks++;
  auto initYields = initYieldOp->getOperands();
  // We know init and loop yields match, so only need to check one
  if (initYields.size() < totalClocks)
    return emitOpError()
           << "init and loop regions must yield at least as many clock "
              "values as there are clock arguments to the circuit region";
  for (size_t i = 0; i < totalClocks; i++) {
    if (!isa<seq::ClockType>(initYieldOp->getOperand(i).getType()))
      return emitOpError()
             << "init and loop regions must yield as many clock values as "
                "there are clock arguments in the circuit region "
                "before any other values";
  }
  if (getNumRegs() > 0 && totalClocks == 0)
    return emitOpError("num_regs is non-zero, but the circuit region has no "
                       "clock inputs to clock the registers");
  auto initialValues = getInitialValues();
  if (initialValues.size() != getNumRegs()) {
    return emitOpError()
           << "number of initial values must match the number of registers";
  }
  for (auto attr : initialValues) {
    if (!isa<IntegerAttr, UnitAttr>(attr))
      return emitOpError()
             << "initial values must be integer or unit attributes";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// SimulationOp
//===----------------------------------------------------------------------===//

LogicalResult SimulationOp::verifyRegions() {
  if (getBody()->getNumArguments() != 2)
    return emitOpError() << "must have two block arguments";
  if (!isa<seq::ClockType>(getBody()->getArgument(0).getType()))
    return emitOpError() << "block argument #0 must be of type `!seq.clock`";
  if (!getBody()->getArgument(1).getType().isSignlessInteger(1))
    return emitOpError() << "block argument #1 must be of type `i1`";

  auto *yieldOp = getBody()->getTerminator();
  if (yieldOp->getNumOperands() != 2)
    return yieldOp->emitOpError() << "must have two operands";
  if (!yieldOp->getOperand(0).getType().isSignlessInteger(1))
    return yieldOp->emitOpError() << "operand #0 must be of type `i1`";
  if (!yieldOp->getOperand(1).getType().isSignlessInteger(1))
    return yieldOp->emitOpError() << "operand #1 must be of type `i1`";

  return success();
}

void SimulationOp::getAsmBlockArgumentNames(Region &region,
                                            OpAsmSetValueNameFn setNameFn) {
  if (region.empty() || region.getNumArguments() != 2)
    return;
  setNameFn(region.getArgument(0), "clock");
  setNameFn(region.getArgument(1), "init");
}

//===----------------------------------------------------------------------===//
// Generated code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/Verif/Verif.cpp.inc"
