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
// Formal contract verifiers
//===----------------------------------------------------------------------===//

LogicalResult ContractOp::verifyRegions() {
  // Retrieve the number of inputs from the parent module
  auto parent = (*this)->getParentOfType<hw::HWModuleOp>();
  // Sanity check: parent should always be a hw.module
  if (!parent)
    return emitOpError() << "parent of contract must be an hw.module!";

  auto nRes = (*this)->getNumResults();
  auto resTypes = (*this)->getResultTypes();
  auto *yield = getBody().front().getTerminator();

  // Check that the region terminator yields the same number of ops as the
  // number of results
  if (yield->getNumOperands() != nRes)
    return emitOpError() << "region terminator must yield the same number of "
                         << "operands as there are results!";

  // Check that the region terminator yields the same types of ops as the
  // types of results
  if (yield->getOperandTypes() != resTypes)
    return emitOpError() << "region terminator must yield the same types of "
                         << "operands as the result types!";

  // Check that the region block arguments share the same types as the results
  if (getBody().front().getArgumentTypes() != resTypes)
    return emitOpError() << "region must have the same type of arguments "
                         << "as the type of results!";

  return success();
}

//===----------------------------------------------------------------------===//
// LogicalEquivalenceCheckingOp
//===----------------------------------------------------------------------===//

LogicalResult LogicEquivalenceCheckingOp::verifyRegions() {
  if (getFirstCircuit().getArgumentTypes() !=
      getSecondCircuit().getArgumentTypes())
    return emitOpError() << "block argument types of both regions must match";
  if (getFirstCircuit().front().getTerminator()->getOperandTypes() !=
      getSecondCircuit().front().getTerminator()->getOperandTypes())
    return emitOpError()
           << "types of the yielded values of both regions must match";

  return success();
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
  return success();
}

//===----------------------------------------------------------------------===//
// Generated code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/Verif/Verif.cpp.inc"
