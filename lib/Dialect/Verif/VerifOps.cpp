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
#include "circt/Support/CustomDirectiveImpl.h"
#include "circt/Support/FoldUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace circt;
using namespace verif;
using namespace mlir;

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
// ClockedAssertLikeOps
//===----------------------------------------------------------------------===//

namespace {
// Verify function for clocked assert / assume / cover ops.
// This checks that they do not contiain any nested clocks or disable operations
// Clocked assertlike ops are a simple form of assertions that only
// contain one clock and one disable condition.
struct ClockedAssertLikeOp {
  static LogicalResult verify(Operation *clockedAssertLikeOp) {
    // Used to perform a DFS search through the module to visit all operands
    // before they are used
    llvm::SmallMapVector<Operation *, OperandRange::iterator, 16> worklist;

    // Keeps track of operations that have been visited
    llvm::DenseSet<Operation *> handledOps;

    Operation *property = clockedAssertLikeOp->getOperand(0).getDefiningOp();

    // Fill in our worklist
    worklist.insert({property, property->operand_begin()});

    // Process the elements in our worklist
    while (!worklist.empty()) {
      auto &[op, operandIt] = worklist.back();

      if (operandIt == op->operand_end()) {
        // Check that our property doesn't contain any illegal ops
        if (isa<ltl::ClockOp, ltl::DisableOp>(op)) {
          op->emitError("Nested clock or disable operations are not "
                        "allowed for clock_assertlike operations.");
          return failure();
        }

        // Record that our op has been visited
        handledOps.insert(op);
        worklist.pop_back();
        continue;
      }

      // Send the operands of our op to the worklist in case they are still
      // un-visited
      Value operand = *(operandIt++);
      auto *defOp = operand.getDefiningOp();

      // Make sure that we don't visit the same operand twice
      if (!defOp || handledOps.contains(defOp))
        continue;

      // This is triggered if our operand is already in the worklist and
      // wasn't handled
      if (!worklist.insert({defOp, defOp->operand_begin()}).second) {
        op->emitError("dependency cycle");
        return failure();
      }
    }
    return success();
  }
};
} // namespace

LogicalResult ClockedAssertOp::verify() {
  return ClockedAssertLikeOp::verify(getOperation());
}

LogicalResult ClockedAssumeOp::verify() {
  return ClockedAssertLikeOp::verify(getOperation());
}

LogicalResult ClockedCoverOp::verify() {
  return ClockedAssertLikeOp::verify(getOperation());
}

//===----------------------------------------------------------------------===//
// Generated code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/Verif/Verif.cpp.inc"
