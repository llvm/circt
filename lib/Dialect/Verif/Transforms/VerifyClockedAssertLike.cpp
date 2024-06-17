//===- VerifyClockedAssertLike.cpp - Check Clocked Asserts -------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Checks that clocked assert-like ops are constructed correctly.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWModuleGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/LTL/LTLTypes.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Dialect/Verif/VerifPasses.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace verif {
#define GEN_PASS_DEF_VERIFYCLOCKEDASSERTLIKE
#include "circt/Dialect/Verif/Passes.h.inc"
} // namespace verif
} // namespace circt

using namespace circt;
using namespace verif;

namespace {
// Verify function for clocked assert / assume / cover ops.
// This checks that they do not contiain any nested clocks or disable operations
// Clocked assertlike ops are a simple form of assertions that only
// contain one clock and one disable condition.
struct VerifyClockedAssertLikePass
    : circt::verif::impl::VerifyClockedAssertLikeBase<
          VerifyClockedAssertLikePass> {
private:
  // Used to perform a DFS search through the module to visit all operands
  // before they are used
  llvm::SmallMapVector<Operation *, OperandRange::iterator, 16> worklist;

  // Keeps track of operations that have been visited
  llvm::DenseSet<Operation *> handledOps;

public:
  void runOnOperation() override;

private:
  void verify(Operation *clockedAssertLikeOp) {

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
          return;
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
        return;
      }
    }

    // Clear worklist and such
    worklist.clear();
    handledOps.clear();
  }
};
} // namespace

void VerifyClockedAssertLikePass::runOnOperation() {
  getOperation()->walk([&](Operation *op) {
    llvm::TypeSwitch<Operation *, void>(op)
        .Case<verif::ClockedAssertOp, verif::ClockedAssumeOp,
              verif::ClockedCoverOp>([&](auto clockedOp) { verify(clockedOp); })
        .Default([&](auto) {});
  });
}

std::unique_ptr<mlir::Pass> circt::verif::createVerifyClockedAssertLikePass() {
  return std::make_unique<VerifyClockedAssertLikePass>();
}
