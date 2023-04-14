//===- LowerMatches.cpp - Lower match statements to whens -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerMatchesPass, which lowers match statements to when
// statements
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"

using namespace circt;
using namespace firrtl;

namespace {
class LowerMatchesPass : public LowerMatchesBase<LowerMatchesPass> {
  void runOnOperation() override;
};
} // end anonymous namespace

static void lowerMatch(MatchOp match) {
  ImplicitLocOpBuilder b(match.getLoc(), match);

  auto input = match.getInput();
  auto numCases = match->getNumRegions();
  for (size_t i = 0; i < numCases - 1; ++i) {
    // Create a WhenOp which tests the enum's tag.
    auto condition = b.create<IsTagOp>(input, match.getFieldIndexAttr(i));
    auto when = b.create<WhenOp>(condition, /*withElse=*/true);

    // Move the case block to the WhenOp.
    auto *thenBlock = &when.getThenBlock();
    auto *caseBlock = &match.getRegion(i).front();
    caseBlock->moveBefore(thenBlock);
    thenBlock->erase();

    // Replace the block argument with a subtag op.
    b.setInsertionPointToStart(caseBlock);
    auto data = b.create<SubtagOp>(input, match.getFieldIndexAttr(i));
    caseBlock->getArgument(0).replaceAllUsesWith(data);
    caseBlock->eraseArgument(0);

    // Change insertion to the else block.
    b.setInsertionPointToStart(&when.getElseBlock());
  }

  // The final else is not handled with a conditional, move the case block to
  // the default block.
  auto *defaultBlock = b.getInsertionBlock();
  auto *caseBlock = &match->getRegions().back().front();
  caseBlock->moveBefore(defaultBlock);
  defaultBlock->erase();

  // Replace the block argument with a subtag op.
  b.setInsertionPointToStart(caseBlock);
  auto data = b.create<SubtagOp>(input, match.getFieldIndexAttr(numCases - 1));
  caseBlock->getArgument(0).replaceAllUsesWith(data);
  caseBlock->eraseArgument(0);

  // Erase the match op.
  match->erase();
}

void LowerMatchesPass::runOnOperation() {
  getOperation()->walk(&lowerMatch);
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createLowerMatchesPass() {
  return std::make_unique<LowerMatchesPass>();
}
