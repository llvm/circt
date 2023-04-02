//===- RemoveUnusedArcArguments.cpp - Implement RemoveUnusedArcArgs Pass --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implement pass to remove unused arguments of arc::DefineOps. Also adjusts the
// arc::StateOps referencing the arc.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "arc-remove-unused-arc-arguments"

using namespace mlir;
using namespace circt::arc;

//===----------------------------------------------------------------------===//
// RemoveUnusedArcArgumentsPattern
//===----------------------------------------------------------------------===//

namespace {
struct UnusedArcArgsStatistics {
  unsigned numArgsRemoved = 0;
  unsigned numArcsTouched = 0;
  unsigned numArgsMissed = 0;
};

class RemoveUnusedArcArgumentsPattern : public OpRewritePattern<DefineOp> {
public:
  RemoveUnusedArcArgumentsPattern(MLIRContext *context,
                                  SymbolUserMap &symbolUsers,
                                  UnusedArcArgsStatistics &statistics)
      : OpRewritePattern<DefineOp>(context), symbolUsers(symbolUsers),
        statistics(statistics) {}

  LogicalResult matchAndRewrite(DefineOp op,
                                PatternRewriter &rewriter) const final;

private:
  SymbolUserMap &symbolUsers;
  UnusedArcArgsStatistics &statistics;
};
} // namespace

LogicalResult RemoveUnusedArcArgumentsPattern::matchAndRewrite(
    DefineOp op, PatternRewriter &rewriter) const {
  BitVector toDelete(op.getNumArguments());
  for (auto [i, arg] : llvm::enumerate(op.getArguments()))
    if (arg.use_empty())
      toDelete.set(i);

  if (toDelete.none())
    return failure();

  // Collect the mutable callers in a first iteration. If there is a user that
  // does not implement the interface, we have to abort the rewrite and have to
  // make sure that we didn't change anything so far.
  SmallVector<CallOpMutableInterface> mutableUsers;
  for (auto *user : symbolUsers.getUsers(op)) {
    if (auto callOpMutable = dyn_cast<CallOpMutableInterface>(user)) {
      mutableUsers.push_back(callOpMutable);
    } else {
      statistics.numArgsMissed += toDelete.count();
      return failure();
    }
  }

  // Do the actual rewrites.
  for (auto user : mutableUsers)
    for (int i = toDelete.size() - 1; i >= 0; --i)
      if (toDelete[i])
        user.getArgOperandsMutable().erase(i);

  op.eraseArguments(toDelete);
  op.setFunctionType(
      rewriter.getFunctionType(op.getArgumentTypes(), op.getResultTypes()));

  statistics.numArgsRemoved += toDelete.count();
  // If an arc is touched multiple times because more arguments got unused
  // during the process, it is counted multiple times.
  ++statistics.numArcsTouched;
  return success();
}

//===----------------------------------------------------------------------===//
// RemoveUnusedArcArguments pass
//===----------------------------------------------------------------------===//

namespace {
struct RemoveUnusedArcArgumentsPass
    : public RemoveUnusedArcArgumentsBase<RemoveUnusedArcArgumentsPass> {
  void runOnOperation() override;
};
} // namespace

void RemoveUnusedArcArgumentsPass::runOnOperation() {
  SymbolTableCollection collection;
  SymbolUserMap symbolUsers(collection, getOperation());

  UnusedArcArgsStatistics statistics;
  RewritePatternSet patterns(&getContext());
  patterns.add<RemoveUnusedArcArgumentsPattern>(&getContext(), symbolUsers,
                                                statistics);

  if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
    signalPassFailure();

  numArcArgsRemoved = statistics.numArgsRemoved;
  numArcArgsMissed = statistics.numArgsMissed;
  numArcsTouched = statistics.numArcsTouched;
}

std::unique_ptr<Pass> arc::createRemoveUnusedArcArgumentsPass() {
  return std::make_unique<RemoveUnusedArcArgumentsPass>();
}
