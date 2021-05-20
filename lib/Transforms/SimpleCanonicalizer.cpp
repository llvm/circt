//===- SimpleCanonicalizer.cpp - A simplified Canonicalizer pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a simplified canonicalizer pass that doesn't do CFG
// optimizations and other things that aren't helpful for many hardware IRs.
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/Passes.h"

#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
using namespace circt;

#define GEN_PASS_CLASSES
#include "circt/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// GreedyPatternRewriteDriver
//===----------------------------------------------------------------------===//

namespace {
/// This is a worklist-driven driver for the PatternMatcher, which repeatedly
/// applies the locally optimal patterns in a roughly "bottom up" way.
class GreedyPatternRewriteDriver : public PatternRewriter {
public:
  explicit GreedyPatternRewriteDriver(
      MLIRContext *ctx, const mlir::FrozenRewritePatternSet &patterns)
      : PatternRewriter(ctx), matcher(patterns), folder(ctx) {
    worklist.reserve(64);

    // Apply a simple cost model based solely on pattern benefit.
    matcher.applyDefaultCostModel();
  }

  void simplify(MutableArrayRef<Region> regions, int maxIterations);

  void addToWorklist(Operation *op) {
    // Check to see if the worklist already contains this op.
    if (worklistMap.count(op))
      return;

    worklistMap[op] = worklist.size();
    worklist.push_back(op);
  }

  Operation *popFromWorklist() {
    auto *op = worklist.back();
    worklist.pop_back();

    // This operation is no longer in the worklist, keep worklistMap up to date.
    if (op)
      worklistMap.erase(op);
    return op;
  }

  /// If the specified operation is in the worklist, remove it.  If not, this is
  /// a no-op.
  void removeFromWorklist(Operation *op) {
    auto it = worklistMap.find(op);
    if (it != worklistMap.end()) {
      assert(worklist[it->second] == op && "malformed worklist data structure");
      worklist[it->second] = nullptr;
      worklistMap.erase(it);
    }
  }

  // These are hooks implemented for PatternRewriter.
protected:
  // Implement the hook for inserting operations, and make sure that newly
  // inserted ops are added to the worklist for processing.
  void notifyOperationInserted(Operation *op) override { addToWorklist(op); }

  // If an operation is about to be removed, make sure it is not in our
  // worklist anymore because we'd get dangling references to it.
  void notifyOperationRemoved(Operation *op) override {
    addToWorklist(op->getOperands());
    op->walk([this](Operation *operation) {
      removeFromWorklist(operation);
      folder.notifyRemoval(operation);
    });
  }

  // When the root of a pattern is about to be replaced, it can trigger
  // simplifications to its users - make sure to add them to the worklist
  // before the root is changed.
  void notifyRootReplaced(Operation *op) override {
    for (auto result : op->getResults())
      for (auto *user : result.getUsers())
        addToWorklist(user);
  }

private:
  // Look over the provided operands for any defining operations that should
  // be re-added to the worklist. This function should be called when an
  // operation is modified or removed, as it may trigger further
  // simplifications.
  template <typename Operands>
  void addToWorklist(Operands &&operands) {
    for (Value operand : operands) {
      // If the use count of this operand is now < 2, we re-add the defining
      // operation to the worklist.
      // TODO: This is based on the fact that zero use operations
      // may be deleted, and that single use values often have more
      // canonicalization opportunities.
      if (!operand || (!operand.use_empty() && !operand.hasOneUse()))
        continue;
      if (auto *defInst = operand.getDefiningOp())
        addToWorklist(defInst);
    }
  }

  /// The low-level pattern applicator.
  mlir::PatternApplicator matcher;

  /// The worklist for this transformation keeps track of the operations that
  /// need to be revisited, plus their index in the worklist.  This allows us to
  /// efficiently remove operations from the worklist when they are erased, even
  /// if they aren't the root of a pattern.
  std::vector<Operation *> worklist;
  DenseMap<Operation *, unsigned> worklistMap;

  /// Non-pattern based folder for operations.
  mlir::OperationFolder folder;
};
} // end anonymous namespace

/// Performs the rewrites while folding and erasing any dead ops. Returns true
/// if the rewrite converges in `maxIterations`.
void GreedyPatternRewriteDriver::simplify(MutableArrayRef<Region> regions,
                                          int maxIterations) {
  // Add the given operation to the worklist.
  auto collectOps = [this](Operation *op) { addToWorklist(op); };

  bool changed = false;
  int i = 0;
  do {

    // Add all nested operations to the worklist.
    for (auto &region : regions)
      region.walk<mlir::WalkOrder::PreOrder>(
          [this](Operation *op) { worklist.push_back(op); });

    // Reverse the list so our pop-back loop processes them in-order.
    std::reverse(worklist.begin(), worklist.end());
    // Remember the reverse index.
    for (size_t i = 0, e = worklist.size(); i != e; ++i)
      worklistMap[worklist[i]] = i;

    // These are scratch vectors used in the folding loop below.
    SmallVector<Value, 8> originalOperands, resultValues;

    changed = false;
    while (!worklist.empty()) {
      auto *op = popFromWorklist();

      // Nulls get added to the worklist when operations are removed, ignore
      // them.
      if (op == nullptr)
        continue;

      // If the operation is trivially dead - remove it.
      if (isOpTriviallyDead(op)) {
        notifyOperationRemoved(op);
        op->erase();
        changed = true;
        continue;
      }

      // Collects all the operands and result uses of the given `op` into work
      // list. Also remove `op` and nested ops from worklist.
      originalOperands.assign(op->operand_begin(), op->operand_end());
      auto preReplaceAction = [&](Operation *op) {
        // Add the operands to the worklist for visitation.
        addToWorklist(originalOperands);

        // Add all the users of the result to the worklist so we make sure
        // to revisit them.
        for (auto result : op->getResults())
          for (auto *userOp : result.getUsers())
            addToWorklist(userOp);

        notifyOperationRemoved(op);
      };

      // Try to fold this op.
      bool inPlaceUpdate;
      if ((succeeded(folder.tryToFold(op, collectOps, preReplaceAction,
                                      &inPlaceUpdate)))) {
        changed = true;
        if (!inPlaceUpdate)
          continue;
      }

      // Try to match one of the patterns. The rewriter is automatically
      // notified of any necessary changes, so there is nothing else to do here.
      changed |= succeeded(matcher.matchAndRewrite(op, *this));
    }
  } while (changed && ++i < maxIterations);
}

//===----------------------------------------------------------------------===//
// SimpleCanonicalizer
//===----------------------------------------------------------------------===//

namespace {
/// Canonicalize operations in nested regions.
struct SimpleCanonicalizer
    : public SimpleCanonicalizerBase<SimpleCanonicalizer> {
  /// Initialize the canonicalizer by building the set of patterns used during
  /// execution.
  LogicalResult initialize(MLIRContext *context) override {
    RewritePatternSet owningPatterns(context);
    for (auto *op : context->getRegisteredOperations())
      op->getCanonicalizationPatterns(owningPatterns, context);
    patterns = std::move(owningPatterns);
    return success();
  }
  void runOnOperation() override;

  mlir::FrozenRewritePatternSet patterns;
};
} // end anonymous namespace

void SimpleCanonicalizer::runOnOperation() {
  auto regions = getOperation()->getRegions();
  if (regions.empty())
    return;

  // The top-level operation must be known to be isolated from above to
  // prevent performing canonicalizations on operations defined at or above
  // the region containing 'op'.
  assert(getOperation()->hasTrait<OpTrait::IsIsolatedFromAbove>() &&
         "patterns can only be applied to operations IsolatedFromAbove");

  // Start the pattern driver.
  GreedyPatternRewriteDriver driver(getOperation()->getContext(), patterns);

  // FIXME: 10 is a bad value for a magic number here!
  driver.simplify(regions, 10);
  return;
}

/// Create a Canonicalizer pass.
std::unique_ptr<Pass> circt::createSimpleCanonicalizerPass() {
  return std::make_unique<SimpleCanonicalizer>();
}
