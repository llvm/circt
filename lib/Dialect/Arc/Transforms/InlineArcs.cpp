//===- InlineArcs.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-inline"

using namespace circt;
using namespace arc;
using llvm::SetVector;
using llvm::SmallDenseSet;
using mlir::InlinerInterface;
using mlir::SymbolUserMap;

namespace {

struct InlineArcsPass : public InlineArcsBase<InlineArcsPass> {
  InlineArcsPass() = default;
  InlineArcsPass(const InlineArcsPass &pass) : InlineArcsPass() {}

  void runOnOperation() override;
  bool shouldInline(DefineOp defOp, ArrayRef<Operation *> users);
};

/// A simple implementation of the `InlinerInterface` that marks all inlining as
/// legal since we know that we only ever attempt to inline `DefineOp` bodies
/// at `StateOp` sites.
struct ArcInliner : public InlinerInterface {
  using InlinerInterface::InlinerInterface;

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }
  void handleTerminator(Operation *op,
                        ArrayRef<Value> valuesToRepl) const override {
    assert(isa<arc::OutputOp>(op));
    for (auto [from, to] : llvm::zip(valuesToRepl, op->getOperands()))
      from.replaceAllUsesWith(to);
  }
};

} // namespace

void InlineArcsPass::runOnOperation() {
  auto module = getOperation();
  SymbolTableCollection symbolTable;
  SymbolUserMap symbolUserMap(symbolTable, module);

  for (auto defOp : llvm::make_early_inc_range(module.getOps<DefineOp>())) {
    // Check if we should inline the arc.
    auto users = symbolUserMap.getUsers(defOp);
    if (!shouldInline(defOp, users))
      continue;
    LLVM_DEBUG(llvm::dbgs() << "Inlining " << defOp.getSymName() << "\n");

    // Inline all uses of the arc. Currently we inline all of them but in the
    // future we may decide per use site whether to inline or not.
    unsigned numUsersLeft = users.size();
    for (auto *user : users) {
      auto useOp = dyn_cast<StateOp>(user);
      if (!useOp)
        continue;
      if (useOp.getLatency() > 0)
        continue;

      bool isLastArcUse = --numUsersLeft == 0;
      ArcInliner inliner(&getContext());
      if (failed(mlir::inlineRegion(inliner, &defOp.getBody(), useOp,
                                    useOp.getOperands(), useOp.getResults(),
                                    std::nullopt, !isLastArcUse))) {
        useOp.emitError("failed to inline arc '") << defOp.getName() << "'";
        return signalPassFailure();
      }
      useOp.erase();
      if (isLastArcUse) {
        defOp.erase();
        ++numRemovedArcs;
      }
      ++numInlinedArcs;
    }
  }
}

bool InlineArcsPass::shouldInline(DefineOp defOp, ArrayRef<Operation *> users) {
  // Count the number of non-trivial ops in the arc. If there are only a few,
  // inline the arc.
  unsigned numNonTrivialOps = 0;
  defOp.getBodyBlock().walk([&](Operation *op) {
    if (!op->hasTrait<OpTrait::ConstantLike>() && !isa<OutputOp>(op))
      ++numNonTrivialOps;
  });
  if (numNonTrivialOps <= 3) {
    ++numTrivialArcs;
    return true;
  }
  LLVM_DEBUG(llvm::dbgs() << "Arc " << defOp.getSymName() << " has "
                          << numNonTrivialOps << " non-trivial ops\n");

  // Check if the arc is only ever used once.
  if (users.size() == 1) {
    ++numSingleUseArcs;
    return true;
  }

  return false;
}

std::unique_ptr<Pass> arc::createInlineArcsPass() {
  return std::make_unique<InlineArcsPass>();
}
