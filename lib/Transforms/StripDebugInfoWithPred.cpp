//===- StripDebugInfoWithPred.cpp - Strip debug information selectively ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {
struct StripDebugInfoWithPred
    : public circt::StripDebugInfoWithPredBase<StripDebugInfoWithPred> {
  StripDebugInfoWithPred(std::function<bool(mlir::Location)> pred)
      : pred(std::move(pred)) {}
  void runOnOperation() override;

  // Return stripped location for the given `loc`.
  mlir::Location getStrippedLoc(Location loc) {
    // If `pred` return true, strip the location.
    if (pred(loc))
      return UnknownLoc::get(loc.getContext());

    if (auto fusedLoc = loc.dyn_cast<FusedLoc>()) {
      SmallVector<mlir::Location> newLocations;
      newLocations.reserve(fusedLoc.getLocations().size());
      for (auto loc : fusedLoc.getLocations())
        newLocations.push_back(getStrippedLoc(loc));
      return FusedLoc::get(&getContext(), newLocations, fusedLoc.getMetadata());
    }

    // TODO: Handle other loc type.
    return loc;
  }

private:
  std::function<bool(mlir::Location)> pred = {};
};
} // namespace

void StripDebugInfoWithPred::runOnOperation() {
  // If pred is null and dropSuffix is non-empty, initialize the predicate to
  // strip file info with that suffix.
  if (!pred && !dropSuffix.empty()) {
    pred = [&](mlir::Location loc) {
      if (auto fileLoc = loc.dyn_cast<FileLineColLoc>())
        return fileLoc.getFilename().getValue().endswith(dropSuffix);
      return false;
    };
  }

  if (!pred) {
    getOperation().emitWarning()
        << "predicate is uninitialized. No debug information is stripped.";
    return;
  }

  // TODO: Consider parallelize this.
  getOperation().walk([&](Operation *op) {
    op->setLoc(getStrippedLoc(op->getLoc()));

    // Strip block arguments debug info.
    for (Region &region : op->getRegions()) {
      for (Block &block : region.getBlocks()) {
        for (BlockArgument &arg : block.getArguments()) {
          arg.setLoc(getStrippedLoc(arg.getLoc()));
        }
      }
    }
  });
}

namespace circt {
/// Creates a pass to strip debug information from a function.
std::unique_ptr<Pass>
createStripDebugInfoWithPredPass(std::function<bool(mlir::Location)> pred) {
  return std::make_unique<StripDebugInfoWithPred>(std::move(pred));
}
} // namespace circt
