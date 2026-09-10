//===- StripDebugInfoWithPred.cpp - Strip debug information selectively ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallVector.h"

namespace circt {
#define GEN_PASS_DEF_STRIPDEBUGINFOWITHPRED
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
template <typename OpOrBlockArgument>
static void updateLocIfChanged(OpOrBlockArgument *op, Location newLoc) {
  if (op->getLoc() != newLoc)
    op->setLoc(newLoc);
}

namespace {
struct StripDebugInfoWithPred
    : public circt::impl::StripDebugInfoWithPredBase<StripDebugInfoWithPred> {
  StripDebugInfoWithPred(const std::function<bool(mlir::Location)> &pred)
      : pred(pred) {}
  void runOnOperation() override;

  // Return stripped location for the given `loc`.
  mlir::Location getStrippedLoc(Location loc) {
    SetVector<Location> newLocations;
    SmallVector<Location, 8> worklist({loc});
    while (!worklist.empty()) {
      auto currentLoc = worklist.pop_back_val();
      if (auto fusedLoc = dyn_cast<FusedLoc>(currentLoc)) {
        for (auto loc : fusedLoc.getLocations())
          worklist.push_back(loc);
      } else {
        if (!pred(currentLoc))
          newLocations.insert(currentLoc);
      }
    }

    // NOTE: Don't use FusedLoc::get(&getContext(), locations, metadata) to
    // avoid a bytecode reader bug llvm-project#99626.
    return FusedLoc::get(newLocations.getArrayRef(), Attribute(),
                         &getContext());
  }

  void updateLocArray(Operation *op, StringRef attributeName) {
    SmallVector<Attribute> newLocs;
    if (auto resLocs = op->getAttrOfType<ArrayAttr>(attributeName)) {
      bool changed = false;
      for (auto loc : resLocs.getAsRange<LocationAttr>()) {
        auto newLoc = getStrippedLoc(loc);
        changed |= newLoc != loc;
        newLocs.push_back(newLoc);
      }
      if (changed)
        op->setAttr(attributeName, ArrayAttr::get(&getContext(), newLocs));
    }
  }

private:
  std::function<bool(mlir::Location)> pred;
};
} // namespace

void StripDebugInfoWithPred::runOnOperation() {
  // If pred is null and dropSuffix is non-empty, initialize the predicate to
  // strip file info with that suffix.
  if (!pred && !dropSuffix.empty()) {
    pred = [&](mlir::Location loc) {
      if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
        return fileLoc.getFilename().getValue().ends_with(dropSuffix);
      return false;
    };
  }

  if (!pred) {
    getOperation().emitWarning()
        << "predicate is uninitialized. No debug information is stripped.";
    return;
  }

  auto stripLocsOnOp = [this](Operation *op) {
    updateLocIfChanged(op, getStrippedLoc(op->getLoc()));
    updateLocArray(op, "arg_locs");
    updateLocArray(op, "result_locs");
    updateLocArray(op, "port_locs");
  };

  // Handle operations sequentially if they have no regions,
  // and parallelize walking over the remainder.
  // If not for this special sequential-vs-parallel handling, would instead
  // only do a simple walk and defer to pass scheduling for parallelism.
  SmallVector<Operation *> topLevelOpsToWalk;
  for (auto &op : getOperation().getOps()) {
    // Gather operations with regions for parallel processing.
    if (op.getNumRegions() != 0) {
      topLevelOpsToWalk.push_back(&op);
      continue;
    }

    // Otherwise, handle right now -- not worth the cost.
    stripLocsOnOp(&op);
  }

  parallelForEach(&getContext(), topLevelOpsToWalk, [&](Operation *toplevelOp) {
    toplevelOp->walk([&](Operation *op) {
      stripLocsOnOp(op);
      // Strip block arguments debug info.
      for (Region &region : op->getRegions())
        for (Block &block : region.getBlocks())
          for (BlockArgument &arg : block.getArguments())
            updateLocIfChanged(&arg, getStrippedLoc(arg.getLoc()));
    });
  });
}

namespace circt {
/// Creates a pass to strip debug information from a function.
std::unique_ptr<Pass> createStripDebugInfoWithPredPass(
    const std::function<bool(mlir::Location)> &pred) {
  return std::make_unique<StripDebugInfoWithPred>(pred);
}
} // namespace circt
