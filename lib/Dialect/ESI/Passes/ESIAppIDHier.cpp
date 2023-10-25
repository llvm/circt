//===- ESIAppIDHier.cpp - ESI build AppID hierarchy pass --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../PassDetails.h"

#include "circt/Dialect/ESI/AppID.h"
#include "circt/Dialect/ESI/ESIOps.h"
#include "circt/Dialect/ESI/ESIPasses.h"

using namespace circt;
using namespace esi;

namespace {
struct ESIAppIDHierPass : public ESIAppIDHierBase<ESIAppIDHierPass> {
  void runOnOperation() override;

private:
  // Existing node blocks.
  DenseMap<AppIDPathAttr, Block *> nodeBlock;

  /// Get the node's block for a particular path.
  // NOLINTNEXTLINE(misc-no-recursion)
  Block *getBlock(AppIDPathAttr path) {
    Block *&block = nodeBlock[path];
    if (block)
      return block;

    // Check if we need to create a root node.
    if (path.getPath().empty()) {
      auto rootOp = OpBuilder::atBlockEnd(getOperation().getBody())
                        .create<AppIDHierRootOp>(UnknownLoc::get(&getContext()),
                                                 path.getRoot());
      block = &rootOp.getChildren().emplaceBlock();
    } else {
      // Create a normal node underneath the parent AppID.
      auto *parentBlock = getBlock(path.getParent());
      auto node = OpBuilder::atBlockEnd(parentBlock)
                      .create<AppIDHierNodeOp>(UnknownLoc::get(&getContext()),
                                               path.getPath().back());
      block = &node.getChildren().emplaceBlock();
    }
    return block;
  };
};
} // anonymous namespace

void ESIAppIDHierPass::runOnOperation() {
  auto mod = getOperation();
  AppIDIndex index(mod);
  if (!index.isValid())
    return signalPassFailure();

  // Clone in manifest data, creating the instance hierarchy as we go.
  LogicalResult rc =
      index.walk(top, [&](AppIDPathAttr appidPath, Operation *op) {
        auto *block = getBlock(appidPath);
        if (isa<IsManifestData>(op))
          OpBuilder::atBlockEnd(block).clone(*op);
      });
  if (failed(rc))
    return signalPassFailure();
}

std::unique_ptr<OperationPass<ModuleOp>> circt::esi::createESIAppIDHierPass() {
  return std::make_unique<ESIAppIDHierPass>();
}
