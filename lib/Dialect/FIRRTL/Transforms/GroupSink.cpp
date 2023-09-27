//===- GroupSink.cpp - Sink ops into FIRRTL optional groups ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file sinks operations in FIRRTL optional groups.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"

#define DEBUG_TYPE "firrtl-group-sink"

using namespace circt;
using namespace firrtl;

namespace {
/// A control-flow sink pass.
struct GroupSink : public GroupSinkBase<GroupSink> {
  void runOnOperation() override;
};
} // end anonymous namespace

void GroupSink::runOnOperation() {
  LLVM_DEBUG(
      llvm::dbgs() << "==----- Running GroupSink "
                      "---------------------------------------------------===\n"
                   << "Module: '" << getOperation().getName() << "'\n";);
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  getOperation()->walk([&](GroupOp group) {
    SmallVector<Region *> regionsToSink({&group.getRegion()});
    numSunk = controlFlowSink(
        regionsToSink, domInfo,
        [](Operation *op, Region *) { return !hasDontTouch(op); },
        [](Operation *op, Region *region) {
          // Move the operation to the beginning of the region's entry block.
          // This guarantees the preservation of SSA dominance of all of the
          // operation's uses are in the region.
          op->moveBefore(&region->front(), region->front().begin());
        });
  });
}

std::unique_ptr<mlir::Pass> circt::firrtl::createGroupSinkPass() {
  return std::make_unique<GroupSink>();
}
