//===- LayerSink.cpp - Sink ops into layer blocks -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file sinks operations into layer blocks.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

#include "circt/Support/Debug.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/ControlFlowSinkUtils.h"

#define DEBUG_TYPE "firrtl-layer-sink"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LAYERSINK
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
/// A control-flow sink pass.
struct LayerSink : public circt::firrtl::impl::LayerSinkBase<LayerSink> {
  void runOnOperation() override;
};
} // end anonymous namespace

void LayerSink::runOnOperation() {
  LLVM_DEBUG(debugPassHeader(this)
                 << "\n"
                 << "Module: '" << getOperation().getName() << "'\n";);
  auto &domInfo = getAnalysis<mlir::DominanceInfo>();
  getOperation()->walk([&](LayerBlockOp layerBlock) {
    SmallVector<Region *> regionsToSink({&layerBlock.getRegion()});
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

std::unique_ptr<mlir::Pass> circt::firrtl::createLayerSinkPass() {
  return std::make_unique<LayerSink>();
}
