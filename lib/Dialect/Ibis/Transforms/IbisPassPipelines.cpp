//===- IbisPassPipelines.cpp - Ibis pass pipelines ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/IbisPassPipelines.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace circt;
using namespace ibis;

/// Create a simple canonicalizer pass.
static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.useTopDownTraversal = true;
  config.enableRegionSimplification = mlir::GreedySimplifyRegionLevel::Disabled;
  return mlir::createCanonicalizerPass(config);
}

void circt::ibis::loadIbisLowLevelPassPipeline(mlir::PassManager &pm) {
  // Inner ref: We create an inner ref verification pass to initially validate
  // the IR, as well as after all structure-changing passes.
  // In the future, could consider hiding this behind a flag to reduce overhead.
  pm.addPass(hw::createVerifyInnerRefNamespacePass());
  pm.nest<ibis::DesignOp>().addPass(createContainerizePass());
  pm.addPass(hw::createVerifyInnerRefNamespacePass());

  // Pre-tunneling CSE pass. This ensures that duplicate get_port calls are
  // removed before we start tunneling - no reason to tunnel the same thing
  // twice.
  pm.addPass(mlir::createCSEPass());
  pm.nest<DesignOp>().addPass(
      createTunnelingPass(IbisTunnelingOptions{"", ""}));
  pm.addPass(hw::createVerifyInnerRefNamespacePass());
  pm.addPass(createPortrefLoweringPass());
  pm.addPass(createSimpleCanonicalizerPass());
  pm.nest<DesignOp>().addPass(createCleanSelfdriversPass());
  pm.addPass(createContainersToHWPass());
  pm.addPass(hw::createVerifyInnerRefNamespacePass());
}

void circt::ibis::loadIbisHighLevelPassPipeline(mlir::PassManager &pm) {
  pm.nest<ibis::DesignOp>()
      .nest<ibis::ClassOp>()
      .nest<ibis::MethodOp>()
      .addPass(ibis::createInlineSBlocksPass());
  pm.addPass(mlir::createMem2Reg());

  // TODO @mortbopet: Add a verification pass to ensure that there are no more
  // memref.alloca's - we want all memories to be mem2reg'able, unless they are
  // member variable accesses.
  // - just add it as an illegal op.

  // Now, perform SSA maximizations.
  pm.addPass(circt::createMaximizeSSAPass());

  // SSA maximal form achieved. Reconstruct the Ibis sblocks.
  pm.nest<ibis::DesignOp>()
      .nest<ibis::ClassOp>()
      .nest<ibis::MethodOp>()
      .addPass(ibis::createReblockPass());
  pm.addPass(ibis::createArgifyBlocksPass());
  pm.addPass(createSimpleCanonicalizerPass());
}
