//===- KanagawaPassPipelines.cpp - Kanagawa pass pipelines ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaPassPipelines.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Transforms/Passes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace circt;
using namespace kanagawa;

/// Create a simple canonicalizer pass.
static std::unique_ptr<Pass> createSimpleCanonicalizerPass() {
  mlir::GreedyRewriteConfig config;
  config.setUseTopDownTraversal(true);
  config.setRegionSimplificationLevel(
      mlir::GreedySimplifyRegionLevel::Disabled);
  return mlir::createCanonicalizerPass(config);
}

void circt::kanagawa::loadKanagawaLowLevelPassPipeline(mlir::PassManager &pm) {
  // Inner ref: We create an inner ref verification pass to initially validate
  // the IR, as well as after all structure-changing passes.
  // In the future, could consider hiding this behind a flag to reduce overhead.
  pm.addPass(hw::createVerifyInnerRefNamespacePass());
  pm.nest<kanagawa::DesignOp>().addPass(createContainerizePass());
  pm.addPass(hw::createVerifyInnerRefNamespacePass());

  // This pass ensures that duplicate get_port calls are removed before we
  // start tunneling - no reason to tunnel the same thing twice.
  pm.nest<DesignOp>().nest<ContainerOp>().addPass(
      createEliminateRedundantOpsPass());

  pm.nest<DesignOp>().addPass(
      createTunnelingPass(KanagawaTunnelingOptions{"", ""}));
  pm.addPass(hw::createVerifyInnerRefNamespacePass());
  pm.addPass(createPortrefLoweringPass());
  pm.addPass(createSimpleCanonicalizerPass());
  // Run this again as some of the above passes may create redundant ops.
  pm.nest<DesignOp>().nest<ContainerOp>().addPass(
      createEliminateRedundantOpsPass());
  pm.nest<DesignOp>().addPass(createCleanSelfdriversPass());
  pm.addPass(createContainersToHWPass());
  pm.addPass(hw::createVerifyInnerRefNamespacePass());
}

void circt::kanagawa::loadKanagawaHighLevelPassPipeline(mlir::PassManager &pm) {
  pm.nest<kanagawa::DesignOp>()
      .nest<kanagawa::ClassOp>()
      .nest<kanagawa::MethodOp>()
      .addPass(kanagawa::createInlineSBlocksPass());
  pm.addPass(mlir::createMem2Reg());

  // TODO @mortbopet: Add a verification pass to ensure that there are no more
  // memref.alloca's - we want all memories to be mem2reg'able, unless they are
  // member variable accesses.
  // - just add it as an illegal op.

  // Now, perform SSA maximizations.
  pm.addPass(circt::createMaximizeSSAPass());

  // SSA maximal form achieved. Reconstruct the Kanagawa sblocks.
  pm.nest<kanagawa::DesignOp>()
      .nest<kanagawa::ClassOp>()
      .nest<kanagawa::MethodOp>()
      .addPass(kanagawa::createReblockPass());
  pm.addPass(kanagawa::createArgifyBlocksPass());
  pm.addPass(createSimpleCanonicalizerPass());
}
