//===- pipelines.cpp - Arcilator lowering pipelines -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the 'arcilator' lowering pipelines.
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/arcilator/pipelines.h"

#include "circt/Conversion/ArcToLLVM.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/ConvertToArcs.h"
#include "circt/Conversion/Passes.h"
#include "circt/Conversion/SeqToSV.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Emit/EmitPasses.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/OM/OMPasses.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Sim/SimPasses.h"
#include "circt/Support/Passes.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace circt;
using namespace arc;

void circt::populateArcPreprocessingPipeline(
    OpPassManager &pm, const ArcPreprocessingOptions &options) {
  pm.addPass(om::createStripOMPass());
  pm.addPass(emit::createStripEmitPass());
  pm.addPass(createLowerFirMemPass());
  pm.addPass(createLowerVerifSimulationsPass());
  {
    arc::AddTapsOptions opts;
    opts.tapPorts = options.observePorts;
    opts.tapWires = options.observeWires;
    opts.tapNamedValues = options.observeNamedValues;
    pm.addPass(arc::createAddTapsPass(opts));
  }
  pm.addPass(arc::createStripSVPass(options.asyncResetsAsSync));
  {
    arc::InferMemoriesOptions opts;
    opts.tapPorts = options.observePorts;
    opts.tapMemories = options.observeMemories;
    pm.addPass(arc::createInferMemoriesPass(opts));
  }
  pm.addPass(sim::createLowerDPIFunc());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
}

void circt::populateArcConversionPipeline(OpPassManager &pm,
                                          const ArcConversionOptions &options) {
  {
    ConvertToArcsPassOptions opts;
    opts.tapRegisters = options.observeRegisters;
    pm.addPass(createConvertToArcsPass(opts));
  }
  if (options.shouldDedup)
    pm.addPass(arc::createDedupPass());
  pm.addPass(hw::createFlattenModules());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
}

void circt::populateArcOptimizationPipeline(
    OpPassManager &pm, const ArcOptimizationOptions &options) {
  // Perform arc-level optimizations that are not specific to software
  // simulation.
  pm.addPass(arc::createSplitLoopsPass());
  if (options.shouldDedup)
    pm.addPass(arc::createDedupPass());
  {
    arc::InferStatePropertiesOptions opts;
    opts.detectEnables = options.shouldDetectEnables;
    opts.detectResets = options.shouldDetectResets;
    pm.addPass(arc::createInferStateProperties(opts));
  }
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
  pm.addNestedPass<hw::HWModuleOp>(arc::createMergeTaps());
  if (options.shouldMakeLUTs)
    pm.addPass(arc::createMakeTablesPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());

  // Now some arguments may be unused because reset conditions are not passed as
  // inputs anymore pm.addPass(arc::createRemoveUnusedArcArgumentsPass());
  // Because we replace a lot of StateOp inputs with constants in the enable
  // patterns we may be able to sink a lot of them
  // TODO: maybe merge RemoveUnusedArcArguments with SinkInputs?
  // pm.addPass(arc::createSinkInputsPass());
  // pm.addPass(createCSEPass());
  // pm.addPass(createSimpleCanonicalizerPass());
  // Removing some muxes etc. may lead to additional dedup opportunities
  // if (options.shouldDedup)
  // pm.addPass(arc::createDedupPass());
}

void circt::populateArcStateLoweringPipeline(
    OpPassManager &pm, const ArcStateLoweringOptions &options) {
  pm.addPass(arc::createLowerStatePass());

  // TODO: LowerClocksToFuncsPass might not properly consider scf.if operations
  // (or nested regions in general) and thus errors out when muxes are also
  // converted in the hw.module or arc.model
  // TODO: InlineArcs seems to not properly handle scf.if operations, thus the
  // following is commented out
  // pm.addPass(arc::createMuxToControlFlowPass());
  if (options.shouldInline)
    pm.addPass(arc::createInlineArcsPass());

  pm.addPass(arc::createMergeIfsPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
}

void circt::populateArcStateAllocationPipeline(
    OpPassManager &pm, const ArcStateAllocationOptions &options) {
  pm.addPass(arc::createLowerArcsToFuncsPass());
  pm.nest<arc::ModelOp>().addPass(arc::createAllocateStatePass());
  pm.addPass(arc::createLowerClocksToFuncsPass()); // no CSE between state alloc
                                                   // and clock func lowering
  if (options.splitFuncsThreshold.getNumOccurrences()) {
    pm.addPass(arc::createSplitFuncs({options.splitFuncsThreshold}));
  }
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
}

void circt::populateArcToLLVMPipeline(OpPassManager &pm) {
  pm.addPass(createLowerArcToLLVMPass());
  pm.addPass(createCSEPass());
  pm.addPass(arc::createArcCanonicalizerPass());
}
