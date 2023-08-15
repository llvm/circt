//===- LowerSeqToSV.cpp - Seq to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq ops to SV.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SeqToSV.h"
#include "FirMemLowering.h"
#include "FirRegLowering.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/IntervalMap.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace seq;
using hw::HWModuleOp;
using llvm::MapVector;

namespace {
#define GEN_PASS_DEF_LOWERSEQFIRRTLTOSV
#include "circt/Conversion/Passes.h.inc"

struct SeqFIRRTLToSVPass
    : public impl::LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass> {
  void runOnOperation() override;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::disableRegRandomization;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::emitSeparateAlwaysBlocks;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::LowerSeqFIRRTLToSVBase;
  using LowerSeqFIRRTLToSVBase<SeqFIRRTLToSVPass>::numSubaccessRestored;
};
} // anonymous namespace

void SeqFIRRTLToSVPass::runOnOperation() {
  auto circuit = getOperation();
  auto modules = llvm::to_vector(circuit.getOps<HWModuleOp>());

  FirMemLowering memLowering(circuit);

  // Identify memories and group them by module.
  auto uniqueMems = memLowering.collectMemories(modules);
  MapVector<HWModuleOp, SmallVector<FirMemLowering::MemoryConfig>> memsByModule;
  for (auto &[config, memOps] : uniqueMems) {
    // Create the `HWModuleGeneratedOp`s for each unique configuration.
    auto genOp = memLowering.createMemoryModule(config, memOps);

    // Group memories by their parent module for parallelism.
    for (auto memOp : memOps) {
      auto parent = memOp->getParentOfType<HWModuleOp>();
      memsByModule[parent].emplace_back(&config, genOp, memOp);
    }
  }

  // Lower memories and registers in modules in parallel.
  mlir::parallelForEach(&getContext(), modules, [&](HWModuleOp module) {
    FirRegLowering regLowering(module, disableRegRandomization,
                               emitSeparateAlwaysBlocks);
    regLowering.lower();
    numSubaccessRestored += regLowering.numSubaccessRestored;

    if (auto *it = memsByModule.find(module); it != memsByModule.end())
      memLowering.lowerMemoriesInModule(module, it->second);
  });
}

std::unique_ptr<Pass>
circt::createSeqFIRRTLLowerToSVPass(const LowerSeqFIRRTLToSVOptions &options) {
  return std::make_unique<SeqFIRRTLToSVPass>(options);
}
