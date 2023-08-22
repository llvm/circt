//===- LowerFirMem.cpp - Seq FIRRTL memory lowering -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq FirMem ops to instances of HW generated modules.
//
//===----------------------------------------------------------------------===//

#include "FirMemLowering.h"
#include "circt/Conversion/SeqToSV.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Threading.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "lower-seq-firmem"

using namespace circt;
using namespace hw;
using llvm::MapVector;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
#define GEN_PASS_DEF_LOWERFIRMEM
#include "circt/Conversion/Passes.h.inc"

struct LowerFirMemPass : public impl::LowerFirMemBase<LowerFirMemPass> {
  void runOnOperation() override;
};
} // namespace

void LowerFirMemPass::runOnOperation() {
  auto circuit = getOperation();

  auto modules = llvm::to_vector(circuit.getOps<HWModuleOp>());
  LLVM_DEBUG(llvm::dbgs() << "Lowering memories in " << modules.size()
                          << " modules\n");

  FirMemLowering lowering(circuit);

  // Gather all `FirMemOp`s in the HW modules and group them by configuration.
  auto uniqueMems = lowering.collectMemories(modules);
  LLVM_DEBUG(llvm::dbgs() << "Found " << uniqueMems.size()
                          << " unique memory congiurations\n");
  if (uniqueMems.empty()) {
    markAllAnalysesPreserved();
    return;
  }

  // Group the list of memories that we need to update per HW module. This will
  // allow us to parallelize across HW modules.
  MapVector<HWModuleOp, SmallVector<FirMemLowering::MemoryConfig>> memsByModule;
  for (auto &[config, memOps] : uniqueMems) {
    // Create the `HWModuleGeneratedOp`s for each unique configuration.
    auto genOp = lowering.createMemoryModule(config, memOps);

    // Group memories by their parent module for parallelism.
    for (auto memOp : memOps) {
      auto parent = memOp->getParentOfType<HWModuleOp>();
      memsByModule[parent].emplace_back(&config, genOp, memOp);
    }
  }

  // Replace all `FirMemOp`s with instances of the generated module.
  mlir::parallelForEach(&getContext(), memsByModule, [&](auto pair) {
    lowering.lowerMemoriesInModule(pair.first, pair.second);
  });
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

std::unique_ptr<Pass> circt::createLowerFirMemPass() {
  return std::make_unique<LowerFirMemPass>();
}
