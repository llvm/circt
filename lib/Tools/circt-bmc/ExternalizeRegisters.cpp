//===- ExternalizeRegisters.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/PostOrderIterator.h"

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace igraph;

namespace circt {
#define GEN_PASS_DEF_EXTERNALIZEREGISTERS
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// Convert Lower To BMC pass
//===----------------------------------------------------------------------===//

namespace {
struct ExternalizeRegistersPass
    : public circt::impl::ExternalizeRegistersBase<ExternalizeRegistersPass> {
  using ExternalizeRegistersBase::ExternalizeRegistersBase;
  void runOnOperation() override;
};
} // namespace

void ExternalizeRegistersPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  DenseSet<Operation *> handled;
  DenseMap<StringAttr, SmallVector<Type>> addedInputs;
  DenseMap<StringAttr, SmallVector<Type>> addedOutputs;

  // Iterate over all instances in the instance graph. This ensures we visit
  // every module, even private top modules (private and never instantiated).
  for (auto *startNode : instanceGraph) {
    if (handled.count(startNode->getModule().getOperation()))
      continue;

    // Visit the instance subhierarchy starting at the current module, in a
    // depth-first manner. This allows us to inline child modules into parents
    // before we attempt to inline parents into their parents.
    for (InstanceGraphNode *node : llvm::post_order(startNode)) {
      if (!handled.insert(node->getModule().getOperation()).second)
        continue;

      auto module =
          dyn_cast_or_null<HWModuleOp>(node->getModule().getOperation());
      if (!module)
        continue;

      unsigned numRegs = 0;
      module->walk([&](Operation *op) {
        if (auto regOp = dyn_cast<seq::CompRegOp>(op)) {
          addedInputs[module.getSymNameAttr()].push_back(regOp.getType());
          addedOutputs[module.getSymNameAttr()].push_back(
              regOp.getInput().getType());

          OpBuilder builder(regOp);
          Value clk =
              builder.create<seq::FromClockOp>(op->getLoc(), regOp.getClk());
          regOp.getResult().replaceAllUsesWith(
              module.appendInput("", regOp.getType()).second);
          module.appendOutput("", regOp.getInput());
          regOp->erase();
          ++numRegs;
          return;
        }
        if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
          auto newInputs =
              addedInputs[instanceOp.getModuleNameAttr().getAttr()];
          auto newOutputs =
              addedOutputs[instanceOp.getModuleNameAttr().getAttr()];
          addedInputs[module.getSymNameAttr()].append(newInputs);
          addedOutputs[module.getSymNameAttr()].append(newOutputs);
          for (auto input : newInputs)
            instanceOp.getInputsMutable().append(
                module.appendInput("", input).second);
          OpBuilder builder(instanceOp);
          SmallVector<Type> resTypes(instanceOp->getResultTypes());
          resTypes.append(newOutputs);
          auto newInst = builder.create<InstanceOp>(
              instanceOp.getLoc(), resTypes, instanceOp.getInstanceNameAttr(),
              instanceOp.getModuleNameAttr(), instanceOp.getInputs(),
              instanceOp.getArgNamesAttr(), instanceOp.getResultNamesAttr(),
              instanceOp.getParametersAttr(), instanceOp.getInnerSymAttr());
          for (auto output : newInst->getResults().take_back(newOutputs.size()))
            module.appendOutput("", output);
          instanceOp.erase();
          return;
        }
      });

      module->setAttr(
          "num_regs",
          IntegerAttr::get(IntegerType::get(&getContext(), 32), numRegs));
    }
  }
}
