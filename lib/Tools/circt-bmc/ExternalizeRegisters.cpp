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
#include "circt/Dialect/Seq/SeqTypes.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/LLVM.h"
#include "circt/Support/Namespace.h"
#include "circt/Tools/circt-bmc/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/Twine.h"
#include <optional>

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace igraph;

namespace circt {
#define GEN_PASS_DEF_EXTERNALIZEREGISTERS
#include "circt/Tools/circt-bmc/Passes.h.inc"
} // namespace circt

//===----------------------------------------------------------------------===//
// Externalize Registers Pass
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
  DenseMap<StringAttr, SmallVector<StringAttr>> addedInputNames;
  DenseMap<StringAttr, SmallVector<Type>> addedOutputs;
  DenseMap<StringAttr, SmallVector<StringAttr>> addedOutputNames;

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
      bool foundClk = false;
      for (auto ty : module.getInputTypes()) {
        if (isa<seq::ClockType>(ty)) {
          if (foundClk) {
            module.emitError("modules with multiple clocks not yet supported");
            return signalPassFailure();
          }
          foundClk = true;
        }
      }
      module->walk([&](Operation *op) {
        if (auto regOp = dyn_cast<seq::CompRegOp>(op)) {
          if (!isa<BlockArgument>(regOp.getClk())) {
            regOp.emitError("only clocks directly given as block arguments "
                            "are supported");
            return signalPassFailure();
          }
          if (regOp.getReset()) {
            regOp.emitError("registers with reset signals not yet supported");
            return signalPassFailure();
          }
          if (regOp.getInitialValue()) {
            regOp.emitError("registers with initial values not yet supported");
            return signalPassFailure();
          }
          addedInputs[module.getSymNameAttr()].push_back(regOp.getType());
          addedOutputs[module.getSymNameAttr()].push_back(
              regOp.getInput().getType());
          OpBuilder builder(regOp);
          auto regName = regOp.getName();
          StringAttr newInputName, newOutputName;
          if (regName && !regName.value().empty()) {
            newInputName = builder.getStringAttr(regName.value() + "_state");
            newOutputName = builder.getStringAttr(regName.value() + "_input");
          } else {
            newInputName =
                builder.getStringAttr("reg_" + Twine(numRegs) + "_state");
            newOutputName =
                builder.getStringAttr("reg_" + Twine(numRegs) + "_input");
          }
          addedInputNames[module.getSymNameAttr()].push_back(newInputName);
          addedOutputNames[module.getSymNameAttr()].push_back(newOutputName);

          regOp.getResult().replaceAllUsesWith(
              module.appendInput(newInputName, regOp.getType()).second);
          module.appendOutput(newOutputName, regOp.getInput());
          regOp->erase();
          ++numRegs;
          return;
        }
        if (auto instanceOp = dyn_cast<InstanceOp>(op)) {
          OpBuilder builder(instanceOp);
          auto newInputs =
              addedInputs[instanceOp.getModuleNameAttr().getAttr()];
          auto newInputNames =
              addedInputNames[instanceOp.getModuleNameAttr().getAttr()];
          auto newOutputs =
              addedOutputs[instanceOp.getModuleNameAttr().getAttr()];
          auto newOutputNames =
              addedOutputNames[instanceOp.getModuleNameAttr().getAttr()];
          addedInputs[module.getSymNameAttr()].append(newInputs);
          addedInputNames[module.getSymNameAttr()].append(newInputNames);
          addedOutputs[module.getSymNameAttr()].append(newOutputs);
          addedOutputNames[module.getSymNameAttr()].append(newOutputNames);
          SmallVector<Attribute> argNames(instanceOp.getArgNamesAttr().begin(),
                                          instanceOp.getArgNamesAttr().end());
          SmallVector<Attribute> resultNames(
              instanceOp.getResultNamesAttr().begin(),
              instanceOp.getResultNamesAttr().end());

          for (auto [input, name] : zip_equal(newInputs, newInputNames)) {
            instanceOp.getInputsMutable().append(
                module.appendInput(name, input).second);
            argNames.push_back(name);
          }
          for (auto outputName : newOutputNames) {
            resultNames.push_back(outputName);
          }
          SmallVector<Type> resTypes(instanceOp->getResultTypes());
          resTypes.append(newOutputs);
          auto newInst = builder.create<InstanceOp>(
              instanceOp.getLoc(), resTypes, instanceOp.getInstanceNameAttr(),
              instanceOp.getModuleNameAttr(), instanceOp.getInputs(),
              builder.getArrayAttr(argNames), builder.getArrayAttr(resultNames),
              instanceOp.getParametersAttr(), instanceOp.getInnerSymAttr());
          for (auto [output, name] :
               zip(newInst->getResults().take_back(newOutputs.size()),
                   newOutputNames))
            module.appendOutput(name, output);
          numRegs += newInputs.size();
          instanceOp.replaceAllUsesWith(
              newInst.getResults().take_front(instanceOp->getNumResults()));
          instanceGraph.replaceInstance(instanceOp, newInst);
          instanceOp->erase();
          return;
        }
      });

      module->setAttr(
          "num_regs",
          IntegerAttr::get(IntegerType::get(&getContext(), 32), numRegs));
    }
  }
}
