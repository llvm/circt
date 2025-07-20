//===- ExternalizeRegisters.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
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

private:
  DenseMap<StringAttr, SmallVector<Type>> addedInputs;
  DenseMap<StringAttr, SmallVector<StringAttr>> addedInputNames;
  DenseMap<StringAttr, SmallVector<Type>> addedOutputs;
  DenseMap<StringAttr, SmallVector<StringAttr>> addedOutputNames;
  DenseMap<StringAttr, SmallVector<Attribute>> initialValues;

  LogicalResult externalizeReg(HWModuleOp module, Operation *op, Twine regName,
                               Value clock, Attribute initState, Value reset,
                               bool isAsync, Value resetValue, Value next);
};
} // namespace

void ExternalizeRegistersPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  DenseSet<Operation *> handled;

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
          mlir::Attribute initState = {};
          if (auto initVal = regOp.getInitialValue()) {
            // Find the constant op that defines the reset value in an initial
            // block (if it exists)
            if (!initVal.getDefiningOp<seq::InitialOp>()) {
              regOp.emitError("registers with initial values not directly "
                              "defined by a seq.initial op not yet supported");
              return signalPassFailure();
            }
            if (auto constantOp = circt::seq::unwrapImmutableValue(initVal)
                                      .getDefiningOp<hw::ConstantOp>()) {
              // Fetch value from constant op - leave removing the dead op to
              // DCE
              initState = constantOp.getValueAttr();
            } else {
              regOp.emitError("registers with initial values not directly "
                              "defined by a hw.constant op in a seq.initial op "
                              "not yet supported");
              return signalPassFailure();
            }
          }
          Twine regName = regOp.getName() && !(regOp.getName().value().empty())
                              ? regOp.getName().value()
                              : "reg_" + Twine(numRegs);

          if (failed(externalizeReg(module, op, regName, regOp.getClk(),
                                    initState, regOp.getReset(), false,
                                    regOp.getResetValue(), regOp.getInput()))) {
            return signalPassFailure();
          }
          regOp->erase();
          ++numRegs;
          return;
        }
        if (auto regOp = dyn_cast<seq::FirRegOp>(op)) {
          mlir::Attribute initState = {};
          if (auto preset = regOp.getPreset()) {
            // Get preset value as initState
            initState = mlir::IntegerAttr::get(
                mlir::IntegerType::get(&getContext(), preset->getBitWidth()),
                *preset);
          }
          Twine regName = regOp.getName().empty() ? "reg_" + Twine(numRegs)
                                                  : regOp.getName();

          if (failed(externalizeReg(module, op, regName, regOp.getClk(),
                                    initState, regOp.getReset(),
                                    regOp.getIsAsync(), regOp.getResetValue(),
                                    regOp.getNext()))) {
            return signalPassFailure();
          }
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
          initialValues[module.getSymNameAttr()].append(
              initialValues[instanceOp.getModuleNameAttr().getAttr()]);
          SmallVector<Attribute> argNames(
              instanceOp.getInputNames().getValue());
          SmallVector<Attribute> resultNames(
              instanceOp.getOutputNames().getValue());

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
              instanceOp.getParametersAttr(), instanceOp.getInnerSymAttr(),
              instanceOp.getDoNotPrintAttr());
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

      module->setAttr("initial_values",
                      ArrayAttr::get(&getContext(),
                                     initialValues[module.getSymNameAttr()]));
    }
  }
}

LogicalResult ExternalizeRegistersPass::externalizeReg(
    HWModuleOp module, Operation *op, Twine regName, Value clock,
    Attribute initState, Value reset, bool isAsync, Value resetValue,
    Value next) {
  if (!isa<BlockArgument>(clock)) {
    op->emitError("only clocks directly given as block arguments "
                  "are supported");
    return failure();
  }

  OpBuilder builder(op);
  auto result = op->getResult(0);
  auto regType = result.getType();

  // If there's no initial value just add a unit attribute to maintain
  // one-to-one correspondence with module ports
  initialValues[module.getSymNameAttr()].push_back(
      initState ? initState : mlir::UnitAttr::get(&getContext()));

  StringAttr newInputName(builder.getStringAttr(regName + "_state")),
      newOutputName(builder.getStringAttr(regName + "_next"));
  addedInputs[module.getSymNameAttr()].push_back(regType);
  addedInputNames[module.getSymNameAttr()].push_back(newInputName);
  addedOutputs[module.getSymNameAttr()].push_back(next.getType());
  addedOutputNames[module.getSymNameAttr()].push_back(newOutputName);

  // Replace the register with newInput and newOutput
  auto newInput = module.appendInput(newInputName, regType).second;
  result.replaceAllUsesWith(newInput);
  if (reset) {
    if (isAsync) {
      // Async reset
      op->emitError("registers with an async reset are not yet supported");
      return failure();
    }
    // Sync reset
    auto mux = builder.create<comb::MuxOp>(op->getLoc(), regType, reset,
                                           resetValue, next);
    module.appendOutput(newOutputName, mux);
  } else {
    // No reset
    module.appendOutput(newOutputName, next);
  }

  return success();
}
