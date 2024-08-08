//===- SinkClockGates.cpp - Sink clock gates to its users. -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"
#include <string>

namespace circt {
namespace seq {
#define GEN_PASS_DEF_SINKCLOCKGATES
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

#define DEBUG_TYPE "sink-clock-gates"

using namespace circt;
using namespace seq;
using namespace hw;

namespace {
struct SinkClockGatesPass
    : public circt::seq::impl::SinkClockGatesBase<SinkClockGatesPass> {
  using SinkClockGatesBase::SinkClockGatesBase;

  // TODO: Can be moved to some utils !!
  /// Append inputs to the instance, and also update the referenced modules.
  void appendInputsToInstance(HWInstanceLike instance,
                              ArrayRef<std::pair<StringAttr, Value>> inputs,
                              ArrayRef<HWModuleOp> refModules) {

    SmallVector<Attribute> argNames;
    if (auto args = instance->getAttrOfType<ArrayAttr>("argNames"))
      argNames.insert(argNames.begin(), args.begin(), args.end());

    SmallVector<Value> newOperands;
    newOperands.reserve(inputs.size());
    SmallVector<std::tuple<StringAttr, Type, Location>> newInputs;
    newInputs.reserve(inputs.size());
    for (const auto &[name, value] : inputs) {
      // If no argNames attribute, donot add it.
      if (!argNames.empty())
        argNames.push_back(name);
      // Additional operands to the Instance op.
      newOperands.push_back(value);
      // Prepare the additional input ports that need to be added to the
      // referenced module.
      newInputs.emplace_back(name, value.getType(), value.getLoc());
    }
    // Add the new operands to the instance.
    instance->insertOperands(instance->getNumOperands(), newOperands);
    if (!argNames.empty())
      instance->setAttr("argNames",
                        ArrayAttr::get(instance->getContext(), argNames));
    // Update all the modules with the new input ports.
    for (auto refMod : refModules) {
      refMod.appendPorts(newInputs, {});
    }
  }

  /// Get the referenced modules for an instance.
  SmallVector<HWModuleOp, 1> getReferencedModules(HWInstanceLike inst) {
    SmallVector<HWModuleOp, 1> refMods;
    for (auto refModName : inst.getReferencedModuleNamesAttr()) {
      auto *node = graph->lookup(cast<StringAttr>(refModName));
      if (auto instModule = dyn_cast_or_null<HWModuleOp>(*node->getModule()))
        refMods.push_back(instModule);
    }
    return refMods;
  }

  /// If the `argNames` attribute exists, return it.
  SmallVector<StringAttr> getInstanceArgNames(HWInstanceLike inst) {
    SmallVector<StringAttr> argNames;
    if (auto args = inst->getAttrOfType<ArrayAttr>("argNames")) {
      auto attr = args.getAsRange<StringAttr>();
      argNames.insert(argNames.begin(), attr.begin(), attr.end());
    }
    return argNames;
  }

  /// Combine two consecutive clock_gate ops, into one.
  void collapseConsecutiveClockGates(ClockGateOp driverGate,
                                     ClockGateOp userGate) {
    mlir::OpBuilder builder(context);
    builder.setInsertionPoint(userGate);
    Value enable = getEnable(driverGate);
    enable = builder.create<comb::AndOp>(userGate->getLoc(), enable,
                                         userGate.getEnable());
    userGate.getInputMutable().assign(driverGate.getInput());
    userGate.getEnableMutable().assign(enable);
  }

  /// If the clock_gate op has `test_enable`, combine it with the `enable` and
  /// return the final enable.
  Value getEnable(ClockGateOp clkGate) {
    Value enable = clkGate.getEnable();
    // If there is testEnable, or it.
    if (auto testEnable = clkGate.getTestEnable()) {
      OpBuilder builder(clkGate.getContext());
      builder.setInsertionPointAfter(clkGate);
      enable =
          builder.create<comb::OrOp>(clkGate->getLoc(), enable, testEnable);
    }
    return enable;
  }

  void runOnOperation() override;

private:
  InstanceGraph *graph;
  DenseSet<Operation *> opsToErase;
  mlir::MLIRContext *context;
};

void SinkClockGatesPass::runOnOperation() {
  graph = &getAnalysis<circt::hw::InstanceGraph>();
  DenseSet<HWInstanceLike> instancesWithGatedClk;
  DenseSet<Operation *> opsWithGatedClock;
  context = getOperation().getContext();

  // Record all the instances that have clock_gate as an input. This will be
  // used to push the clock_gate down through the hierarchy to its users.
  // This also collapses consecutive clock gates.
  auto findInstancesWithGatedClock = [&](HWModuleOp hwModule) {
    hwModule->walk([&](ClockGateOp clkGate) {
      bool canErase = true;
      for (auto &use : clkGate->getUses()) {
        auto *useOp = use.getOwner();
        if (auto inst = dyn_cast<HWInstanceLike>(useOp))
          instancesWithGatedClk.insert(inst);
        else if (auto userGate = dyn_cast<ClockGateOp>(useOp)) {
          collapseConsecutiveClockGates(clkGate, userGate);
        } else {
          canErase = false;
          opsWithGatedClock.insert(useOp);
        }
      }
      if (canErase)
        opsToErase.insert(clkGate);
    });
  };
  // Seed phase, find the instances to start the traversal from.
  // These instances have a gated clock as input.
  getOperation()->walk(
      [&](hw::HWModuleOp hwModule) { findInstancesWithGatedClock(hwModule); });

  // Transformation phase, push the clock_gate through the instances and find
  // all the indirect instances that have a gated clock.

  // Iterate until none of the instances have a gated clock as input.
  while (!instancesWithGatedClk.empty()) {

    // Remove an entry from the set.
    auto entry = instancesWithGatedClk.begin();
    HWInstanceLike inst = *entry;
    instancesWithGatedClk.erase(entry);

    // Now, find all the gated clocks that are input to the instance, and
    // duplicate it at the referenced module.

    SmallVector<HWModuleOp> refMods = getReferencedModules(inst);
    // Index of the instance operands that are a gated clock.
    SmallVector<unsigned> gatedClkPorts;
    // The new enable ports that will be added to the instance.
    SmallVector<std::pair<StringAttr, Value>> enablePorts;
    // The instance may or maynot have argNames attribute.
    SmallVector<StringAttr> argNames = getInstanceArgNames(inst);
    // Iterate over all instance operands to find a gated clock.
    for (auto [index, in] : llvm::enumerate(inst->getOperands())) {
      if (!isa<seq::ClockType>(in.getType()))
        continue;
      auto clkGate = dyn_cast_or_null<ClockGateOp>(in.getDefiningOp());
      if (!clkGate)
        continue;
      gatedClkPorts.push_back(index);
      Value enable = getEnable(clkGate);

      // Replace the gated clock with the original base clock.
      inst->setOperand(index, clkGate.getInput());
      // Create a name for the new enable port.
      auto clockIndexStr = std::to_string(index);
      enablePorts.emplace_back(
          StringAttr::get(context,
                          (argNames.empty()
                               ? "enable_"
                               : argNames[index].getValue() + "_enable_") +
                              clockIndexStr),
          enable);
    }
    unsigned oldNumInputs = inst->getNumOperands();
    // Now update the instance and all the referenced modules with the new
    // enable ports.
    appendInputsToInstance(inst, enablePorts, refMods);
    // Third phase, Once all the modules are updated with the additional enable
    // ports, duplicate the clock_gate op in the referenced module.
    for (auto mod : refMods) {
      auto *block = mod.getBodyBlock();
      auto builder = mlir::OpBuilder::atBlockBegin(block);
      for (auto [index, clkPort] : llvm::enumerate(gatedClkPorts)) {
        auto clk = block->getArgument(clkPort);
        assert(isa<ClockType>(clk.getType()));
        auto enable = block->getArgument(oldNumInputs + index);
        auto clkGate = builder.create<ClockGateOp>(enable.getLoc(), clk, enable,
                                                   Value(), hw::InnerSymAttr());
        // Replace all the original clock users with the gated clock.
        clk.replaceAllUsesExcept(clkGate.getResult(), clkGate);
      }
      // After adding the new clock gates, scan the module for instances that
      // have a gated clock as input, and continue sinking the clock gates.
      findInstancesWithGatedClock(mod);
    }
  }
  for (auto *gatedOp : opsWithGatedClock) {
    LLVM_DEBUG(llvm::dbgs() << "\n Gated clock user: " << gatedOp);
    // TODO: Handle all possible ops here, to replace the clocked op with the
    // ungated clock and sink the enable into the op.
    if (auto compReg = dyn_cast<CompRegOp>(gatedOp)) {
      Value clk = compReg.getClk();
      auto gate = dyn_cast_or_null<ClockGateOp>(clk.getDefiningOp());
      if (!gate)
        continue;
      clk = gate.getInput();
      Value enable = getEnable(gate);
      OpBuilder builder(context);
      builder.setInsertionPoint(compReg);
      auto compEn = builder.create<CompRegClockEnabledOp>(
          compReg->getLoc(), compReg.getInput(), clk, enable,
          compReg.getNameAttr(), compReg.getReset(), compReg.getResetValue(),
          compReg.getPowerOnValue(), compReg.getInnerSymAttr());
      compReg->replaceAllUsesWith(compEn);
      opsToErase.insert(compReg);
    }
  }
  for (auto *op : opsToErase)
    op->erase();
}
} // anonymous namespace

std::unique_ptr<Pass> circt::seq::createSinkClockGatesPass() {
  return std::make_unique<SinkClockGatesPass>();
}
