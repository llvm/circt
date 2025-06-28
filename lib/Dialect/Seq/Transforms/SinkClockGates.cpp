//===----------SinkClockGates.cpp - Sink clock gates to its users.---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a transformation that "tries" to sink as many clock
// gates as possible. This is a best-effort transformation, no assumptions can
// be made post this transformation.
// A clock gate can be sunk through an instance to a module, if all the
// instances of the module are known and they reference a unique HWModuleOp. If
// there are instances of a module, that have gated clock while others have
// un-gated clock, then the un-gated versions add a constant true enable
// condition. This analysis relies on the fact that the HW instance graph cannot
// have cycles/recursion. It is guaranteed that any instance operand will be
// updated only once. This assert checks for it:
// `assert(llvm::isa_and_nonnull<ConstantOp>(def))`
// This doesnot handle gated clocks returned from an output port yet.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortImplementation.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Dialect/Seq/SeqTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
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

  /// Append input ports to the module and all its instances.
  /// Add the given values as operands to the `instance`. Add constant true op
  /// as input to the other instances of the module.
  SmallVector<Value> appendInputsToModuleAndInstances(
      HWInstanceLike instance, ArrayRef<HWInstanceLike> moduleInstances,
      ArrayRef<std::pair<StringAttr, Value>> inputs, HWModuleOp refMod) {

    if (inputs.empty())
      return {};
    SmallVector<Attribute> argNames;
    if (auto args = instance->getAttrOfType<ArrayAttr>("argNames"))
      argNames.insert(argNames.begin(), args.begin(), args.end());

    SmallVector<Value> newOperands;
    newOperands.reserve(inputs.size());
    SmallVector<PortInfo> newInputs;
    newInputs.reserve(inputs.size());
    for (const auto &[name, value] : inputs) {
      // If no argNames attribute, donot add it.
      if (!argNames.empty())
        argNames.push_back(name);
      // Additional operands to the Instance op.
      newOperands.push_back(value);
      // Prepare the additional input ports that need to be added to the
      // referenced module.
      PortInfo p;
      p.name = name;
      p.type = value.getType();
      p.dir = ModulePort::Direction::Input;
      p.loc = value.getLoc();
      newInputs.push_back(p);
    }
    // Add the new operands to the instance.
    instance->insertOperands(instance->getNumOperands(), newOperands);
    if (!argNames.empty())
      instance->setAttr("argNames",
                        ArrayAttr::get(instance->getContext(), argNames));

    // Now update other instances of the module.
    for (auto otherInstance : moduleInstances) {
      if (otherInstance == instance)
        continue;
      SmallVector<Value, 2> otherInstanceNewOperands;
      otherInstanceNewOperands.reserve(inputs.size());
      mlir::OpBuilder builder(otherInstance);
      for (unsigned i = 0; i < inputs.size(); ++i)
        otherInstanceNewOperands.emplace_back(builder.create<ConstantOp>(
            otherInstance->getLoc(), builder.getI1Type(), 1));

      otherInstance->insertOperands(otherInstance->getNumOperands(),
                                    otherInstanceNewOperands);
      if (otherInstance->hasAttr("argNames"))
        otherInstance->setAttr(
            "argNames", ArrayAttr::get(instance->getContext(), argNames));
    }
    SmallVector<Value> inputPorts;
    // Update all the modules with the new input ports.
    refMod.appendPorts(newInputs, {}, inputPorts);
    return inputPorts;
  }

  /// Get the referenced module for an instance, and add all the instances of
  /// the module to `instances`.
  HWModuleOp getReferencedModules(HWInstanceLike inst,
                                  SmallVectorImpl<HWInstanceLike> &instances) {
    HWModuleOp instModule = {};
    auto refModNames = inst.getReferencedModuleNamesAttr();
    if (refModNames.size() != 1) {
      inst.emitWarning("expected an instance with a single reference");
      return {};
    }
    auto *node = graph->lookup(cast<StringAttr>(*refModNames.begin()));
    if (node)
      instModule = dyn_cast_or_null<HWModuleOp>(*node->getModule());

    // Cannot modify public module signature.
    if (!instModule || instModule.isPublic())
      return {};

    for (auto *use : node->uses()) {
      auto otherInstance = use->getInstance<HWInstanceLike>();
      if (!otherInstance) {
        instModule.emitWarning("Cannot find all instances");
        return {};
      }
      if (otherInstance.getReferencedModuleNamesAttr().size() != 1) {

        otherInstance.emitWarning(
            "expected an instance with a single reference");
        return {};
      }
      instances.push_back(otherInstance);
    }

    return instModule;
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
    auto *context = &getContext();
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
    auto *context = &getContext();
    Value enable = clkGate.getEnable();
    // If there is testEnable, or it.
    if (auto testEnable = clkGate.getTestEnable()) {
      OpBuilder builder(context);
      builder.setInsertionPointAfter(clkGate);
      enable =
          builder.create<comb::OrOp>(clkGate->getLoc(), enable, testEnable);
    }
    return enable;
  }

  void runOnOperation() override;

private:
  using ClockEnablePortPairs = DenseMap<unsigned, unsigned>;
  InstanceGraph *graph;
  DenseSet<Operation *> opsToErase;
  DenseMap<HWModuleOp, ClockEnablePortPairs> moduleClockGateEnableMap;
};

void SinkClockGatesPass::runOnOperation() {
  graph = &getAnalysis<circt::hw::InstanceGraph>();
  // A setvector is required for a deterministic output.
  SetVector<HWInstanceLike> instancesWithGatedClk;
  DenseSet<Operation *> opsWithGatedClock;
  auto *context = &getContext();

  // Record all the instances that have clock_gate as an input. This will be
  // used to push the clock_gate down through the hierarchy to its users.
  // This also collapses consecutive clock gates.
  auto findInstancesWithGatedClock = [&](HWModuleOp hwModule) {
    hwModule->walk([&](ClockGateOp clkGate) {
      for (auto &use : clkGate->getUses()) {
        auto *useOp = use.getOwner();
        if (auto inst = dyn_cast<HWInstanceLike>(useOp))
          instancesWithGatedClk.insert(inst);
        else if (auto userGate = dyn_cast<ClockGateOp>(useOp)) {
          collapseConsecutiveClockGates(clkGate, userGate);
        } else {
          opsWithGatedClock.insert(useOp);
        }
      }
    });
  };
  // Seed phase, find the instances to start the traversal from.
  // These instances have a gated clock as input.
  for (auto *node : *graph) {
    if (auto hwModule = node->getModule<HWModuleOp>())
      findInstancesWithGatedClock(hwModule);
  }
  // Get the initial set of clock gates which this pass tries to sink.
  numClockGatesConverted = instancesWithGatedClk.size();

  // Transformation phase, push the clock_gate through the instances and find
  // all the indirect instances that have a gated clock.

  // Iterate until none of the instances have a gated clock as input.
  while (!instancesWithGatedClk.empty()) {

    // Remove an entry from the set.
    const auto *entry = instancesWithGatedClk.begin();
    HWInstanceLike inst = *entry;
    instancesWithGatedClk.erase(entry);

    // Now, find all the gated clocks that are input to the instance, and
    // duplicate it at the referenced module.

    SmallVector<HWInstanceLike, 2> moduleInstances;
    HWModuleOp refMod = getReferencedModules(inst, moduleInstances);
    if (!refMod) {
      // If we cannot identify a unique HWModuleOp, ignore the instance.
      // Multiple references and extern modules cannot be handled, and gated
      // clocks cannot be sunk into them.
      continue;
    }
    // Index of the instance operands that are a gated clock.
    SmallVector<unsigned> gatedClkPorts;
    // The new enable ports that will be added to the instance.
    SmallVector<std::pair<StringAttr, Value>> enablePorts;
    // The instance may or maynot have argNames attribute.
    SmallVector<StringAttr> argNames = getInstanceArgNames(inst);
    unsigned enablePortNum = inst->getNumOperands();
    ClockEnablePortPairs &clockEnablePortIndices =
        moduleClockGateEnableMap[refMod];
    // Iterate over all instance operands to find a gated clock.
    for (auto [index, in] : llvm::enumerate(inst->getOperands())) {
      if (!isa<seq::ClockType>(in.getType()))
        continue;
      auto clkGate = dyn_cast_or_null<ClockGateOp>(in.getDefiningOp());
      if (!clkGate)
        continue;
      if (clkGate->hasOneUse())
        opsToErase.insert(clkGate);
      // Replace the gated clock with the original base clock.
      inst->setOperand(index, clkGate.getInput());
      Value enable = getEnable(clkGate);
      if (clockEnablePortIndices.contains(index)) {
        // The enable for this clock is already added, (from another instance of
        // the module).
        auto oldEnablePort = clockEnablePortIndices[index];
        auto oldEnableVal = inst->getOperand(oldEnablePort);
        auto *def = oldEnableVal.getDefiningOp();
        assert(llvm::isa_and_nonnull<ConstantOp>(def));
        inst->setOperand(oldEnablePort, enable);
        if (def->getUses().empty())
          opsToErase.insert(def);
        continue;
      }

      clockEnablePortIndices[index] = enablePortNum++;
      gatedClkPorts.push_back(index);

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
    if (enablePorts.empty())
      continue;

    // Now update the instance and all the referenced modules with the new
    // enable ports.
    auto newEnableInputs = appendInputsToModuleAndInstances(
        inst, moduleInstances, enablePorts, refMod);
    // Third phase, Once all the modules are updated with the additional enable
    // ports, duplicate the clock_gate op in the referenced module.
    auto *block = refMod.getBodyBlock();
    auto builder = mlir::OpBuilder::atBlockBegin(block);
    assert(gatedClkPorts.size() == newEnableInputs.size());
    for (auto [index, clkPort] : llvm::enumerate(gatedClkPorts)) {
      auto clk = block->getArgument(clkPort);
      assert(isa<ClockType>(clk.getType()));
      auto enable = newEnableInputs[index];
      auto clkGate = builder.create<ClockGateOp>(enable.getLoc(), clk, enable,
                                                 Value(), hw::InnerSymAttr());
      // Replace all the original clock users with the gated clock.
      clk.replaceAllUsesExcept(clkGate.getResult(), clkGate);
    }
    // After adding the new clock gates, scan the module for instances that
    // have a gated clock as input, and continue sinking the clock gates.
    findInstancesWithGatedClock(refMod);
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
  moduleClockGateEnableMap.clear();
}
} // anonymous namespace

std::unique_ptr<Pass> circt::seq::createSinkClockGatesPass() {
  return std::make_unique<SinkClockGatesPass>();
}
