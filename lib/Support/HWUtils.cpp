//===- HWUtils.cpp - HW Rewriting Utilities ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Support/HWUtils.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace circt::igraph;
using namespace hw;
using namespace mlir;

void circt::appendPorts(HWModuleOp module, ArrayRef<PortInfo> ports,
                        ArrayRef<Value> outputVals,
                        SmallVectorImpl<Value> &inputVals) {

  auto *context = module->getContext();
  auto oldType = module.getHWModuleType();
  auto *body = module.getBodyBlock();

  ArrayRef<ModulePort> oldPorts = oldType.getPorts();

  unsigned newPortsCount = oldPorts.size() + ports.size();

  SmallVector<ModulePort> newPorts;
  newPorts.reserve(newPortsCount);
  newPorts.append(oldPorts.begin(), oldPorts.end());

  SmallVector<Attribute> newResultLocs;
  if (!outputVals.empty()) {
    // Update the output op.
    auto oldResultLocs = module.getResultLocs();
    if (oldResultLocs.has_value())
      newResultLocs.append(oldResultLocs->getValue().begin(),
                           oldResultLocs->getValue().end());
    auto outputOp = cast<OutputOp>(body->getTerminator());
    auto oldOperands = outputOp->getOperands();

    SmallVector<Value> newOutputOperands;
    llvm::append_range(newOutputOperands, oldOperands);
    llvm::append_range(newOutputOperands, outputVals);
    outputOp->setOperands(newOutputOperands);
  }
  auto oldPortAttrs = module.getAllPortAttrs();
  SmallVector<Attribute> newPortAttrs(oldPortAttrs);
  if (!newPortAttrs.empty())
    newPortAttrs.append(ports.size(), DictionaryAttr::get(context));
  inputVals.reserve(ports.size() - outputVals.size());

  for (auto &portInfo : ports) {
    assert(!portInfo.isInOut() && "can only handle input or output ports");
    if (portInfo.isInput())
      inputVals.push_back(body->addArgument(portInfo.type, portInfo.loc));
    else if (portInfo.isOutput())
      newResultLocs.emplace_back(portInfo.loc);
    newPorts.push_back(portInfo);
    if (portInfo.attrs) {

      newPortAttrs[newPorts.size() - 1] = portInfo.attrs;
    }
  }

  if (!outputVals.empty())
    module.setResultLocsAttr(ArrayAttr::get(context, newResultLocs));

  auto newType = ModuleType::get(context, newPorts);

  module.setModuleType(newType);
  if (newPortAttrs.empty())
    return;
  assert(newPortAttrs.size() == newPorts.size());
  module.setAllPortAttrs(newPortAttrs);
}

/// Create an identical instance to a module.
static InstanceOpInterface rewriteInstance(OpBuilder &builder,
                                           InstanceOpInterface oldInst,
                                           HWModuleLike target,
                                           SmallVector<Value> newOperands) {
  auto moduleTy = target.getHWModuleType();
  return TypeSwitch<Operation *, InstanceOpInterface>(oldInst)
      .Case<InstanceOp>([&](auto instOp) -> InstanceOpInterface {
        return InstanceOp::create(
            builder, oldInst.getLoc(), moduleTy.getOutputTypes(),
            instOp.getInstanceNameAttr(), instOp.getModuleNameAttr(),
            newOperands, builder.getArrayAttr(moduleTy.getInputNames()),
            builder.getArrayAttr(moduleTy.getOutputNames()),
            instOp.getParametersAttr(), instOp.getInnerSymAttr(),
            instOp.getDoNotPrintAttr());
      })
      .Default([](auto op) -> InstanceOpInterface {
        llvm_unreachable("unknown instance op");
        return {};
      });
}

/// Helper function to remove ports from a HWModuleLike operation.
/// This handles both HWModuleOp and HWModuleExternOp.
static void
removePortsImpl(HWModuleLike module, igraph::InstanceGraph &instanceGraph,
                const std::function<bool(const hw::PortInfo &)> &shouldRemove,
                const std::function<bool(BlockArgument)> &dropModuleArg,
                const std::function<bool(Operation *, unsigned)> &dropResult) {
  auto *context = module->getContext();
  auto oldType = module.getHWModuleType();

  ArrayRef<ModulePort> oldPorts = oldType.getPorts();
  SmallVector<unsigned> erasedInputIndices;
  SmallVector<unsigned> erasedOutputIndices;
  DenseSet<unsigned> erasedOutputs;

  // Collect ports to keep and track removed inputs/outputs
  SmallVector<Attribute> newPortAttrs;
  auto oldPortAttrs = module.getAllPortAttrs();

  // For HWModuleOp, track block arguments and output operands
  SmallVector<Value> removedInputs;
  SmallVector<Value> newOutputOperands;
  SmallVector<Attribute> newResultLocs;

  Block *body = nullptr;
  OutputOp outputOp;
  std::optional<ArrayAttr> oldResultLocs;

  if (auto hwMod = dyn_cast<HWModuleOp>(module.getOperation())) {
    body = hwMod.getBodyBlock();
    outputOp = cast<OutputOp>(body->getTerminator());
    oldResultLocs = hwMod.getResultLocs();
  }

  unsigned inputIndex = 0;
  unsigned outputIndex = 0;

  for (unsigned i = 0; i < oldPorts.size(); ++i) {
    const auto &port = oldPorts[i];

    // Convert ModulePort to PortInfo for the predicate
    hw::PortInfo portInfo = module.getPort(i);

    if (shouldRemove(portInfo)) {
      // Port should be removed
      if (port.dir == ModulePort::Direction::Input) {
        erasedInputIndices.push_back(inputIndex);
        if (body) {
          removedInputs.push_back(body->getArgument(inputIndex));
        }
        inputIndex++;
      } else if (port.dir == ModulePort::Direction::Output) {
        erasedOutputIndices.push_back(outputIndex);
        erasedOutputs.insert(outputIndex);
        outputIndex++;
      }
    } else {
      // Port should be kept
      if (!oldPortAttrs.empty()) {
        newPortAttrs.push_back(oldPortAttrs[i]);
      }

      if (port.dir == ModulePort::Direction::Input) {
        inputIndex++;
      } else if (port.dir == ModulePort::Direction::Output) {
        if (outputOp) {
          newOutputOperands.push_back(outputOp->getOperand(outputIndex));
          if (oldResultLocs.has_value()) {
            newResultLocs.push_back(oldResultLocs->getValue()[outputIndex]);
          }
        }
        outputIndex++;
      }
    }
  }

  // Compute new port list and port locations by filtering out ports that should
  // be removed
  SmallVector<ModulePort> newPorts;
  SmallVector<Attribute> newPortLocs;

  // Get all port locations (different for HWModuleOp vs HWModuleExternOp)
  SmallVector<Location> allPortLocs = module.getAllPortLocs();

  for (unsigned i = 0; i < oldPorts.size(); ++i) {
    hw::PortInfo portInfo = module.getPort(i);
    if (!shouldRemove(portInfo)) {
      newPorts.push_back(oldPorts[i]);
      if (i < allPortLocs.size()) {
        newPortLocs.push_back(allPortLocs[i]);
      }
    }
  }

  // For HWModuleOp, handle the body (block arguments and output op)
  if (body) {
    auto hwMod = cast<HWModuleOp>(module.getOperation());

    // Remove the input block arguments in reverse order to maintain indices
    for (auto it = removedInputs.rbegin(); it != removedInputs.rend(); ++it) {
      auto blockArg = cast<BlockArgument>(*it);
      if (dropModuleArg)
        dropModuleArg(blockArg);
      blockArg.dropAllUses();
      body->eraseArgument(blockArg.getArgNumber());
    }

    // Update the output op with new operands
    outputOp->setOperands(newOutputOperands);

    // Update result locations if they exist
    if (oldResultLocs.has_value() && !newResultLocs.empty()) {
      hwMod.setResultLocsAttr(ArrayAttr::get(context, newResultLocs));
    } else if (oldResultLocs.has_value() && newResultLocs.empty()) {
      hwMod.removeResultLocsAttr();
    }
  }

  // Update module type (works for both HWModuleOp and HWModuleExternOp)
  auto newType = ModuleType::get(context, newPorts);
  module.setHWModuleType(newType);

  // Update port attributes
  if (!newPortAttrs.empty()) {
    assert(newPortAttrs.size() == newPorts.size());
    module.setAllPortAttrs(newPortAttrs);
  } else if (!oldPortAttrs.empty()) {
    module.removeAllPortAttrs();
  }

  // Update port locations (different for HWModuleOp vs HWModuleExternOp)
  if (!newPortLocs.empty()) {
    module.setAllPortLocsAttrs(newPortLocs);
  } else if (auto extMod = dyn_cast<HWModuleExternOp>(module.getOperation())) {
    // For extern modules with no ports left, remove the port_locs attribute
    extMod->removeAttr("port_locs");
  }

  // Update all instances of this module
  auto *moduleNode = instanceGraph.lookup(module.getModuleNameAttr());
  if (moduleNode)
    for (auto *use : moduleNode->uses()) {
      if (auto inst = use->getInstance()) {
        // Erase operands in reverse order to maintain valid indices
        for (auto inputIndex : llvm::reverse(erasedInputIndices))
          inst->eraseOperand(inputIndex);
        // Create new instance with updated operands
        OpBuilder builder(inst);
        auto operands = inst->getOperands();
        auto newInst = rewriteInstance(builder, inst, module, operands);
        instanceGraph.replaceInstance(inst, newInst);
        // Drop results and replace uses with new instance results.
        // Track the new result index separately since newInst has fewer
        // results.
        unsigned newResultIndex = 0;
        for (auto [index, result] : llvm::enumerate(inst->getResults())) {
          if (erasedOutputs.contains(index)) {
            dropResult(inst, index);
            // Drop all uses of the erased result so the instance can be erased
            result.dropAllUses();
          } else {
            result.replaceAllUsesWith(newInst->getResult(newResultIndex));
            newResultIndex++;
          }
        }
        // Now remove the old instance.
        inst->erase();
      }
    }
}

void circt::removePorts(
    HWModuleOp module, igraph::InstanceGraph &instanceGraph,
    const std::function<bool(const hw::PortInfo &)> &shouldRemove,
    const std::function<bool(BlockArgument)> &dropModuleArg,
    const std::function<bool(Operation *, unsigned)> &dropResult) {
  removePortsImpl(module, instanceGraph, shouldRemove, dropModuleArg,
                  dropResult);
}

void circt::removePorts(
    HWModuleExternOp module, igraph::InstanceGraph &instanceGraph,
    const std::function<bool(const hw::PortInfo &)> &shouldRemove,
    const std::function<bool(Operation *, unsigned)> &dropResult) {
  // Delegate to the unified implementation with a null dropModuleArg callback
  // (extern modules have no body/block arguments)
  removePortsImpl(module, instanceGraph, shouldRemove, nullptr, dropResult);
}
