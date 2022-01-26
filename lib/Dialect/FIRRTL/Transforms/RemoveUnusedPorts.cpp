//===- RemoveUnusedPorts.cpp - Remove Dead Ports ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "firrtl-remove-unused-ports"

using namespace circt;
using namespace firrtl;

namespace {
struct RemoveUnusedPortsPass
    : public RemoveUnusedPortsBase<RemoveUnusedPortsPass> {
  void runOnOperation() override;
  void removeUnusedModulePorts(FModuleOp module,
                               InstanceGraphNode *instanceGraphNode);
};
} // namespace

void RemoveUnusedPortsPass::runOnOperation() {
  auto instanceGraph = getAnalysis<InstanceGraph>();
  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                          << "\n");
  CircuitOp circuit = getOperation();
  // Iterate in the reverse order of instance graph iterator, i.e. from leaves
  // to top.
  for (auto *node : llvm::reverse(instanceGraph))
    if (auto module = dyn_cast<FModuleOp>(node->getModule()))
      // Don't prune the main module.
      if (circuit.getMainModule() != module)
        removeUnusedModulePorts(module, node);
}

void RemoveUnusedPortsPass::removeUnusedModulePorts(
    FModuleOp module, InstanceGraphNode *instanceGraphNode) {
  LLVM_DEBUG(llvm::dbgs() << "Prune ports of module: " << module.getName()
                          << "\n");
  // This tracks port indexes that can be erased.
  SmallVector<unsigned> removalPortIndexes;
  // This tracks constant values of output ports. None indicates an invalid
  // value.
  SmallVector<llvm::Optional<APSInt>> outputPortConstants;
  auto ports = module.getPorts();

  for (auto e : llvm::enumerate(ports)) {
    unsigned index = e.index();
    auto port = e.value();
    auto arg = module.getArgument(index);

    // If the port is don't touch or has unprocessed annotations, we cannot
    // remove the port. Maybe we can allow annotations though.
    if (hasDontTouch(arg) || !port.annotations.empty())
      continue;

    // TODO: Handle inout ports.
    if (port.isInOut())
      continue;

    // If the port is input and has an user, we cannot remove the
    // port.
    if (port.isInput() && !arg.use_empty())
      continue;

    // If the port is output, then check that the port is only connected to
    // invalid or constant.
    if (port.isOutput()) {
      if (arg.use_empty()) {
        outputPortConstants.push_back(None);
      } else if (arg.hasOneUse()) {
        auto connect = dyn_cast<ConnectOp>(*arg.user_begin());
        if (!connect || !isa_and_nonnull<InvalidValueOp, ConstantOp>(
                            connect.src().getDefiningOp()))
          continue;

        Operation *srcOp;
        if (auto constant =
                dyn_cast<ConstantOp>(connect.src().getDefiningOp())) {
          outputPortConstants.push_back(constant.value());
          srcOp = constant;
        } else {
          assert(isa<InvalidValueOp>(connect.src().getDefiningOp()) &&
                 "only expect invalid");
          srcOp = connect.src().getDefiningOp();
          outputPortConstants.push_back(None);
        }
        // Erase connect op because we are going to remove this output ports.
        connect.erase();
        if (srcOp->use_empty())
          srcOp->erase();

      } else {
        continue;
      }
    }

    removalPortIndexes.push_back(index);
  }

  // If there is nothing to remove, abort.
  if (removalPortIndexes.empty())
    return;

  module.erasePorts(removalPortIndexes);

  LLVM_DEBUG(llvm::for_each(removalPortIndexes, [&](unsigned index) {
               llvm::dbgs() << "Delete port: " << ports[index].name << "\n";
             }););

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = use->getInstance();
    OpBuilder builder(instance);
    unsigned outputPortIndex = 0;
    for (auto index : removalPortIndexes) {
      auto result = instance.getResult(index);
      assert(!ports[index].isInOut() && "don't expect inout ports");

      // If the port is input, replace the port with an unwritten wire
      // so that we can remove use-chains in SV dialect canonicalization.
      if (ports[index].isInput()) {
        WireOp wire =
            builder.create<WireOp>(instance.getLoc(), result.getType());
        result.replaceUsesWithIf(wire, [&](OpOperand &op) -> bool {
          // Connects can be deleted directly.
          if (isa<ConnectOp>(op.getOwner())) {
            op.getOwner()->erase();
            return false;
          }
          return true;
        });

        // If the wire doesn't have an user, just erase it.
        if (wire.use_empty())
          wire.erase();
        continue;
      }
      auto portConstant = outputPortConstants[outputPortIndex++];
      // Output case. Replace with the output port with an invalid or constant
      // value.
      Value value;
      if (portConstant)
        value = builder.create<ConstantOp>(instance.getLoc(), *portConstant);
      else
        value =
            builder.create<InvalidValueOp>(instance.getLoc(), result.getType());

      result.replaceAllUsesWith(value);
    }

    // Create a new instance op without unused ports.
    instance.erasePorts(builder, removalPortIndexes);
    // Remove old one.
    instance.erase();
  }

  numRemovedPorts += removalPortIndexes.size();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createRemoveUnusedPortsPass() {
  return std::make_unique<RemoveUnusedPortsPass>();
}
