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
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/ADT/PostOrderIterator.h"
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

  /// If true, the pass will remove unused ports even if they have carry a
  /// symbol or annotations. This is likely to break the IR, but may be useful
  /// for `circt-reduce` where preserving functional correctness of the IR is
  /// not important.
  bool ignoreDontTouch = false;
};
} // namespace

void RemoveUnusedPortsPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<InstanceGraph>();
  LLVM_DEBUG(llvm::dbgs() << "===----- Remove unused ports -----==="
                          << "\n");
  // Iterate in the reverse order of instance graph iterator, i.e. from leaves
  // to top.
  for (auto *node : llvm::post_order(&instanceGraph))
    if (auto module = dyn_cast<FModuleOp>(*node->getModule()))
      // Don't prune the main module.
      if (!module.isPublic())
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
    if ((hasDontTouch(arg) || !port.annotations.empty()) && !ignoreDontTouch)
      continue;

    // TODO: Handle inout ports.
    if (port.isInOut())
      continue;

    // If the port is input and has an user, we cannot remove the
    // port.
    if (port.isInput() && !arg.use_empty())
      continue;

    // If the port is an input and has an inner symbol, we cannot remove it.
    // For output ports we can create a local wire to capture any locally driven
    // value, but not so for inputs.
    bool hasSymbol = port.sym && !port.sym.getValue().empty();
    if (port.isInput() && hasSymbol)
      continue;

    auto portIsUnused = [&](InstanceRecord *a) -> bool {
      auto port = a->getInstance()->getResult(arg.getArgNumber());
      return port.getUses().empty();
    };

    // Output port.
    if (port.isOutput()) {
      Optional<APSInt> constantValue; // constant value of the port, if any
      FConnectLike constantConnect;   // set if the sole use is a const connect

      if (arg.hasOneUse()) {
        // Check if the port has a single connect driving it to a constant or
        // invalid value.
        Operation *op = arg.use_begin().getUser();
        constantConnect = dyn_cast<FConnectLike>(op);
        if (!constantConnect)
          continue;
        auto *srcOp = constantConnect.src().getDefiningOp();
        if (!srcOp)
          continue;
        if (auto constant = dyn_cast<ConstantOp>(srcOp)) {
          constantValue = constant.value();
        } else if (isa<InvalidValueOp>(srcOp)) {
          constantValue = None;
        } else {
          continue;
        }
      } else if (!arg.use_empty() &&
                 !llvm::all_of(instanceGraphNode->uses(), portIsUnused)) {
        // If the port is internally driven and has external uses we cannot
        // remove it.
        continue;
      }
      LLVM_DEBUG({
        llvm::dbgs() << "- Removing " << port.name << "\n";
        if (constantConnect)
          llvm::dbgs() << "  - Constant " << constantValue << "\n";
      });

      // At this point we have decided to remove the port. First of all, replace
      // the port with an internal wire in case the port has internal uses or a
      // symbol.
      //
      // NOTE: There are usecases where the output port has no internal connects
      // anymore as a result of other canonicalizations running before port
      // removal. In that case the local wire is only needed if the port carries
      // a symbol which someone might refer to.
      WireOp wire;
      auto builder =
          ImplicitLocOpBuilder::atBlockBegin(arg.getLoc(), module.getBody());
      if ((!arg.use_empty() && !constantConnect) || hasSymbol) {
        LLVM_DEBUG(llvm::dbgs() << "  - Creating replacement wire\n");
        wire = builder.create<WireOp>(arg.getType());
        if (hasSymbol)
          wire.inner_symAttr(port.sym);
        arg.replaceAllUsesWith(wire);
      }

      // At this point we might have a wire that has replaced the port in all
      // operations. In case there were no internal uses, drive an invalid value
      // onto the wire.
      if (wire && wire.use_empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  - Invalidating due to lack of internal uses\n");
        auto invalidValue = builder.create<InvalidValueOp>(wire.getType());
        builder.create<StrictConnectOp>(wire, invalidValue);
      }

      // In case the port was connected to a constant or invalid value, and we
      // had no need for a replacement wire, delete the constant connect.
      if (constantConnect && constantConnect.dest() != wire) {
        LLVM_DEBUG(llvm::dbgs()
                   << "  - Cleaning up " << constantConnect << "\n");
        auto *srcOp = constantConnect.src().getDefiningOp();
        constantConnect->erase();
        constantConnect = {};
        if (srcOp && srcOp->use_empty()) {
          LLVM_DEBUG(llvm::dbgs() << "  - Cleaning up " << *srcOp << "\n");
          srcOp->erase();
        }
      }

      outputPortConstants.push_back(constantValue);
    }

    removalPortIndexes.push_back(index);
  }

  // If there is nothing to remove, abort.
  if (removalPortIndexes.empty())
    return;

  // Delete ports from the module.
  module.erasePorts(removalPortIndexes);
  LLVM_DEBUG(llvm::for_each(removalPortIndexes, [&](unsigned index) {
               llvm::dbgs() << "Delete port: " << ports[index].name << "\n";
             }););

  // Rewrite all uses.
  for (auto *use : instanceGraphNode->uses()) {
    auto instance = ::cast<InstanceOp>(*use->getInstance());
    ImplicitLocOpBuilder builder(instance.getLoc(), instance);
    unsigned outputPortIndex = 0;
    for (auto index : removalPortIndexes) {
      auto result = instance.getResult(index);
      assert(!ports[index].isInOut() && "don't expect inout ports");

      // If the port is input, replace the port with an unwritten wire
      // so that we can remove use-chains in SV dialect canonicalization.
      if (ports[index].isInput()) {
        WireOp wire = builder.create<WireOp>(result.getType());

        // Check that the input port is only written. Sometimes input ports are
        // used as temporary wires. In that case, we cannot erase connections.
        bool onlyWritten = llvm::all_of(result.getUsers(), [&](Operation *op) {
          if (auto connect = dyn_cast<FConnectLike>(op))
            return connect.dest() == result;
          return false;
        });

        result.replaceUsesWithIf(wire, [&](OpOperand &op) -> bool {
          // Connects can be deleted directly.
          if (onlyWritten && isa<FConnectLike>(op.getOwner())) {
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

      // Output port. Replace with the output port with an invalid or constant
      // value.
      auto portConstant = outputPortConstants[outputPortIndex++];
      Value value;
      if (portConstant)
        value = builder.create<ConstantOp>(*portConstant);
      else
        value = builder.create<InvalidValueOp>(result.getType());

      result.replaceAllUsesWith(value);
    }

    // Create a new instance op without unused ports.
    instance.erasePorts(builder, removalPortIndexes);
    // Remove old one.
    instance.erase();
  }

  numRemovedPorts += removalPortIndexes.size();
}

std::unique_ptr<mlir::Pass>
circt::firrtl::createRemoveUnusedPortsPass(bool ignoreDontTouch) {
  auto pass = std::make_unique<RemoveUnusedPortsPass>();
  pass->ignoreDontTouch = ignoreDontTouch;
  return pass;
}
