//===- WireDFT.cpp - Create DFT module ports --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the WireDFT pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/AnnotationDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLInstanceGraph.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/SetVector.h"

using namespace circt;
using namespace firrtl;

/// This calculates the lowest common ancestor of the instance operations. It
/// supports finding the LCA of an arbitrary number of nodes (instead of the
/// usual 2 at a time), and incremental target node discovery.
static InstanceGraphNode *
lowestCommonAncestor(InstanceGraphNode *top,
                     llvm::function_ref<bool(InstanceRecord *)> predicate) {
  struct StackElement {
    StackElement(InstanceGraphNode *node)
        : node(node), it(node->begin()), count(0) {}
    // The node itself.
    InstanceGraphNode *node;
    // The current iterator of this node's children.
    InstanceGraphNode::iterator it;
    // The number of interesting nodes under this one.
    unsigned count;
  };

  // Since we support incremental discovery of the interesting modules, we keep
  // track of node which has the most children under it so far.
  InstanceGraphNode *currentLCA = nullptr;
  unsigned currentCount = 0;

  // This is used to pass the count of a child back to its parent.
  unsigned childCount = 0;

  // The workstack for a depth-first walk.
  SmallVector<StackElement> stack;
  stack.emplace_back(top);
  while (!stack.empty()) {
    auto &element = stack.back();
    auto &node = element.node;
    auto &it = element.it;

    // Add the count of the just-processed child node.  If we are coming from
    // the parent node, childCount will be 0.
    element.count += childCount;

    // Check if we're done processing this nodes children.
    if (it == node->end()) {
      // Store the current count in the childCount, so that we may return the
      // count to this node's parent op.
      childCount = element.count;

      // If this node has more children than any other node, it is the best LCA
      // of all the nodes we have found *so far*.
      if (childCount > currentCount) {
        currentLCA = node;
        currentCount = element.count;
      }

      // Pop back to the parent op.
      stack.pop_back();
      continue;
    }

    // If the current node is interesting, increase this node's count.
    auto *instanceNode = *it++;
    if (predicate(instanceNode))
      ++element.count;

    // Set up to iterate the child node.
    stack.emplace_back(instanceNode->getTarget());
    childCount = 0;
  }
  return currentLCA;
}

namespace {
class WireDFTPass : public WireDFTBase<WireDFTPass> {
  void runOnOperation() override;
};
} // namespace

void WireDFTPass::runOnOperation() {
  auto circuit = getOperation();

  // This is the module marked as the device under test.
  FModuleOp dut = nullptr;

  // This is the signal marked as the DFT enable, a 1-bit signal to be wired to
  // the EICG modules.
  Value enableSignal;
  FModuleOp enableModule;

  // Walk all modules looking for the DUT module and the annotated enable
  // signal.
  for (auto &op : *circuit.getBodyBlock()) {
    auto module = dyn_cast<FModuleOp>(op);

    // If this isn't a regular module, continue.
    if (!module)
      continue;

    // Check if this module is the DUT.
    AnnotationSet annos(module);
    if (annos.hasAnnotation(dutAnnoClass)) {
      // Check if we already found the DUT.
      if (dut) {
        auto diag = module->emitError("more than one module marked DUT");
        diag.attachNote(dut->getLoc()) << "first module here";
        signalPassFailure();
        return;
      }
      dut = module;
    }

    // See if this module has any port marked as the DFT enable.
    bool error = false;
    AnnotationSet::removePortAnnotations(module, [&](unsigned i,
                                                     Annotation anno) {
      // If we have already encountered an error or this is not the right
      // annotation kind, continue.
      if (error || !anno.isClass(dftTestModeEnableAnnoClass))
        return false;
      // If we have already found a DFT enable, emit an error.
      if (enableSignal) {
        auto diag =
            module->emitError("more than one thing marked as a DFT enable");
        diag.attachNote(enableSignal.getLoc()) << "first thing defined here";
        error = true;
        return false;
      }
      // Grab the enable value and remove the annotation.
      enableSignal =
          getValueByFieldID(ImplicitLocOpBuilder::atBlockBegin(
                                module->getLoc(), module.getBodyBlock()),
                            module.getArgument(i), anno.getFieldID());
      enableModule = module;
      return true;
    });
    if (error)
      return signalPassFailure();

    // Walk the module body looking for any operation marked as the DFT enable.
    auto walkResult = module->walk([&](Operation *op) {
      AnnotationSet::removeAnnotations(op, [&](Annotation anno) {
        // If we have already encountered an error or this is not the right
        // annotation kind, continue.
        if (error || !anno.isClass(dftTestModeEnableAnnoClass))
          return false;
        if (enableSignal) {
          auto diag =
              op->emitError("more than one thing marked as a DFT enable");
          diag.attachNote(enableSignal.getLoc()) << "first thing defined here";
          error = true;
          return false;
        }
        // Grab the enable value and remove the annotation.
        enableSignal = getValueByFieldID(
            ImplicitLocOpBuilder::atBlockEnd(op->getLoc(), op->getBlock()),
            op->getResult(0), anno.getFieldID());
        enableModule = module;
        return true;
      });
      if (error)
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted())
      return signalPassFailure();
  }

  // No enable signal means we have no work to do.
  if (!enableSignal)
    return markAllAnalysesPreserved();

  // This pass requires a DUT.
  if (!dut) {
    circuit->emitError("no DUT module found.");
    return signalPassFailure();
  }

  auto &instanceGraph = getAnalysis<InstanceGraph>();

  // Find the LowestCommonAncestor node of all the nodes to be wired together,
  // and collect all the ClockGate modules.
  llvm::SetVector<InstanceRecord *> clockGates;
  auto *lca = lowestCommonAncestor(
      instanceGraph.lookup(dut), [&](InstanceRecord *node) {
        auto module = node->getTarget()->getModule();
        // If this is a clock gate, record the module and return true.
        if (module.moduleName().startswith("EICG_wrapper")) {
          clockGates.insert(node);
          return true;
        }
        // Return true if this is the module with the enable signal.
        return node->getParent()->getModule() == enableModule;
      });

  // If there are no clock gates under the DUT, we can stop now.
  if (clockGates.empty())
    return;

  // Stash UInt<1> type for use throughout.
  auto uint1Type = enableSignal.getType().cast<FIRRTLType>();

  // Hard coded port result number; the clock gate test_en port is 1.
  // Language below reflects this as well.
  const unsigned testEnPortNo = 1;

  // Scan gathered clock gates and check for basic compatibility.
  for (auto *cgnode : clockGates) {
    auto module = cgnode->getTarget()->getModule();
    auto genErr = [&]() { return module.emitError("clock gate module "); };
    FExtModuleOp ext = dyn_cast<FExtModuleOp>(module.getOperation());
    if (!ext) {
      genErr() << "must be an extmodule";
      return signalPassFailure();
    }
    static_assert(testEnPortNo == 1, "update this code");
    if (ext.getNumPorts() <= testEnPortNo) {
      genErr() << "must have at least two ports";
      return signalPassFailure();
    }
    if (ext.getPortType(testEnPortNo) != uint1Type) {
      genErr()
          .append("must have second port with type UInt<1>")
          .attachNote() // Use port location once available.
          .append("Second port (\"")
          .append(ext.getPortName(testEnPortNo))
          .append("\") has type ")
          .append(ext.getPortType(testEnPortNo))
          .append(", expected ")
          .append(uint1Type);
      return signalPassFailure();
    }
    if (ext.getPortDirection(testEnPortNo) != Direction::In) {
      genErr() << "must have second port with input direction";
      return signalPassFailure();
    }
  }

  // Handle enable signal (only) outside DUT.
  if (!instanceGraph.isAncestor(enableModule, lca->getModule())) {
    // Current LCA covers the clock gates we care about.
    // Compute new LCA from enable to that node.
    lca = lowestCommonAncestor(
        instanceGraph.getTopLevelNode(), [&](InstanceRecord *node) {
          return node->getTarget() == lca ||
                 node->getParent()->getModule() == enableModule;
        });
    // Handle unreachable case.
    if (!lca) {
      auto diag =
          circuit.emitError("unable to connect enable signal and DUT, may not "
                            "be reachable from top-level module");
      diag.attachNote(enableSignal.getLoc()) << "enable signal here";
      diag.attachNote(dut.getLoc()) << "DUT here";
      diag.attachNote(instanceGraph.getTopLevelModule().getLoc())
          << "top-level module here";

      return signalPassFailure();
    }
  }

  // Check all gates we're wiring are only within the DUT.
  if (!allUnder(clockGates.getArrayRef(), instanceGraph.lookup(dut))) {
    dut->emitError()
        << "clock gates within DUT must not be instantiated outside the DUT";
    return signalPassFailure();
  }

  // Stash some useful things.
  auto loc = lca->getModule().getLoc();
  auto portName = StringAttr::get(&getContext(), "test_en");

  // This maps an enable signal to each module.
  DenseMap<InstanceGraphNode *, Value> signals;

  // Helper to insert a port into an instance op. We have to replace the whole
  // op and then keep the instance graph updated.
  auto insertPortIntoInstance =
      [&](InstanceRecord *instanceNode,
          std::pair<unsigned, PortInfo> port) -> InstanceOp {
    auto instance = cast<InstanceOp>(*instanceNode->getInstance());
    auto clone = instance.cloneAndInsertPorts({port});
    instanceGraph.replaceInstance(instance, clone);
    instance->replaceAllUsesWith(clone.getResults().drop_back());
    instance->erase();
    return clone;
  };

  // At this point we have found the enable signal, all important clock gates,
  // and the ancestor of these. From here we need wire the enable signal
  // upward to the LCA, and then wire the enable signal down to all clock
  // gates.

  // This first part wires the enable signal up ward to the LCA module.
  auto *node = instanceGraph.lookup(enableModule);
  auto signal = enableSignal;
  PortInfo portInfo = {portName, uint1Type, Direction::Out, {}, loc};
  while (node != lca) {
    // If there is more than one parent the we are in trouble. We can't handle
    // more than one enable signal to wire everywhere else.
    if (!node->hasOneUse()) {
      auto diag = emitError(enableSignal.getLoc(),
                            "mutliple instantiations of the DFT enable signal");
      auto it = node->usesBegin();
      diag.attachNote((*it++)->getInstance()->getLoc())
          << "first instance here";
      diag.attachNote((*it)->getInstance()->getLoc()) << "second instance here";
      return signalPassFailure();
    }

    // Record the signal for this module.
    signals[node] = signal;

    // Create an output port to this module.
    auto module = cast<FModuleOp>(*node->getModule());
    unsigned portNo = module.getNumPorts();
    module.insertPorts({{portNo, portInfo}});
    auto builder = ImplicitLocOpBuilder::atBlockEnd(module.getLoc(),
                                                    module.getBodyBlock());
    builder.create<ConnectOp>(module.getArgument(portNo), signal);

    // Add an output port to the instance of this module.
    auto *instanceNode = (*node->usesBegin());
    auto clone = insertPortIntoInstance(instanceNode, {portNo, portInfo});

    // Set up for the next iteration.
    signal = clone.getResult(portNo);
    node = instanceNode->getParent();
  }

  // Record the enable signal in the LCA.
  signals[node] = signal;

  // Drill the enable signal to each of the leaf clock gates. We do this
  // searching upward in the hiearchy until we find a module with the signal.
  // This is a recursive function due to lazyness.
  portInfo = {portName, uint1Type, Direction::In, {}, loc};
  std::function<Value(InstanceGraphNode *)> getSignal =
      [&](InstanceGraphNode *node) -> Value {
    // Mutable signal reference.
    auto &signal = signals[node];

    // Early break if this module has already been wired.
    if (signal)
      return signal;

    // Add an input signal to this module.
    auto module = cast<FModuleOp>(*node->getModule());
    unsigned portNo = module.getNumPorts();
    module.insertPorts({{portNo, portInfo}});
    auto arg = module.getArgument(portNo);

    // Record the new signal.
    signal = arg;

    // Attach the input signal to each instance of this module.
    for (auto *instanceNode : node->uses()) {
      // Add an input signal to this instance op.
      auto clone = insertPortIntoInstance(instanceNode, {portNo, portInfo});

      // Wire the parent signal to the instance op.
      auto *parent = instanceNode->getParent();
      auto module = cast<FModuleOp>(*parent->getModule());
      auto signal = getSignal(parent);
      auto builder = ImplicitLocOpBuilder::atBlockEnd(module->getLoc(),
                                                      module.getBodyBlock());
      builder.create<ConnectOp>(clone.getResult(portNo), signal);
    }

    return arg;
  };

  // Wire the signal to each clock gate using the helper above.
  for (auto *instance : clockGates) {
    auto *parent = instance->getParent();
    auto module = cast<FModuleOp>(*parent->getModule());
    auto builder = ImplicitLocOpBuilder::atBlockEnd(module->getLoc(),
                                                    module.getBodyBlock());
    builder.create<ConnectOp>(
        cast<InstanceOp>(*instance->getInstance()).getResult(testEnPortNo),
        getSignal(parent));
  }

  // And we're done!
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createWireDFTPass() {
  return std::make_unique<WireDFTPass>();
}
