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
#include "circt/Dialect/FIRRTL/FIRRTLAnnotations.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/InstanceGraph.h"
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

/// This gets the value targeted by a field id.  If the field id is targeting
/// the value itself, it returns it unchanged. If it is targeting a single field
/// in a aggregate value, such as a bundle or vector, this will create the
/// necessary subaccesses to get the value.
static Value getEnable(ImplicitLocOpBuilder builder, Value value,
                       unsigned fieldID) {
  // When the fieldID hits 0, we've found the target value.
  while (fieldID != 0) {
    auto type = value.getType();
    if (auto bundle = type.dyn_cast<BundleType>()) {
      auto index = bundle.getIndexForFieldID(fieldID);
      value = builder.create<SubfieldOp>(value, index);
      fieldID -= bundle.getFieldID(index);
    } else {
      auto vector = type.cast<FVectorType>();
      auto index = vector.getIndexForFieldID(fieldID);
      value = builder.create<SubindexOp>(value, index);
      fieldID -= vector.getFieldID(index);
    }
  }
  return value;
}

static const char dutClass[] = "sifive.enterprise.firrtl.MarkDUTAnnotation";
static const char dftEnableClass[] =
    "sifive.enterprise.firrtl.DFTTestModeEnableAnnotation";

namespace {
class WireDFTPass : public WireDFTBase<WireDFTPass> {
  void runOnOperation() override;
};
} // namespace

void WireDFTPass::runOnOperation() {
  auto circuit = getOperation();

  // This is the module marked as the device under test.
  Operation *dut = nullptr;

  // This is the signal marked as the DFT enable, a 1-bit signal to be wired to
  // the EICG modules.
  Value enableSignal;
  FModuleOp enableModule;

  // Walk all modules looking for the DUT module and the annotated enable
  // signal.
  for (auto &op : *circuit.getBody()) {
    auto module = dyn_cast<FModuleOp>(op);

    // If this isn't a regular module, continue.
    if (!module)
      continue;

    // Check if this module is the DUT.
    AnnotationSet annos(module);
    if (annos.hasAnnotation(dutClass)) {
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
      if (error || !anno.isClass(dftEnableClass))
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
      enableSignal = getEnable(ImplicitLocOpBuilder::atBlockBegin(
                                   module->getLoc(), module.getBody()),
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
        if (error || !anno.isClass(dftEnableClass))
          return false;
        if (enableSignal) {
          auto diag =
              op->emitError("more than one thing marked as a DFT enable");
          diag.attachNote(enableSignal.getLoc()) << "first thing defined here";
          error = true;
          return false;
        }
        // Grab the enable value and remove the annotation.
        enableSignal = getEnable(
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
        FModuleLike module = cast<FModuleLike>(node->getTarget()->getModule());
        // If this is a clock gate, record the module and return true.
        if (module.moduleName().startswith("EICG_wrapper")) {
          clockGates.insert(node);
          return true;
        }
        // Return true if this is the module with the enable signal.
        return node->getParent()->getModule() == enableModule;
      });
  auto lcaModule = cast<FModuleOp>(lca->getModule());

  // If there are no clock gates under the DUT, we can stop now.
  if (!clockGates.size())
    return;

  // Stash some useful things.
  auto *context = &getContext();
  auto uint1Type = enableSignal.getType().cast<FIRRTLType>();
  auto loc = lcaModule.getLoc();
  auto portName = StringAttr::get(context, "test_en");

  // This maps an enable signal to each module.
  DenseMap<InstanceGraphNode *, Value> signals;

  // Helper to insert a port into an instance op. We have to replace the whole
  // op and then keep the instance graph updated.
  auto insertPortIntoInstance =
      [&](InstanceRecord *instanceNode,
          std::pair<unsigned, PortInfo> port) -> InstanceOp {
    auto instance = instanceNode->getInstance();
    auto clone = instance.cloneAndInsertPorts({port});
    instanceGraph.replaceInstance(instance, clone);
    instance->replaceAllUsesWith(clone.getResults().drop_back());
    instance->erase();
    return clone;
  };

  // At this point we have found the the enable signal, all important clock
  // gates, and the ancestor of these. From here we need wire the enable signal
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
    auto module = cast<FModuleOp>(node->getModule());
    unsigned portNo = module.getNumPorts();
    module.insertPorts({{portNo, portInfo}});
    auto builder =
        ImplicitLocOpBuilder::atBlockEnd(module.getLoc(), module.getBody());
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
    auto module = cast<FModuleOp>(node->getModule());
    unsigned portNo = module.getNumPorts();
    module.insertPorts({{portNo, portInfo}});

    // Attach the input signal to each instance of this module.
    for (auto *instanceNode : node->uses()) {
      // Add an input signal to this instance op.
      auto clone = insertPortIntoInstance(instanceNode, {portNo, portInfo});

      // Wire the parent signal to the instance op.
      auto *parent = instanceNode->getParent();
      auto module = cast<FModuleOp>(parent->getModule());
      auto signal = getSignal(parent);
      auto builder =
          ImplicitLocOpBuilder::atBlockEnd(module->getLoc(), module.getBody());
      builder.create<ConnectOp>(clone.getResult(portNo), signal);
    }

    // Record and return the new signal.
    signal = module.getArgument(portNo);
    return signal;
  };

  // Wire the signal to each clock gate using the helper above.
  for (auto *instance : clockGates) {
    auto *parent = instance->getParent();
    auto module = cast<FModuleOp>(parent->getModule());
    auto builder =
        ImplicitLocOpBuilder::atBlockEnd(module->getLoc(), module.getBody());
    // Hard coded port result number; the clock gate test_en port is 1
    auto testEnPortNo = 1;
    builder.create<ConnectOp>(instance->getInstance().getResult(testEnPortNo),
                              getSignal(parent));
  }

  // And we're done!
  markAnalysesPreserved<InstanceGraph>();
}

std::unique_ptr<mlir::Pass> circt::firrtl::createWireDFTPass() {
  return std::make_unique<WireDFTPass>();
}
