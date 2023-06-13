//===- HWRaiseInOutPorts.cpp - Generator Callout Pass ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Call arbitrary programs and pass them the attributes attached to external
// modules.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace sv;
using namespace hw;

//===----------------------------------------------------------------------===//
// HWRaiseInOutPortsPass
//===----------------------------------------------------------------------===//

namespace {

struct HWRaiseInOutPortsPass
    : public sv::HWRaiseInOutPortsBase<HWRaiseInOutPortsPass> {
  void runOnOperation() override;

private:
  LogicalResult raise(InstanceGraphNode *topModule);

  LogicalResult convertPort(InstanceGraphNode *module, PortInfo port);
};
} // end anonymous namespace

LogicalResult HWRaiseInOutPortsPass::convertPort(InstanceGraphNode *node,
                                                 PortInfo inoutPort) {

  HWModuleOp mod = cast<HWModuleOp>(node->getModule());
  BlockArgument inoutArg = mod.getArgument(inoutPort.argNum);
  InOutType inoutType = inoutArg.getType().dyn_cast<InOutType>();
  assert(inoutType && "expected inout type");
  Type elementType = inoutType.getElementType();
  auto users = inoutArg.getUsers();

  // Gather readers and writers (how to handle sv.passign?)
  llvm::SmallVector<sv::ReadInOutOp, 4> readers;
  llvm::SmallVector<sv::AssignOp, 4> writers;

  for (auto user : users) {
    if (auto read = dyn_cast<sv::ReadInOutOp>(user))
      readers.push_back(read);
    else if (auto write = dyn_cast<sv::AssignOp>(user))
      writers.push_back(write);
  }

  bool hasReaders = !readers.empty();
  bool hasWriter = !writers.empty();

  if (!hasReaders && !hasWriter)
    return success();

  if (writers.size() > 1)
    return emitError(inoutArg.getLoc()) << "multiple writers to inout port";

  // Input port rewriting
  if (hasReaders) {
    auto newInput = mod.insertInput(inoutPort.argNum + 1,
                                    inoutPort.getName() + "_in", elementType);
    // Replace all readers with the new input.
    for (auto reader : readers) {
      reader.replaceAllUsesWith(newInput.second);
      reader.erase();
    }
  }

  // Output port rewriting.
  if (hasWriter) {
    auto writer = writers.front();
    mod.appendOutput(inoutPort.getName() + "_out", writer.getSrc());
    writer.erase();
  }

  // Erase the inout port.
  modifyModulePorts(mod, {}, {}, {static_cast<unsigned>(inoutPort.argNum)}, {},
                    mod.getBodyBlock());

  // TODO: all of the 3 above port modifications can be done in one go.

  // Instantiation rewriting.
  OpBuilder b(mod.getContext());

  for (auto user : node->uses()) {
    // Skip anything that isn't plain old instance ops.
    auto inst = dyn_cast_or_null<InstanceOp>(user->getInstance());
    if (!inst)
      continue;

    llvm::SmallVector<Value> newArgs;
    newArgs = inst.getOperands();
    b.setInsertionPoint(inst);

    Value inourArgToInstance = newArgs[inoutPort.argNum];

    if (hasReaders) {
      // Create a read_inout op at the instantiation point. This effectively
      // pushes the read_inout op from the module to the instantiation.
      newArgs[inoutPort.argNum] =
          b.create<ReadInOutOp>(inst.getLoc(), inourArgToInstance).getResult();
    } else
      newArgs.erase(newArgs.begin() + inoutPort.argNum);

    // Replace the instance
    auto newInst = b.create<InstanceOp>(inst.getLoc(), mod,
                                        inst.getInstanceName(), newArgs);
    inst.replaceAllUsesWith(newInst.getResults().drop_back(hasWriter ? 1 : 0));
    if (hasWriter) {
      // Create a sv.assign at the instantiation point. This effectively
      // pushes the assign op from the module to the instantiation.
      // This will always be the last result of the instance.
      b.create<AssignOp>(inst.getLoc(), inourArgToInstance,
                         newInst.getResult(newInst.getNumResults() - 1));
    }
    inst.erase();
  }

  return success();
}

LogicalResult HWRaiseInOutPortsPass::raise(InstanceGraphNode *instanceNode) {
  hw::HWModuleLike moduleLike = instanceNode->getModule();

  // Will only touch HWModuleOp
  hw::HWModuleOp moduleOp = dyn_cast<hw::HWModuleOp>(moduleLike.getOperation());
  if (!moduleOp)
    return success();

  // Gather inout ports of this module.
  llvm::MapVector<size_t, PortInfo> inoutPorts;
  for (size_t portIdx = 0; portIdx < moduleOp.getNumInOrInoutPorts();
       ++portIdx) {
    auto portInfo = moduleOp.getInOrInoutPort(portIdx);
    if (portInfo.direction != hw::PortDirection::INOUT)
      continue;

    inoutPorts[portIdx] = portInfo;
  }

  if (inoutPorts.empty())
    return success();

  // Convert each port
  for (auto inoutPort : inoutPorts)
    if (failed(convertPort(instanceNode, inoutPort.second)))
      return failure();

  return success();
}

void HWRaiseInOutPortsPass::runOnOperation() {
  circt::hw::InstanceGraph &analysis = getAnalysis<circt::hw::InstanceGraph>();
  auto res = analysis.getInferredTopLevelNodes();

  if (failed(res)) {
    signalPassFailure();
    return;
  }

  // Maintain the set of visited modules; there may be multiple top modules
  // which share subcircuits.
  llvm::DenseSet<InstanceGraphNode *> visited;
  for (InstanceGraphNode *topModule : res.value()) {
    // Visit the instance hierarchy in a depth-first manner, modifying child
    // modules and their ports before their parents.
    for (InstanceGraphNode *node : llvm::post_order(topModule)) {
      if (visited.count(node))
        continue;
      if (failed(raise(node)))
        return signalPassFailure();
      visited.insert(node);
    }
  }
}

std::unique_ptr<Pass> circt::sv::createHWRaiseInOutPortsPass() {
  return std::make_unique<HWRaiseInOutPortsPass>();
}
