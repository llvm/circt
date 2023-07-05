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
#include "circt/Dialect/HW/PortConverter.h"
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
};
} // end anonymous namespace

namespace {

class HWInOutSignalStandard : public SignalingStandard {
public:
  HWInOutSignalStandard(PortConverterImpl &converter, hw::PortInfo port);

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

private:
  void buildInputSignals() override;
  void buildOutputSignals() override;

  // Readers of this port internal in the module.
  llvm::SmallVector<sv::ReadInOutOp, 4> readers;
  // Writers of this port internal in the module.
  llvm::SmallVector<sv::AssignOp, 4> writers;

  bool hasReaders() { return !readers.empty(); }
  bool hasWriters() { return !writers.empty(); }

  // Handles to port info of the newly created ports.
  PortInfo readPort, writePort;
};

HWInOutSignalStandard::HWInOutSignalStandard(PortConverterImpl &converter,
                                             hw::PortInfo port)
    : SignalingStandard(converter, port) {
  // Gather readers and writers (how to handle sv.passign?)
  for (auto *user : body->getArgument(port.argNum).getUsers()) {
    if (auto read = dyn_cast<sv::ReadInOutOp>(user))
      readers.push_back(read);
    else if (auto write = dyn_cast<sv::AssignOp>(user))
      writers.push_back(write);
    else
      user->emitWarning() << "Use of inout port " << port.name
                          << " is not supported";
  }

  if (writers.size() > 1)
    converter.getModule()->emitWarning()
        << "Multiple writers of inout port " << port.name
        << " detected. Will only create an output for the first write";
}

void HWInOutSignalStandard::buildInputSignals() {
  if (hasReaders()) {
    Value readValue =
        converter.createNewInput(origPort, "_rd", origPort.type, readPort);

    // Replace all sv::ReadInOutOp's with the new input.
    Value origInput = body->getArgument(origPort.argNum);
    for (auto *user : llvm::make_early_inc_range(origInput.getUsers())) {
      sv::ReadInOutOp read = dyn_cast<sv::ReadInOutOp>(user);
      if (!read)
        continue;

      read.replaceAllUsesWith(readValue);
      read.erase();
    }
  }

  if (hasWriters()) {
    sv::AssignOp write = writers.front();
    converter.createNewOutput(origPort, "_wr", origPort.type, write.getSrc(),
                              writePort);
    write.erase();
  }
}

void HWInOutSignalStandard::buildOutputSignals() {
  // TODO: could support hw.inout outputs (always create read/write ports) -
  // don't need it for now, though.
  assert(false && "hw.inout outputs not yet supported");
}

void HWInOutSignalStandard::mapInputSignals(OpBuilder &b, Operation *inst,
                                            Value instValue,
                                            SmallVectorImpl<Value> &newOperands,
                                            ArrayRef<Backedge> newResults) {

  if (hasReaders()) {
    // Create a read_inout op at the instantiation point. This effectively
    // pushes the read_inout op from the module to the instantiation site.
    newOperands[readPort.argNum] =
        b.create<ReadInOutOp>(inst->getLoc(), instValue).getResult();
  }

  if (hasWriters()) {
    // Create a sv::AssignOp at the instantiation point. This effectively
    // pushes the write op from the module to the instantiation site.
    Value writeFromInsideMod = newResults[writePort.argNum];
    b.create<sv::AssignOp>(inst->getLoc(), instValue, writeFromInsideMod);
  }
}

void HWInOutSignalStandard::mapOutputSignals(
    OpBuilder &b, Operation *inst, Value instValue,
    SmallVectorImpl<Value> &newOperands, ArrayRef<Backedge> newResults) {
  llvm_unreachable("hw.inout outputs not yet supported");
}

class HWInoutSignalStandardBuilder : public SignalStandardBuilder {
public:
  using SignalStandardBuilder::SignalStandardBuilder;
  FailureOr<std::unique_ptr<SignalingStandard>>
  build(hw::PortInfo port) override {
    if (port.direction == hw::PortDirection::INOUT)
      return {std::make_unique<HWInOutSignalStandard>(converter, port)};
    return SignalStandardBuilder::build(port);
  }
};

} // namespace

void HWRaiseInOutPortsPass::runOnOperation() {
  // Find all modules and run port conversion on them.
  circt::hw::InstanceGraph &instanceGraph =
      getAnalysis<circt::hw::InstanceGraph>();
  llvm::DenseSet<InstanceGraphNode *> visited;
  auto res = instanceGraph.getInferredTopLevelNodes();

  // Visit the instance hierarchy in a depth-first manner, modifying child
  // modules and their ports before their parents.

  // Doing this DFS ensures that all module instance uses of an inout value has
  // been converted before the current instance use. E.g. say you have m1 -> m2
  // -> m3 where both m3 and m2 reads an inout value defined in m1. If we don't
  // do DFS, and we just randomly pick a module, we have to e.g. select m2, see
  // that it also passes that inout value to other module instances, processes
  // those first (which may bubble up read/writes to that hw.inout op), and then
  // process m2... which in essence is a DFS traversal. So we just go ahead and
  // do the DFS to begin with, ensuring the invariant that all module instance
  // uses of an inout value have been converted before converting any given
  // module.

  for (InstanceGraphNode *topModule : res.value()) {
    for (InstanceGraphNode *node : llvm::post_order(topModule)) {
      if (visited.count(node))
        continue;
      auto mutableModule =
          dyn_cast_or_null<hw::HWMutableModuleLike>(*node->getModule());
      if (!mutableModule)
        continue;
      if (failed(PortConverter<HWInoutSignalStandardBuilder>(instanceGraph,
                                                             mutableModule)
                     .run()))
        return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> circt::sv::createHWRaiseInOutPortsPass() {
  return std::make_unique<HWRaiseInOutPortsPass>();
}
