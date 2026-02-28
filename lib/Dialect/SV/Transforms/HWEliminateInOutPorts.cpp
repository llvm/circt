//===- HWEliminateInOutPorts.cpp - Generator Callout Pass
//---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/PortConverter.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/PostOrderIterator.h"

namespace circt {
namespace sv {
#define GEN_PASS_DEF_HWELIMINATEINOUTPORTS
#include "circt/Dialect/SV/SVPasses.h.inc"
} // namespace sv
} // namespace circt

using namespace circt;
using namespace sv;
using namespace hw;
using namespace igraph;

namespace {

struct HWEliminateInOutPortsPass
    : public circt::sv::impl::HWEliminateInOutPortsBase<
          HWEliminateInOutPortsPass> {
  using HWEliminateInOutPortsBase<
      HWEliminateInOutPortsPass>::HWEliminateInOutPortsBase;
  void runOnOperation() override;
};

class HWInOutPortConversion : public PortConversion {
public:
  HWInOutPortConversion(PortConverterImpl &converter, hw::PortInfo port,
                        llvm::StringRef readSuffix,
                        llvm::StringRef writeSuffix);

  void mapInputSignals(OpBuilder &b, Operation *inst, Value instValue,
                       SmallVectorImpl<Value> &newOperands,
                       ArrayRef<Backedge> newResults) override;
  void mapOutputSignals(OpBuilder &b, Operation *inst, Value instValue,
                        SmallVectorImpl<Value> &newOperands,
                        ArrayRef<Backedge> newResults) override;

  LogicalResult init() override;

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

  // Suffix to be used when creating read ports.
  llvm::StringRef readSuffix;
  // Suffix to be used when creating write ports.
  llvm::StringRef writeSuffix;
};

HWInOutPortConversion::HWInOutPortConversion(PortConverterImpl &converter,
                                             hw::PortInfo port,
                                             llvm::StringRef readSuffix,
                                             llvm::StringRef writeSuffix)
    : PortConversion(converter, port), readSuffix(readSuffix),
      writeSuffix(writeSuffix) {}

LogicalResult HWInOutPortConversion::init() {
  // Gather readers and writers (how to handle sv.passign?)
  for (auto *user : body->getArgument(origPort.argNum).getUsers()) {
    if (auto read = dyn_cast<sv::ReadInOutOp>(user))
      readers.push_back(read);
    else if (auto write = dyn_cast<sv::AssignOp>(user))
      writers.push_back(write);
    else
      return user->emitOpError() << "uses hw.inout port " << origPort.name
                                 << " but the operation itself is unsupported.";
  }

  if (writers.size() > 1)
    return converter.getModule()->emitOpError()
           << "multiple writers of inout port " << origPort.name
           << " is unsupported.";

  return success();
}

void HWInOutPortConversion::buildInputSignals() {
  if (hasReaders()) {
    // Replace all sv::ReadInOutOp's with the new input.
    Value readValue =
        converter.createNewInput(origPort, readSuffix, origPort.type, readPort);
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
    // Replace the sv::AssignOp with the new output.
    sv::AssignOp write = writers.front();
    converter.createNewOutput(origPort, writeSuffix, origPort.type,
                              write.getSrc(), writePort);
    write.erase();
  }
}

void HWInOutPortConversion::buildOutputSignals() {
  assert(false &&
         "`hw.inout` outputs not yet supported. Currently, `hw.inout` "
         "outputs are handled by UntouchedPortConversion, given that "
         "output `hw.inout` ports have a `ModulePort::Direction::Output` "
         "direction instead of `ModulePort::Direction::InOut`. If this for "
         "some reason changes, then this assert will fire.");
}

void HWInOutPortConversion::mapInputSignals(OpBuilder &b, Operation *inst,
                                            Value instValue,
                                            SmallVectorImpl<Value> &newOperands,
                                            ArrayRef<Backedge> newResults) {

  if (hasReaders()) {
    // Create a read_inout op at the instantiation point. This effectively
    // pushes the read_inout op from the module to the instantiation site.
    newOperands[readPort.argNum] =
        ReadInOutOp::create(b, inst->getLoc(), instValue).getResult();
  }

  if (hasWriters()) {
    // Create a sv::AssignOp at the instantiation point. This effectively
    // pushes the write op from the module to the instantiation site.
    Value writeFromInsideMod = newResults[writePort.argNum];
    sv::AssignOp::create(b, inst->getLoc(), instValue, writeFromInsideMod);
  }
}

void HWInOutPortConversion::mapOutputSignals(
    OpBuilder &b, Operation *inst, Value instValue,
    SmallVectorImpl<Value> &newOperands, ArrayRef<Backedge> newResults) {
  // FIXME: hw.inout cannot be used in outputs.
  assert(false &&
         "`hw.inout` outputs not yet supported. Currently, `hw.inout` "
         "outputs are handled by UntouchedPortConversion, given that "
         "output `hw.inout` ports have a `ModulePort::Direction::Output` "
         "direction instead of `ModulePort::Direction::InOut`. If this for "
         "some reason changes, then this assert will fire.");
}

class HWInoutPortConversionBuilder : public PortConversionBuilder {
public:
  HWInoutPortConversionBuilder(PortConverterImpl &converter,
                               llvm::StringRef readSuffix,
                               llvm::StringRef writeSuffix)
      : PortConversionBuilder(converter), readSuffix(readSuffix),
        writeSuffix(writeSuffix) {}

  FailureOr<std::unique_ptr<PortConversion>> build(hw::PortInfo port) override {
    if (port.dir == hw::ModulePort::Direction::InOut)
      return {std::make_unique<HWInOutPortConversion>(converter, port,
                                                      readSuffix, writeSuffix)};
    return PortConversionBuilder::build(port);
  }

private:
  llvm::StringRef readSuffix;
  llvm::StringRef writeSuffix;
};

} // namespace

void HWEliminateInOutPortsPass::runOnOperation() {
  // Find all modules and run port conversion on them.
  circt::hw::InstanceGraph &instanceGraph =
      getAnalysis<circt::hw::InstanceGraph>();
  llvm::DenseSet<InstanceGraphNode *> visited;
  FailureOr<llvm::ArrayRef<InstanceGraphNode *>> res =
      instanceGraph.getInferredTopLevelNodes();

  if (failed(res)) {
    signalPassFailure();
    return;
  }

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
      if (failed(PortConverter<HWInoutPortConversionBuilder>(
                     instanceGraph, mutableModule, readSuffix.getValue(),
                     writeSuffix.getValue())
                     .run()))
        return signalPassFailure();
    }
  }
}

std::unique_ptr<Pass> circt::sv::createHWEliminateInOutPortsPass(
    const HWEliminateInOutPortsOptions &options) {
  return std::make_unique<HWEliminateInOutPortsPass>(options);
}
