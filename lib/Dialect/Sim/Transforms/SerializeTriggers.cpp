//===- SerializeTriggers.cpp - TODO ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Sim/SimOps.h"
#include "circt/Dialect/Sim/SimTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "serialize-triggers"

namespace circt {
namespace sim {
#define GEN_PASS_DEF_SERIALIZETRIGGERS
#include "circt/Dialect/Sim/SimPasses.h.inc"
} // namespace sim
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace sim;

namespace {
struct SerializeTriggersPass
    : sim::impl::SerializeTriggersBase<SerializeTriggersPass> {
public:
  void runOnOperation() override;

private:
};

static Value getParentTrigger(Operation *op) {
  if (!op)
    return {};
  return TypeSwitch<Operation *, Value>(op)
      .Case<TriggeredOp>([](auto op) { return op.getTrigger(); })
      .Case<TriggerGateOp>([](auto op) { return op.getInput(); })
      .Case<TriggerSequenceOp>([](auto op) { return op.getParent(); })
      .Default([](auto) { return Value{}; });
}

static void setParentTrigger(Operation *op, Value trigger) {
  TypeSwitch<Operation *>(op)
      .Case<TriggeredOp>(
          [&](auto op) { op.getTriggerMutable().assign(trigger); })
      .Case<TriggerGateOp>(
          [&](auto op) { op.getInputMutable().assign(trigger); })
      .Case<TriggerSequenceOp>(
          [&](auto op) { op.getParentMutable().assign(trigger); })
      .Default([](auto) { assert(false && "Not a valid trigger user op"); });
}

} // namespace

void SerializeTriggersPass::runOnOperation() {
  auto theModule = getOperation();

  OpBuilder builder(theModule);

  bool anyChanged = false;

  struct ValueMapEntry {
    size_t length;
    size_t index;
    TriggerSequenceOp sequence;
  };

  DenseMap<Value, ValueMapEntry> concurrentTriggers;

  theModule.walk([&](Operation *op) {
    auto parentTrigger = getParentTrigger(op);
    if (!parentTrigger || parentTrigger.use_empty() ||
        parentTrigger.hasOneUse())
      return;

    auto mapEntryIt = concurrentTriggers.find(parentTrigger);
    if (mapEntryIt == concurrentTriggers.end()) {
      anyChanged = true;
      auto defOp = parentTrigger.getDefiningOp();
      if (!!defOp && defOp->getBlock() == op->getBlock())
        builder.setInsertionPoint(op);
      else
        builder.setInsertionPointToStart(theModule.getBodyBlock());
      Location loc = !defOp ? theModule.getLoc() : defOp->getLoc();
      size_t numUsers =
          std::distance(parentTrigger.use_begin(), parentTrigger.use_end());
      auto sequence =
          builder.create<TriggerSequenceOp>(loc, parentTrigger, numUsers);
      setParentTrigger(op, sequence.getResult(0));
      concurrentTriggers.insert(std::pair<Value, ValueMapEntry>(
          parentTrigger, ValueMapEntry{numUsers, 1, sequence}));
    } else {
      auto &index = mapEntryIt->second.index;
      setParentTrigger(op, mapEntryIt->second.sequence.getResult(index));
      ++index;
      if (index == mapEntryIt->second.length)
        concurrentTriggers.erase(mapEntryIt);
    }
  });

  if (!concurrentTriggers.empty()) {
    signalPassFailure();
    for (auto leftoverEntry : concurrentTriggers) {
      for (auto user : leftoverEntry.getFirst().getUsers())
        user->emitError("Failed to serialize trigger user op.");
    }
  }

  if (!anyChanged)
    markAllAnalysesPreserved();
}
