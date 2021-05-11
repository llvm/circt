//===- Scheduling.cpp - Scheduling dialect implementation -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Scheduling dialect.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Scheduling/Scheduling.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/RegionKindInterface.h"

#include "llvm/ADT/PriorityWorklist.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace circt::sched;

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Scheduling/SchedulingAttributes.cpp.inc"

#include "circt/Dialect/Scheduling/SchedulableOpInterface.cpp.inc"

void SchedulingDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/Scheduling/SchedulingAttributes.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// OperatorInfo attribute
//===----------------------------------------------------------------------===//

Attribute OperatorInfoAttr::parse(MLIRContext *context, DialectAsmParser &p,
                                  Type type) {
  StringRef name;
  unsigned latency;
  if (p.parseLess() || p.parseOptionalString(&name) || p.parseComma() ||
      p.parseKeyword("latency") || p.parseEqual() || p.parseInteger(latency) ||
      p.parseGreater())
    return {};

  return OperatorInfoAttr::get(context, name, latency);
}

void OperatorInfoAttr::print(DialectAsmPrinter &p) const {
  p << getMnemonic() << "<\"" << getName() << "\", latency=" << getLatency()
    << '>';
}

//===----------------------------------------------------------------------===//
// Dispatching of dialect attribute parsing and printing
//===----------------------------------------------------------------------===//

Attribute SchedulingDialect::parseAttribute(DialectAsmParser &p,
                                            Type type) const {
  StringRef mnemonic;
  if (p.parseKeyword(&mnemonic))
    return {};

  Attribute attr;
  if (generatedAttributeParser(getContext(), p, mnemonic, type, attr)
          .hasValue())
    return attr;

  p.emitError(p.getNameLoc(), "Unexpected sched attribute '" + mnemonic + "'");
  return {};
}

void SchedulingDialect::printAttribute(Attribute attr,
                                       DialectAsmPrinter &p) const {
  if (succeeded(generatedAttributePrinter(attr, p)))
    return;
  llvm_unreachable("Unexpected attribute");
}

//===----------------------------------------------------------------------===//
// A very simple acyclic as-soon-as-possible list scheduler
//===----------------------------------------------------------------------===//

namespace {
class ASAPScheduler : public SchedulerBase {
private:
  llvm::SmallDenseMap<mlir::Operation *, unsigned> startTimes;

public:
  mlir::LogicalResult
  schedule(circt::sched::SchedulableOpInterface schedulableOp) override;
  mlir::Optional<unsigned> getStartTime(mlir::Operation *scheduledOp) override;
};
} // anonymous namespace

LogicalResult ASAPScheduler::schedule(SchedulableOpInterface schedulableOp) {
  auto &blockToSchedule = schedulableOp.getBlockToSchedule();
  auto &operationsToSchedule = blockToSchedule.getOperations();
  startTimes.clear();

  // initialize a worklist with the block's operations
  llvm::PriorityWorklist<Operation *> worklist;
  for (auto revIt = operationsToSchedule.rbegin(),
            revEnd = operationsToSchedule.rend(); revIt != revEnd; ++revIt)
    worklist.insert(&*revIt);

  // iterate until all operations are scheduled
  while (!worklist.empty()) {
    Operation *op = worklist.pop_back_val();
    if (op->getNumOperands() == 0) {
      // operations with no predecessors are scheduled at time step 0
      startTimes.insert(std::make_pair(op, 0));
      continue;
    }

    // `unscheduled` has at least one predecessor. Compute start time as:
    //   max_{p : preds} startTime[p] + latency[p]
    unsigned startTime = 0;
    bool isValid = true;
    for (Value operand : op->getOperands()) {
      if (operand.isa<BlockArgument>())
        continue; // block arguments are available at time step 0

      Operation *operandOp = operand.getDefiningOp();
      if (operandOp->getBlock() != &blockToSchedule)
        continue; // handle values computed in different blocks analogous to
                  // block arguments. TOOD: should these occur?

      auto it = startTimes.find(operandOp);
      if (it != startTimes.end()) {
        // operand is already scheduled
        unsigned operandStart = it->getSecond();
        unsigned operandLatency =
            schedulableOp.getOperatorInfo(operandOp).getLatency();
        startTime = std::max(startTime, operandStart + operandLatency);
      } else {
        // operand is not yet scheduled, but will be. Re-enqueue `op` and give
        // for now
        assert(worklist.count(operandOp));
        isValid = false;
        break;
      }
    }

    if (isValid)
      // operands are arguments or were already scheduled
      startTimes.insert(std::make_pair(op, startTime));
    else
      // not all operands were scheduled; try again later
      worklist.insert(op);
  }

  return LogicalResult::success();
}

Optional<unsigned> ASAPScheduler::getStartTime(Operation *scheduledOp) {
  auto it = startTimes.find(scheduledOp);
  if (it != startTimes.end())
    return it->getSecond();
  return None;
}

std::unique_ptr<SchedulerBase> circt::sched::createASAPScheduler() {
  return std::make_unique<ASAPScheduler>();
}
