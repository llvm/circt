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

#include "circt/Dialect/Scheduling/SchedulingDialect.h"
#include "circt/Dialect/Scheduling/SchedulingInterfaces.h"
#include "circt/Dialect/Scheduling/SchedulingAttributes.h"
#include "circt/Dialect/Scheduling/Algorithms/ASAPScheduler.h"

#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

#include <list>

using namespace mlir;
using namespace circt;
using namespace circt::sched;

//===----------------------------------------------------------------------===//
// Interfaces
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Scheduling/SchedulingInterfaces.cpp.inc"

//===----------------------------------------------------------------------===//
// Attributes
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/Scheduling/SchedulingAttributes.cpp.inc"

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

LogicalResult ASAPScheduler::registerOperation(Operation *op) {
  return success(operations.insert(op));
}
LogicalResult ASAPScheduler::registerDependence(Operation *src,
                                                unsigned int srcIdx,
                                                Operation *dst,
                                                unsigned int dstIdx,
                                                unsigned int distance) {
  if (srcIdx > 0 || dstIdx > 0 || distance > 0)
    return failure(); // or could just warn that backedges are not supported
  if (!operations.contains(src) || !operations.contains(dst))
    return failure();
  dependences[dst].push_back(src);
  return success();
}
LogicalResult
ASAPScheduler::registerOperators(Operation *op,
                                 ArrayRef<OperatorInfoAttr> operatorInfos) {
  if (operatorInfos.size() != 1)
    return failure();
  if (!operations.contains(op))
    return failure();
  operators[op] = operatorInfos.front();
  return success();
}

LogicalResult ASAPScheduler::schedule() {
  startTimes.clear();

  // TODO: validate scheduling problem, e.g. check for cycles

  // initialize a worklist with the block's operations
  std::list<Operation *> worklist;
  worklist.insert(worklist.begin(), operations.begin(), operations.end());

  // iterate until all operations are scheduled
  while (!worklist.empty()) {
    Operation *op = worklist.front();
    worklist.pop_front();
    if (dependences[op].empty()) {
      // operations with no predecessors are scheduled at time step 0
      startTimes[op] = 0;
      continue;
    }

    // op has at least one predecessor. Compute start time as:
    //   max_{p : preds} startTime[p] + latency[p]
    unsigned startTime = 0;
    bool startTimeIsValid = true;
    for (Operation *pred : dependences[op]) {
      auto it = startTimes.find(pred);
      if (it != startTimes.end()) {
        // pred is already scheduled
        unsigned predStart = it->getSecond();
        unsigned predLatency = operators[pred].getLatency();
        startTime = std::max(startTime, predStart + predLatency);
      } else {
        // pred is not yet scheduled, give up and try again later
        startTimeIsValid = false;
        break;
      }
    }

    if (startTimeIsValid)
      startTimes[op] = startTime;
    else
      worklist.push_back(op);
  }

  return LogicalResult::success();
}

Optional<unsigned> ASAPScheduler::getStartTime(Operation *scheduledOp) {
  auto it = startTimes.find(scheduledOp);
  if (it != startTimes.end())
    return it->getSecond();
  return None;
}
