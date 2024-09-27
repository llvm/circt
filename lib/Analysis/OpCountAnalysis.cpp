//===- OpCountAnalysis.cpp - operation count analyses -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the op count analysis.  This is an analysis that
// provides information about the frequency of different kinds of operations
// found in a builtin.module.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/OpCountAnalysis.h"
#include "mlir/IR/Operation.h"

using namespace circt;
using namespace analysis;

OpCountAnalysis::OpCountAnalysis(Operation *moduleOp,
                                 mlir::AnalysisManager &am) {
  moduleOp->walk([&](Operation *op) {
    auto opName = op->getName();
    // Update opCounts
    if (opCounts.find(opName) == opCounts.end())
      opCounts[opName] = 1;
    else
      opCounts[opName]++;

    // Update operandCounts
    auto &counts = operandCounts[opName];
    if (counts.find(op->getNumOperands()) == counts.end())
      counts[op->getNumOperands()] = 1;
    else
      counts[op->getNumOperands()]++;
  });
}

SmallVector<OperationName> OpCountAnalysis::getFoundOpNames() {
  SmallVector<OperationName> opNames;
  for (auto pair : opCounts)
    opNames.push_back(pair.first);
  return opNames;
}

size_t OpCountAnalysis::getOpCount(OperationName opName) {
  if (opCounts.find(opName) != opCounts.end())
    return opCounts[opName];
  return 0;
}

DenseMap<size_t, size_t>
OpCountAnalysis::getOperandCountMap(OperationName opName) {
  return operandCounts[opName];
}
