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
      opCounts[op->getName()]++;

    // Update operandCounts
    if (operandCounts.find(opName) == operandCounts.end())
      operandCounts[opName] = DenseMap<size_t, size_t>();
    if (operandCounts[opName].find(op->getNumOperands()) ==
        operandCounts[opName].end())
      operandCounts[opName][op->getNumOperands()] = 1;
    else
      operandCounts[opName][op->getNumOperands()]++;
  });
}

SmallVector<OperationName> OpCountAnalysis::getFoundOpNames() {
  SmallVector<OperationName> opNames;
  for (auto pair : opCounts)
    opNames.push_back(pair.first);
  return opNames;
}

size_t OpCountAnalysis::getOpCount(OperationName opName) {
  size_t count = 0;
  if (opCounts.find(opName) != opCounts.end())
    count = opCounts[opName];
  return count;
}

DenseMap<size_t, size_t>
OpCountAnalysis::getOperandCountMap(OperationName opName) {
  auto map = DenseMap<size_t, size_t>();
  if (operandCounts.find(opName) != operandCounts.end())
    map = operandCounts[opName];
  return map;
}
