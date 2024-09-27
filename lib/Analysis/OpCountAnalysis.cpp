//===- OpCountAnalysis.cpp - operation count analyses -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the InstanceInfo analysis.  This is an analysis that
// depends on the InstanceGraph analysis, but provides additional information
// about FIRRTL operations.  This is useful if you find yourself needing to
// selectively iterate over parts of the design.
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/OpCountAnalysis.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"

using namespace circt;

circt::analysis::OpCountAnalysis::OpCountAnalysis(Operation *moduleOp, mlir::AnalysisManager &am) {
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
    if (operandCounts[opName].find(op->getNumOperands()) == operandCounts[opName].end())
      operandCounts[opName][op->getNumOperands()] = 1;
    else
      operandCounts[opName][op->getNumOperands()]++;
  });
}

size_t circt::analysis::OpCountAnalysis::getOpCount(OperationName opName) {
  size_t count = 0;
  if (opCounts.find(opName) != opCounts.end())
    count = opCounts[opName];
  return count;
}

DenseMap<size_t, size_t> circt::analysis::OpCountAnalysis::getOperandCountMap(OperationName opName) {
  auto map = DenseMap<size_t, size_t>();
  if (operandCounts.find(opName) != operandCounts.end())
    map = operandCounts[opName];
  return map;
}