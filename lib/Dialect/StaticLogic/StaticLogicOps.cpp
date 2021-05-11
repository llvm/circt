//===- StaticLogicOps.h - StaticLogic MLIR Operations -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the StaticLogic ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/StaticLogic/StaticLogic.h"
#include "mlir/IR/FunctionImplementation.h"

using namespace circt;
using namespace circt::staticlogic;

using circt::sched::OperatorInfoAttr;

#define GET_OP_CLASSES
#include "circt/Dialect/StaticLogic/StaticLogic.cpp.inc"

void StaticLogicDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "circt/Dialect/StaticLogic/StaticLogic.cpp.inc"
      >();
}

OperatorInfoAttr PipelineOp::getOperatorInfo(Operation *op) {
  auto operatorInfo = op->getAttrOfType<OperatorInfoAttr>("opr");
  if (operatorInfo)
    return operatorInfo;
  return OperatorInfoAttr::get(getContext(), "unit", 1);
}

Block& PipelineOp::getBlockToSchedule() {
  return getRegion().front();
}
