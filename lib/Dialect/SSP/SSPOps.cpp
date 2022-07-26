//===- SSPOps.cpp - SSP operation implementation --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SSP (static scheduling problem) dialect operations.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SSP/SSPOps.h"
#include "circt/Dialect/SSP/SSPAttributes.h"
#include "circt/Support/LLVM.h"

#include "mlir/IR/Builders.h"

using namespace circt;
using namespace circt::ssp;

//===----------------------------------------------------------------------===//
// InstanceOp
//===----------------------------------------------------------------------===//

LogicalResult InstanceOp::verify() {
  auto &ops = getBodyBlock()->getOperations();
  auto it = ops.begin();
  if (ops.size() != 2 || !isa<OperatorLibraryOp>(*it) ||
      !isa<DependenceGraphOp>(*(++it)))
    return emitOpError()
           << "must contain exactly one 'library' op and one 'graph' op";

  return success();
}

// The verifier checks that exactly one of each of the container ops is present.
OperatorLibraryOp InstanceOp::getOperatorLibrary() {
  return *getOps<OperatorLibraryOp>().begin();
}

DependenceGraphOp InstanceOp::getDependenceGraph() {
  return *getOps<DependenceGraphOp>().begin();
}

//===----------------------------------------------------------------------===//
// Wrappers for the `custom<Properties>` ODS directive.
//===----------------------------------------------------------------------===//

static ParseResult parseProperties(OpAsmParser &parser, ArrayAttr &attr) {
  auto result = parseOptionalPropertyArray(attr, parser);
  if (!result.hasValue() || succeeded(*result))
    return success();
  return failure();
}

static void printProperties(OpAsmPrinter &p, Operation *op, ArrayAttr attr) {
  if (!attr)
    return;
  printPropertyArray(attr, p);
}

//===----------------------------------------------------------------------===//
// TableGen'ed code
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "circt/Dialect/SSP/SSP.cpp.inc"
