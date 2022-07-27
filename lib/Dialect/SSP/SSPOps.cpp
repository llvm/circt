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
  auto *body = getBodyBlock();
  auto libraryOps = body->getOps<OperatorLibraryOp>();
  auto graphOps = body->getOps<DependenceGraphOp>();

  if (std::distance(libraryOps.begin(), libraryOps.end()) != 1 ||
      std::distance(graphOps.begin(), graphOps.end()) != 1)
    return emitOpError()
           << "must contain exactly one 'library' op and one 'graph' op";

  if ((*graphOps.begin())->isBeforeInBlock(*libraryOps.begin()))
    return emitOpError()
           << "must contain the 'library' op followed by the 'graph' op";

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
