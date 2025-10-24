//===- MooreClasses.cpp - Implement the Moore classes ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the Moore dialect class system.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Moore/MooreDialect.h"
#include "circt/Dialect/Moore/MooreOps.h"
#include "circt/Dialect/Moore/MooreTypes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

using namespace mlir;
using namespace circt::moore;

LogicalResult ClassDeclOp::verify() {
  auto &block = getBody().front();
  for (mlir::Operation &op : block) {

    // allow only property and method decls and terminator
    if (llvm::isa<circt::moore::ClassBodyEndOp,
                  circt::moore::ClassPropertyDeclOp>(&op))
      continue;

    return emitOpError()
           << "body may only contain 'moore.class.propertydecl' operations";
  }
  return mlir::success();
}
