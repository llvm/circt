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
#include "mlir/Interfaces/SideEffectInterfaces.h"

using namespace mlir;
using namespace circt::moore;

/// Resolve a symbol relative to the nearest table.
static Operation *lookupNearest(Operation *from, SymbolRefAttr sym) {
  return SymbolTable::lookupNearestSymbolFrom(from, sym);
}

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

LogicalResult ClassNewOp::verify() {
  // The result is constrained to ClassHandleType in ODS, so this cast should be
  // safe.
  auto handleTy = cast<ClassHandleType>(getResult().getType());
  mlir::SymbolRefAttr classSym = handleTy.getClassSym();
  if (!classSym)
    return emitOpError("result type is missing a class symbol");

  // Resolve the referenced symbol starting from the nearest symbol table.
  mlir::Operation *sym = lookupNearest(getOperation(), classSym);
  if (!sym)
    return emitOpError("referenced class symbol `")
           << classSym << "` was not found";

  if (!llvm::isa<ClassDeclOp>(sym))
    return emitOpError("symbol `")
           << classSym << "` does not name a `moore.class.classdecl`";

  return mlir::success();
}

void ClassNewOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  // Always allocates heap memory.
  effects.emplace_back(MemoryEffects::Allocate::get());
}
