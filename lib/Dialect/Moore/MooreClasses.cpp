//===- MooreClasses.cpp - Implement the Moore classes
//-------------------------===//
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

using namespace mlir;
using namespace circt::moore;

LogicalResult ClassFieldRefOp::verify() {
  // The operand is constrained to ClassHandleRefType in ODS; unwrap it.
  auto instRefTy = cast<RefType>(getInstance().getType());
  if (!instRefTy)
    return emitOpError("instance is not a !moore.ref<...>");

  auto handleTy = dyn_cast<ClassHandleType>(instRefTy.getNestedType());
  if (!handleTy)
    return emitOpError("instance must be !moore.ref<class.object<@C>>");

  // Extract the referenced class symbol from the handle type.
  SymbolRefAttr classSym = handleTy.getClassSym();
  if (!classSym)
    return emitOpError("instance type is missing a class symbol");

  // Resolve the class symbol starting from the nearest symbol table.
  Operation *clsSym =
      SymbolTable::lookupNearestSymbolFrom(getOperation(), classSym);
  if (!clsSym)
    return emitOpError("referenced class symbol `")
           << classSym << "` was not found";
  auto classDecl = dyn_cast<ClassDeclOp>(clsSym);
  if (!classDecl)
    return emitOpError("symbol `")
           << classSym << "` does not name a `moore.class.classdecl`";

  // Look up the field symbol inside the class declaration's symbol table.
  FlatSymbolRefAttr fieldSym = getFieldAttr();
  if (!fieldSym)
    return emitOpError("missing field symbol");

  Operation *fldSym =
      SymbolTable::lookupSymbolIn(classDecl, fieldSym.getAttr());
  if (!fldSym)
    return emitOpError("no field `") << fieldSym << "` in class " << classSym;

  auto fieldDecl = dyn_cast<ClassFieldDeclOp>(fldSym);
  if (!fieldDecl)
    return emitOpError("symbol `")
           << fieldSym << "` is not a `moore.class.fielddecl`";

  // Result must be !moore.ref<T> where T matches the field's declared type.
  auto resRefTy = cast<RefType>(getFieldRef().getType());
  if (!resRefTy)
    return emitOpError("result must be a !moore.ref<T>");

  Type expectedElemTy = fieldDecl.getFieldType();
  if (resRefTy.getNestedType() != expectedElemTy)
    return emitOpError("result element type (")
           << resRefTy.getNestedType() << ") does not match field type ("
           << expectedElemTy << ")";

  return success();
}

LogicalResult ClassNewOp::verify() {
  // The result is constrained to ClassHandleType in ODS, so this cast should be
  // safe.
  auto handleTy = cast<ClassHandleType>(getResult().getType());
  if (!handleTy)
    return emitOpError("result type not a ClassHandleType");

  mlir::SymbolRefAttr classSym = handleTy.getClassSym();
  if (!classSym)
    return emitOpError("result type is missing a class symbol");

  // Resolve the referenced symbol starting from the nearest symbol table.
  mlir::Operation *sym =
      mlir::SymbolTable::lookupNearestSymbolFrom(getOperation(), classSym);
  if (!sym)
    return emitOpError("referenced class symbol `")
           << classSym << "` was not found";

  if (!llvm::isa<ClassDeclOp>(sym))
    return emitOpError("symbol `")
           << classSym << "` does not name a `moore.class.classdecl`";

  return mlir::success();
}

LogicalResult ClassDeclOp::verify() {
  auto &block = getBody().front();
  for (mlir::Operation &op : block) {

    // allow only field and method decls and terminator
    if (llvm::isa<circt::moore::ClassBodyEndOp, circt::moore::ClassFieldDeclOp,
                  circt::moore::ClassMethodDeclOp>(&op))
      continue;

    return emitOpError() << "body may only contain 'moore.class.fielddecl' and "
                            "'moore.class.methoddecl' operations";
  }
  return mlir::success();
}
