//===- OMOps.h - Object Model operation declarations ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains the Object Model operation declarations.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_OM_OMOPS_H
#define CIRCT_DIALECT_OM_OMOPS_H

#include "circt/Dialect/OM/OMAttributes.h"
#include "circt/Dialect/OM/OMDialect.h"
#include "circt/Dialect/OM/OMOpInterfaces.h"
#include "circt/Dialect/OM/OMTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "circt/Dialect/OM/OM.h.inc"

namespace circt::om {

class ClassOpBuilder : public mlir::OpBuilder {
public:
  ClassOpBuilder(mlir::Block *block, mlir::Block::iterator insertPoint,
            OpBuilder::Listener *listener = nullptr)
      : OpBuilder(block->getParent()->getContext(), listener) {
    setInsertionPoint(block, insertPoint);
  }
  static ClassOpBuilder atBlockBegin(mlir::Block *block, OpBuilder::Listener *listener = nullptr) {
    return ClassOpBuilder(block, block->begin(), listener);
  }
  void createClassFieldsOp(llvm::ArrayRef<mlir::Location> locs,
                           llvm::ArrayRef<mlir::Attribute> fieldNames,
                           llvm::ArrayRef<mlir::Value> fieldValues) {
    auto op = this->create<ClassFieldsOp>(this->getFusedLoc(locs), fieldValues);
    op.getOperation()->setAttr(
        "field_names", mlir::ArrayAttr::get(this->getContext(), fieldNames));
  }
};

} // namespace circt::om

#endif // CIRCT_DIALECT_OM_OMOPS_H
