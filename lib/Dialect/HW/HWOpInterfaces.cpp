//===- HWOpInterfaces.cpp - Implement the HW op interfaces ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implement the HW operation interfaces.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypeInterfaces.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

using namespace circt;

LogicalResult hw::verifyInnerSymAttr(InnerSymbolOpInterface op) {
  auto innerSym = op.getInnerSymAttr();
  // If does not have any inner sym then ignore.
  if (!innerSym)
    return success();
  if (isa<InstanceOp>(&op) || op->getNumResults() != 1) {
    // The inner sym can only be specified on fieldID=0.
    if (innerSym.size() > 1 || !innerSym.getSymName()) {
      op->emitOpError("cannot assign symbols to non-zero field id, for ops "
                      "with zero or multiple results");
      return failure();
    }
    return success();
  }

  auto resultType =
      op->getResultTypes()[0].dyn_cast<hw::FieldIDTypeInterface>();
  // If this type doesn't implement the FieldIDTypeInterface, then there is
  // nothing additional to check.
  if (!resultType)
    return success();
  auto maxFields = resultType.getMaxFieldID();
  llvm::SmallBitVector indices(maxFields + 1);
  llvm::SmallPtrSet<Attribute, 8> symNames;
  // Ensure fieldID and symbol names are unique.
  auto uniqSyms = [&](hw::InnerSymPropertiesAttr p) {
    if (maxFields < p.getFieldID()) {
      op->emitOpError("field id:'" + Twine(p.getFieldID()) +
                      "' is greater than the maximum field id:'" +
                      Twine(maxFields) + "'");
      return false;
    }
    if (indices.test(p.getFieldID())) {
      op->emitOpError("cannot assign multiple symbol names to the field id:'" +
                      Twine(p.getFieldID()) + "'");
      return false;
    }
    indices.set(p.getFieldID());
    auto it = symNames.insert(p.getName());
    if (!it.second) {
      op->emitOpError("cannot reuse symbol name:'" + p.getName().getValue() +
                      "'");
      return false;
    }
    return true;
  };

  if (!llvm::all_of(innerSym.getProps(), uniqSyms))
    return failure();

  return success();
}

#include "circt/Dialect/HW/HWOpInterfaces.cpp.inc"
