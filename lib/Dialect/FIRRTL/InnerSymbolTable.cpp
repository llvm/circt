//===- InnerSymbolTable.cpp - InnerSymbolTable and InnerRef verification --===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements InnerSymbolTable and verification for InnerRef's.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOpInterfaces.h"
#include "mlir/IR/Threading.h"

using namespace circt;
using namespace firrtl;

namespace circt {
namespace firrtl {

//===----------------------------------------------------------------------===//
// InnerSymbolTable
//===----------------------------------------------------------------------===//

InnerSymbolTable::InnerSymbolTable(Operation *op) {
  assert(op->hasTrait<OpTrait::InnerSymbolTable>() &&
         "expected operation to have InnerSymbolTable trait");
  // Save the operation this table is for.
  this->innerSymTblOp = op;

  // Walk the operation and add InnerSymbol's to the table.
  StringAttr innerSymId = StringAttr::get(
      op->getContext(), InnerSymbolTable::getInnerSymbolAttrName());
  op->walk([&](Operation *symOp) {
    auto attr = symOp->getAttrOfType<StringAttr>(innerSymId);
    if (!attr)
      return;
    auto it = symbolTable.insert({attr, symOp});
    (void)it;
    assert(it.second && "repeated symbol found");
  });
}

/// Look up a symbol with the specified name, returning null if no such name
/// exists. Names never include the @ on them.
Operation *InnerSymbolTable::lookup(StringRef name) const {
  return lookup(StringAttr::get(innerSymTblOp->getContext(), name));
}
Operation *InnerSymbolTable::lookup(StringAttr name) const {
  return symbolTable.lookup(name);
}

//===----------------------------------------------------------------------===//
// InnerSymbolTableCollection
//===----------------------------------------------------------------------===//

InnerSymbolTable &
InnerSymbolTableCollection::getInnerSymbolTable(Operation *op) {
  auto it = symbolTables.try_emplace(op, nullptr);
  if (it.second)
    it.first->second = ::std::make_unique<InnerSymbolTable>(op);
  return *it.first->second;
}

void InnerSymbolTableCollection::populateTables(Operation *innerRefNSOp) {
  // Gather top-level operations.
  SmallVector<Operation *> childOps(
      llvm::make_pointer_range(innerRefNSOp->getRegion(0).front()));

  // Filter these to those that have the InnerSymbolTable trait.
  SmallVector<Operation *> innerSymTableOps(
      llvm::make_filter_range(childOps, [&](Operation *op) {
        return op->hasTrait<OpTrait::InnerSymbolTable>();
      }));

  // Ensure entries exist for each operation.
  llvm::for_each(innerSymTableOps,
                 [&](auto *op) { symbolTables.try_emplace(op, nullptr); });

  // Construct the tables in parallel (if context allows it).
  mlir::parallelForEach(
      innerRefNSOp->getContext(), innerSymTableOps, [&](auto *op) {
        auto it = symbolTables.find(op);
        assert(it != symbolTables.end());
        if (!it->second)
          it->second = ::std::make_unique<InnerSymbolTable>(op);
      });
}

//===----------------------------------------------------------------------===//
// InnerRefNamespace
//===----------------------------------------------------------------------===//

Operation *InnerRefNamespace::lookup(hw::InnerRefAttr inner) {
  auto *mod = symTable.lookup(inner.getModule());
  assert(mod->hasTrait<mlir::OpTrait::InnerSymbolTable>());
  return innerSymTables.getInnerSymbolTable(mod).lookup(inner.getName());
}

//===----------------------------------------------------------------------===//
// InnerRef verification
//===----------------------------------------------------------------------===//

namespace detail {

LogicalResult verifyInnerRefs(Operation *op) {
  // Construct the symbol tables.
  InnerSymbolTableCollection innerSymTables;
  SymbolTable symbolTable(op);
  InnerRefNamespace ns{symbolTable, innerSymTables};
  innerSymTables.populateTables(op);

  // Conduct parallel walks of the top-level children of this
  // InnerRefNamespace, verifying all InnerRefUserOp's discovered within.
  auto verifySymbolUserFn = [&](Operation *op) -> WalkResult {
    if (auto user = dyn_cast<InnerRefUserOpInterface>(op))
      return WalkResult(user.verifyInnerRefs(ns));
    return WalkResult::advance();
  };
  return mlir::failableParallelForEach(
      op->getContext(), op->getRegion(0).front(), [&](auto &op) {
        return success(!op.walk(verifySymbolUserFn).wasInterrupted());
      });
}

} // namespace detail
} // namespace firrtl
} // namespace circt
