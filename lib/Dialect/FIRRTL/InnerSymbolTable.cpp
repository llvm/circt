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

#include "circt/Dialect/FIRRTL/InnerSymbolTable.h"
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

  auto addSym = [&](StringAttr name, InnerSymTarget target) {
    assert(name && !name.getValue().empty());
    auto it = symbolTable.insert({name, target});
    if (!it.second) {
      auto orig = symbolTable.lookup(name);
      (target.getOp()->emitError("duplicate symbol @") << name << " found")
              .attachNote(orig.getOp()->getLoc())
          << " symbol also found within this operation";
      // TODO: rework so can indicate failure to caller, for use in verif/etc?
    }
    assert(it.second && "repeated symbol found");
  };
  auto addSyms = [&](InnerSymAttr symAttr, InnerSymTarget baseTarget) {
    if (!symAttr)
      return;
    assert(baseTarget.getField() == 0);
    for (const auto &symProp : symAttr.getProps()) {
      addSym(symProp.getName(),
             InnerSymTarget(baseTarget, symProp.getFieldID()));
    }
  };

  // Walk the operation and add InnerSymbolTarget's to the table.
  op->walk([&](Operation *curOp) {
    if (auto symOp = dyn_cast<InnerSymbolOpInterface>(curOp))
      addSyms(symOp.getInnerSymAttr(), InnerSymTarget(symOp));

    // Check for ports
    // TODO: investigate why/confirm ports having empty-string symbols is normal
    // TODO: Add fields per port, once they work that way (use addSyms)
    if (auto mod = dyn_cast<FModuleLike>(curOp)) {
      for (const auto &p : llvm::enumerate(mod.getPorts()))
        if (auto sym = p.value().sym; sym && !sym.getValue().empty())
          addSym(p.value().sym, InnerSymTarget(p.index(), curOp));
    }
  });
}

/// Look up a symbol with the specified name, returning empty InnerSymTarget if
/// no such name exists. Names never include the @ on them.
InnerSymTarget InnerSymbolTable::lookup(StringRef name) const {
  return lookup(StringAttr::get(innerSymTblOp->getContext(), name));
}
InnerSymTarget InnerSymbolTable::lookup(StringAttr name) const {
  return symbolTable.lookup(name);
}

/// Look up a symbol with the specified name, returning null if no such
/// name exists or doesn't target just an operation.
Operation *InnerSymbolTable::lookupOp(StringRef name) const {
  return lookupOp(StringAttr::get(innerSymTblOp->getContext(), name));
}
Operation *InnerSymbolTable::lookupOp(StringAttr name) const {
  auto result = lookup(name);
  if (result.isOpOnly())
    return result.getOp();
  return nullptr;
}

/// Get InnerSymbol for an operation.
StringAttr InnerSymbolTable::getInnerSymbol(Operation *op) {
  if (auto innerSymOp = dyn_cast<InnerSymbolOpInterface>(op))
    return innerSymOp.getInnerNameAttr();
  return {};
}

/// Get InnerSymbol for a target.  Be robust to queries on unexpected
/// operations to avoid users needing to know the details.
StringAttr InnerSymbolTable::getInnerSymbol(InnerSymTarget target) {
  // Assert on misuse, but try to handle queries otherwise.
  assert(target);

  InnerSymAttr base;
  if (target.isPort()) {
    auto mod = dyn_cast<FModuleLike>(target.getOp());
    if (!mod)
      return {};
    assert(target.getPort() < mod.getNumPorts());
    // TODO: update this when ports support per-field symbols
    // base = mod.getPortSymbolAttr(target.getPort());
    auto sym = mod.getPortSymbolAttr(target.getPort());
    // Workaround quirk with empty string for no symbol on ports.
    if (sym && sym.getValue().empty())
      return {};
    return sym;
  } else {
    // InnerSymbols only supported if op implements the interface.
    auto symOp = dyn_cast<InnerSymbolOpInterface>(target.getOp());
    if (!symOp)
      return {};
    base = symOp.getInnerSymAttr();
  }

  return base.getSymIfExists(target.getField());
}

/// Return an InnerRef to the given operation.
hw::InnerRefAttr InnerSymbolTable::getInnerRef(Operation *op) {
  assert(op->getParentWithTrait<OpTrait::InnerSymbolTable>() == innerSymTblOp);
  if (auto sym = getInnerSymbol(op))
    return hw::InnerRefAttr::get(SymbolTable::getSymbolName(innerSymTblOp),
                                 sym);
  return hw::InnerRefAttr();
}

hw::InnerRefAttr InnerSymbolTable::getInnerRef(InnerSymTarget target) {
  assert(target.isPort() && target.getOp() == innerSymTblOp ||
         !target.isPort() &&
             target.getOp()->getParentWithTrait<OpTrait::InnerSymbolTable>() ==
                 innerSymTblOp);
  if (auto sym = getInnerSymbol(target))
    return hw::InnerRefAttr::get(SymbolTable::getSymbolName(innerSymTblOp),
                                 sym);
  return hw::InnerRefAttr();
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

InnerSymTarget InnerRefNamespace::lookup(hw::InnerRefAttr inner) {
  auto *mod = symTable.lookup(inner.getModule());
  assert(mod->hasTrait<mlir::OpTrait::InnerSymbolTable>());
  return innerSymTables.getInnerSymbolTable(mod).lookup(inner.getName());
}

Operation *InnerRefNamespace::lookupOp(hw::InnerRefAttr inner) {
  auto *mod = symTable.lookup(inner.getModule());
  assert(mod->hasTrait<mlir::OpTrait::InnerSymbolTable>());
  return innerSymTables.getInnerSymbolTable(mod).lookupOp(inner.getName());
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
