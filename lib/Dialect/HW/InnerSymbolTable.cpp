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

#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Dialect/HW/HWOpInterfaces.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "innersymboltable"

using namespace circt;
using namespace hw;

namespace circt {
namespace hw {

//===----------------------------------------------------------------------===//
// InnerSymbolTable
//===----------------------------------------------------------------------===//

StringAttr InnerSymbolTable::getName(Operation *innerSymbolOp) {
  return llvm::TypeSwitch<Operation *, StringAttr>(innerSymbolOp)
      .Case<hw::InnerSymbolOpInterface>(
          [](auto innerSym) { return innerSym.getInnerNameAttr(); })
      .Case<mlir::SymbolOpInterface>(
          [](auto sym) { return SymbolTable::getSymbolName(sym); })
      .Default([](Operation *) -> StringAttr { return {}; });
}

InnerSymbolTable::InnerSymbolTable(Operation *op) {
  auto res = InnerSymbolTable::buildSymbolTable(*this, op);
  assert(succeeded(res) && "Expected successful symbol table construction");
}

LogicalResult InnerSymbolTable::buildSymbolTable(InnerSymbolTable &ist,
                                                 Operation *istOp) {
  if (!istOp->hasTrait<OpTrait::InnerSymbolTable>())
    return istOp->emitError(
        "expected operation to have InnerSymbolTable trait");

  // Caching of symbol table defining operations to their symbol tables.
  DenseMap<Operation *, InnerSymbolTable *> symTblCache;
  ist.innerSymTblOp = istOp;
  symTblCache[istOp] = &ist;

  auto symCallback = [&](StringAttr name, const InnerSymTarget &target,
                         Operation *currentIST) -> LogicalResult {
    // Lookup symbol table which this operation resides in.
    auto *currentSymTable = symTblCache.lookup(currentIST);
    if (!currentSymTable) {
      llvm::dbgs() << "IST: Target symbol table is: " << *currentIST << "\n";
      assert(currentSymTable && "Expected parent symbol table to be in cache");
    }

    // Insert the symbol into the current symbol table.
    auto it = currentSymTable->symbolTable.try_emplace(name, target);
    LLVM_DEBUG(llvm::dbgs() << "IST: Inserted symbol " << name.getValue()
                            << " -> " << target.getOp()->getName() << "\n";);

    if (it.second)
      return success();

    auto existing = it.first->second;
    return target.getOp()
        ->emitError()
        .append("redefinition of inner symbol named '", name.strref(), "'")
        .attachNote(existing.getOp()->getLoc())
        .append("see existing inner symbol definition here");
  };

  auto symTableCallback = [&](StringAttr symTblName,
                              Operation *currentSymbolTableOp,
                              Operation *symTblOp) -> LogicalResult {
    // Lookup symbol table which this operation resides in.
    auto *currentSymTable = symTblCache.lookup(currentSymbolTableOp);
    assert(currentSymTable && "Expected parent symbol table to be in cache");

    // Construct a new nested symbol table if this operation is an
    // InnerSymbolTable, and the target operation is not the same as the
    // symbol table of the target operation (symbol-table defining op
    // defines symbols within its own table).
    std::unique_ptr<InnerSymbolTable> nestedSymTbl;
    nestedSymTbl = std::unique_ptr<InnerSymbolTable>(new InnerSymbolTable());
    nestedSymTbl->innerSymTblOp = symTblOp;
    auto it = currentSymTable->nestedSymbolTables.insert(
        {InnerSymbolTable::getName(symTblOp), std::move(nestedSymTbl)});
    auto nestedIt = symTblCache.try_emplace(symTblOp, it.first->second.get());
    assert(nestedIt.second && "Expected nested symbol table to be new");
    LLVM_DEBUG(llvm::dbgs() << "IST: Created nested symbol table for "
                            << symTblOp->getName() << "\n";);
    return success();
  };

  return walkSymbols(istOp, symCallback, symTableCallback);
}

FailureOr<InnerSymbolTable> InnerSymbolTable::get(Operation *istOp) {
  InnerSymbolTable table;
  if (failed(table.buildSymbolTable(table, istOp)))
    return failure();

  return table;
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult InnerSymbolTable::walkSymbolsInner(
    Operation *op, InnerSymCallbackFn symCallback,
    std::optional<InnerSymTableCallbackFn> symTblCallback,
    Operation *currentIST) {

  // Swap/set currently active IST if this op is an IST.
  if (op->hasTrait<OpTrait::InnerSymbolTable>())
    currentIST = op;

  auto walkSym = [&](StringAttr name, const InnerSymTarget &target) {
    assert(name && !name.getValue().empty());
    return symCallback(name, target, currentIST);
  };

  auto walkTable = [&](StringAttr name, Operation *symTblOp) {
    // Call the special symbol table callback, if provided.
    if (symTblCallback)
      if (failed(symTblCallback.value()(name, currentIST, symTblOp)))
        return failure();
    return success();
  };

  auto walkSyms = [&](hw::InnerSymAttr symAttr,
                      const InnerSymTarget &baseTarget) -> LogicalResult {
    assert(baseTarget.getField() == 0);
    for (auto symProp : symAttr) {
      if (failed(walkSym(symProp.getName(),
                         InnerSymTarget::getTargetForSubfield(
                             baseTarget, symProp.getFieldID()))))
        return failure();
    }
    return success();
  };

  // Walk the immediate level of this operation and add InnerSymbolTarget's to
  // the table. If an operation is itself an InnerSymbolTable, create the symbol
  // table(s) after all other operations have been walked.
  auto walkPorts = [&](Operation *op) {
    // Check for ports
    if (auto mod = dyn_cast<PortList>(op)) {
      for (auto [i, port] : llvm::enumerate(mod.getPortList())) {
        if (auto symAttr = port.getSym()) {
          if (failed(walkSyms(symAttr, InnerSymTarget(i, op, 0))))
            return failure();
        }
      }
    }

    return success();
  };

  // First, check the operation itself for ports. This is a separate step
  // since we want ports to be listed in the InnerSymbolTable of the operation
  // itself, and not in the parent InnerSymbolTable.
  if (failed(walkPorts(op)))
    return failure();

  // Perform the walk. We do not use Operation::walk here, since recursing
  // into regions must go through InnerSymbolTable::walkSymbolsInner to properly
  // track nested symbol tables.
  for (auto &region : op->getRegions()) {
    for (auto &block : region.getBlocks()) {
      for (auto &innerOp : block) {
        // Check for nested symbol table.
        if (innerOp.hasTrait<OpTrait::InnerSymbolTable>()) {
          if (failed(walkTable(InnerSymbolTable::getName(&innerOp), &innerOp)))
            return failure();
        }

        // Check for InnerSymbolOpInterface
        if (auto symOp = dyn_cast<InnerSymbolOpInterface>(&innerOp))
          if (auto symAttr = symOp.getInnerSymAttr())
            if (failed(walkSyms(symAttr, InnerSymTarget(symOp, 0))))
              return failure();

        // Check for port-attached symbols.
        if (failed(walkPorts(&innerOp)))
          return failure();

        // Recurse into the operation.
        if (failed(walkSymbolsInner(&innerOp, symCallback, symTblCallback,
                                    currentIST)))
          return failure();
      }
    }
  }

  return success();
}

LogicalResult InnerSymbolTable::walkSymbols(
    Operation *op, InnerSymCallbackFn symCallback,
    std::optional<InnerSymTableCallbackFn> symTblCallback) {
  return walkSymbolsInner(op, symCallback, symTblCallback,
                          /*currentSymbolTable*/ nullptr);
}

InnerSymTarget InnerSymbolTable::lookup(hw::InnerRefAttr path) const {
  return lookup(path.getPath());
}

InnerSymTarget InnerSymbolTable::lookup(llvm::ArrayRef<StringAttr> path) const {
  assert(!path.empty() && "Expected non-empty path");
  assert(llvm::all_of(path, [](auto &attr) { return attr; }) &&
         "Expected non-null path elements");

  if (path.size() == 1) {
    // This was the last element in the path, lookup the symbol.
    return lookup(path.front());
  }

  // Else, we have a path with more than one element; this implies that this
  // target should be a nested symbol table.
  auto nestedSymTbl = nestedSymbolTables.find(path.front());
  if (nestedSymTbl == nestedSymbolTables.end())
    return {};

  // Perform the internal lookup into the nested symbol table.
  return nestedSymTbl->second->lookup(path.drop_front());
}

void InnerSymbolTable::dump(llvm::raw_ostream &os, int indent) const {
  os.indent(indent) << "@"
                    << InnerSymbolTable::getName(getInnerSymTblOp()).getValue()
                    << " symbol table {\n";
  indent += 2;
  for (auto &symbol : symbolTable) {
    os.indent(indent) << "@" << symbol.first.getValue() << " -> "
                      << symbol.second;
    if (auto nestedSymTblIt = nestedSymbolTables.find(symbol.first);
        nestedSymTblIt != nestedSymbolTables.end()) {
      os << "\n";
      nestedSymTblIt->second->dump(os, indent);
    } else {
      os << "\n";
    }
  }
  indent -= 2;
  os.indent(indent) << "}\n";
}

/// Look up a symbol with the specified name, returning empty InnerSymTarget if
/// no such name exists. Names never include the @ on them.
InnerSymTarget InnerSymbolTable::lookup(StringRef name) const {
  return lookup(StringAttr::get(innerSymTblOp->getContext(), name));
}

InnerSymTarget InnerSymbolTable::lookup(StringAttr name) const {
  auto it = symbolTable.find(name);
  if (it == symbolTable.end())
    return {};
  return it->second;
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

StringAttr InnerSymbolTable::getISTSymbol(Operation *op) {
  assert(op->hasTrait<OpTrait::InnerSymbolTable>() &&
         "Expected operation to have InnerSymbolTable trait");
  if (auto innerSym = getInnerSymbol(op))
    return innerSym;

  // If the operation is not an InnerSymbolOp, it must be a SymbolOp.
  if (auto sym = dyn_cast<mlir::SymbolOpInterface>(op))
    return SymbolTable::getSymbolName(sym);

  llvm_unreachable(
      "InnerSymbolTable trait should have verified that this can't happen");
}

/// Get InnerSymbol for a target.  Be robust to queries on unexpected
/// operations to avoid users needing to know the details.
StringAttr InnerSymbolTable::getInnerSymbol(const InnerSymTarget &target) {
  // Assert on misuse, but try to handle queries otherwise.
  assert(target);

  // Obtain the base InnerSymAttr for the specified target.
  auto getBase = [](auto &target) -> hw::InnerSymAttr {
    if (target.isPort()) {
      if (auto mod = dyn_cast<PortList>(target.getOp())) {
        assert(target.getPort() < mod.getNumPorts());
        return mod.getPort(target.getPort()).getSym();
      }
    } else {
      // InnerSymbols only supported if op implements the interface.
      if (auto symOp = dyn_cast<InnerSymbolOpInterface>(target.getOp()))
        return symOp.getInnerSymAttr();
    }
    return {};
  };

  if (auto base = getBase(target))
    return base.getSymIfExists(target.getField());
  return {};
}

//===----------------------------------------------------------------------===//
// InnerSymbolTableCollection
//===----------------------------------------------------------------------===//

namespace detail {

/// This class represents a collection of InnerSymbolTable's.
struct InnerSymbolTableCollection {
  /// Get or create the InnerSymbolTable for the specified operation.
  InnerSymbolTable &getInnerSymbolTable(Operation *op);

  /// Populate tables in parallel for all top-level InnerSymbolTable operations
  /// in the given InnerRefNamespace operation, verifying each and returning the
  /// verification result.
  LogicalResult populateAndVerifyTables(Operation *innerRefNSOp);

  explicit InnerSymbolTableCollection() = default;
  explicit InnerSymbolTableCollection(Operation *innerRefNSOp) {
    // Caller is not interested in verification, no way to report it upwards.
    auto result = populateAndVerifyTables(innerRefNSOp);
    (void)result;
    assert(succeeded(result));
  }
  InnerSymbolTableCollection(const InnerSymbolTableCollection &) = delete;
  InnerSymbolTableCollection &
  operator=(const InnerSymbolTableCollection &) = delete;

  /// Dumps this InnerSymbolTableCollection to the provided output stream.
  void dump(llvm::raw_ostream &os) const;

  /// Top-level symbol tables
  DenseMap<Operation *, std::unique_ptr<InnerSymbolTable>> topLevelSymbolTables;

  /// Mapping between top-level symbol (table) names to the top-level symbol
  /// tables.
  DenseMap<StringAttr, Operation *> topLevelSymbolTableMapping;
};

void InnerSymbolTableCollection::dump(llvm::raw_ostream &os) const {
  os << "IST: Symbol table collection:\n";
  for (auto &[_, symTable] : topLevelSymbolTables) {
    symTable->dump(os);
    os << "\n";
  }
  os << "\n";
}

LogicalResult
InnerSymbolTableCollection::populateAndVerifyTables(Operation *innerRefNSOp) {
  // Gather top-level operations that have the InnerSymbolTable trait.
  SmallVector<Operation *> innerSymTableOps(llvm::make_filter_range(
      llvm::make_pointer_range(innerRefNSOp->getRegion(0).front()),
      [&](Operation *op) {
        return op->hasTrait<OpTrait::InnerSymbolTable>();
      }));

  // Prime the top-level symbol tables.
  bool scanFailed = false;
  llvm::for_each(innerSymTableOps, [&](auto *op) {
    // InnerSymbolTable's must be either an InnerSymbolOp or a SymbolOp.
    StringAttr symTableName = InnerSymbolTable::getISTSymbol(op);
    topLevelSymbolTables.try_emplace(op, nullptr);
    auto it = topLevelSymbolTableMapping.try_emplace(symTableName, op);
    if (!it.second) {
      auto diag = op->emitError()
                      .append("redefinition of top-level symbol '")
                      .append(symTableName.getValue())
                      .append("'");
      diag.attachNote(it.first->second->getLoc())
          .append("see existing top-level symbol definition here");
      scanFailed = true;
    }
  });

  if (scanFailed)
    return failure();

  // Construct the top-tables in parallel (if context allows it).
  auto res = mlir::failableParallelForEach(
      innerRefNSOp->getContext(), innerSymTableOps, [&](auto *op) {
        auto it = topLevelSymbolTables.find(op);
        assert(it != topLevelSymbolTables.end());
        if (!it->second) {
          auto result = InnerSymbolTable::get(op);
          if (failed(result))
            return failure();
          it->second = std::make_unique<InnerSymbolTable>(std::move(*result));
          return success();
        }
        return failure();
      });

  if (failed(res))
    return failure();

  LLVM_DEBUG(dump(llvm::dbgs()););

  return success();
}

} // namespace detail

//===----------------------------------------------------------------------===//
// InnerRefNamespace
//===----------------------------------------------------------------------===//

InnerSymTarget InnerRefNamespace::lookup(hw::InnerRefAttr inner) const {
  // Root reference - lookup from the top-level scope. The first reference
  // in the inner reference path is the top-level symbol table name.
  auto *rootOp =
      innerSymTables->topLevelSymbolTableMapping.lookup(inner.getRoot());
  if (!rootOp)
    return InnerSymTarget();
  assert(rootOp->hasTrait<mlir::OpTrait::InnerSymbolTable>());

  // This was a root reference to a top-level symbol table.
  if (inner.getPath().size() == 1)
    return InnerSymTarget(rootOp);

  // Else, this is a nested reference. Lookup the target in the root symbol
  // table.
  return innerSymTables->topLevelSymbolTables.at(rootOp)->lookup(
      llvm::ArrayRef(inner.getPath()).drop_front());
}

Operation *InnerRefNamespace::lookupOp(hw::InnerRefAttr inner) const {
  auto target = lookup(inner);
  if (!target)
    return nullptr;

  return target.getOp();
}

/// Dump this namespace to the provided output stream.
void InnerRefNamespace::dump(llvm::raw_ostream &os) const {
  innerSymTables->dump(os);
}

InnerRefNamespace::InnerRefNamespace(Operation *op) {
  innerSymTables = std::make_unique<detail::InnerSymbolTableCollection>(op);
}

/// Construct an InnerRefNamespace, checking for verification failure.
/// Emits diagnostics describing encountered issues.
FailureOr<InnerRefNamespace> InnerRefNamespace::get(Operation *op) {
  InnerRefNamespace irn;
  irn.innerSymTables = std::make_unique<detail::InnerSymbolTableCollection>();
  if (failed(irn.innerSymTables->populateAndVerifyTables(op)))
    return failure();

  return FailureOr<InnerRefNamespace>{std::move(irn)};
}

InnerRefNamespace::~InnerRefNamespace() = default;
InnerRefNamespace::InnerRefNamespace() = default;
//===----------------------------------------------------------------------===//
// InnerRefNamespace verification
//===----------------------------------------------------------------------===//

namespace detail {

LogicalResult verifyInnerRefNamespace(Operation *op) {
  // Construct the InnerRefNamespace
  auto irn = InnerRefNamespace::get(op);
  if (failed(irn))
    return failure();

  // Conduct parallel walks of the top-level children of this
  // InnerRefNamespace, verifying all InnerRefUserOp's discovered within.
  auto verifySymbolUserFn = [&](Operation *op) -> WalkResult {
    if (auto user = dyn_cast<InnerRefUserOpInterface>(op))
      return WalkResult(user.verifyInnerRefs(*irn));
    return WalkResult::advance();
  };

  SmallVector<Operation *> topLevelOps;
  for (auto &op : op->getRegion(0).front()) {
    // Gather operations with regions for parallel processing.
    if (op.getNumRegions() != 0) {
      topLevelOps.push_back(&op);
      continue;
    }
    // Otherwise, handle right now -- not worth the cost.
    if (verifySymbolUserFn(&op).wasInterrupted())
      return failure();
  }
  return mlir::failableParallelForEach(
      op->getContext(), topLevelOps, [&](Operation *op) {
        return success(!op->walk(verifySymbolUserFn).wasInterrupted());
      });
}

} // namespace detail

bool InnerRefNamespaceLike::classof(mlir::Operation *op) {
  return op->hasTrait<mlir::OpTrait::InnerRefNamespace>() ||
         op->hasTrait<mlir::OpTrait::SymbolTable>();
}

bool InnerRefNamespaceLike::classof(
    const mlir::RegisteredOperationName *opInfo) {
  return opInfo->hasTrait<mlir::OpTrait::InnerRefNamespace>() ||
         opInfo->hasTrait<mlir::OpTrait::SymbolTable>();
}

bool InnerSymbolTableLike::classof(mlir::Operation *op) {
  return op->hasTrait<mlir::OpTrait::InnerSymbolTable>() ||
         op->hasTrait<mlir::OpTrait::SymbolTable>();
}

bool InnerSymbolTableLike::classof(
    const mlir::RegisteredOperationName *opInfo) {
  return opInfo->hasTrait<mlir::OpTrait::InnerSymbolTable>();
}

} // namespace hw
} // namespace circt
