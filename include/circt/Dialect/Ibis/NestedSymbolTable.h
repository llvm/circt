//===- NestedSymbolTable.h - Inner Symbol Table -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the NestedSymbolTable and related classes, used for
// managing and tracking "nested symbols".
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_IBIS_NESTEDSYMBOLTABLE_H
#define CIRCT_DIALECT_IBIS_NESTEDSYMBOLTABLE_H

#include "circt/Dialect/Ibis/IbisAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace ibis {

class NestedSymbolTable {
public:
  /// Build an inner symbol table for the given operation. The operation must
  /// have the NestedSymbolTable trait.
  explicit NestedSymbolTable(Operation *op);

  /// Non-copyable
  InnerSymbolTable(const InnerSymbolTable &) = delete;
  InnerSymbolTable &operator=(const InnerSymbolTable &) = delete;

  // Moveable
  InnerSymbolTable(InnerSymbolTable &&) = default;
  InnerSymbolTable &operator=(InnerSymbolTable &&) = default;

  /// Look up a symbol with the specified path and name, returning nullptr if no
  /// such operation exists. Names never include the @ on them.
  /// Lookup is performed from the current node towards the leaf symbols and
  /// symbol tables.
  Operation *lookup(ArrayRef<StringAttr> path, StringRef name) const;

  template <typename T>
  Operation *lookupOp(ArrayRef<StringAttr> path, StringRef name) const {
    return dyn_cast_or_null<T>(lookup(path, name));
  }

  /// Look up a symbol with the specified name, returning nullptr if no such
  /// operation exists. Names never include the @ on them.
  /// The operation is expected to be nested directly within this symbol table.
  Operation *lookup(StringRef name) const;
  template <typename T>
  Operation *lookupOp(StringRef name) const {
    return dyn_cast_or_null<T>(lookup(name));
  }

private:
  /// This maps symbol names to operations. If the operation itself is a nested
  /// symbol table, it is stored in the second element of the pair.
  llvm::DenseMap<StringAttr,
                 std::pair<Operation *, std::shared_ptr<NestedSymbolTable>>>
      symbolTable;

  /// This is the operation this table is constructed for, which must have the
  /// NestedSymbolTable trait.
  Operation *nestedSymTblOp;
};

class NestedSymbolNamespace : public NestedSymbolTable {
public:
  /// Look up a symbol with the specified path and name, returning nullptr if no
  /// such operation exists. Names never include the @ on them.
  /// Symbol lookup follows C++-like namespacing.
  /// E.g., if a symbol lookup of A::B::C is requested, the lookup will first
  /// look for symbol "A" in the current symbol table. If "A" was found, lookups
  /// of B and C will be performed in the symbol table of A, and must not
  /// traverse upwards in the symbol table hierarchy.
  /// If 'A' was not found, the lookup will traverse up in the symbol table
  /// hierarchy (i.e. the parent symbol table of 'A') and look for 'A' there.
  /// This occurs recursively, until the root symbol table is reached.
  Operation *lookup(Operation *from, ArrayRef<StringAttr> path,
                    StringRef name) const;

  template <typename T>
  Operation *lookupOp(Operation *from, ArrayRef<StringAttr> path,
                      StringRef name) const {
    return dyn_cast_or_null<T>(lookup(from, path, name));
  }

  /// Look up a symbol with the specified name, returning nullptr if no such
  /// operation exists. Names never include the @ on them.
  /// The operation is expected to be nested directly within this symbol table.
  Operation *lookup(Operation *from, StringRef name) const;
  template <typename T>
  Operation *lookupOp(Operation *from, StringRef name) const {
    return dyn_cast_or_null<T>(lookup(from, name));
  }

  /// Return the symbol table for the given operation. The provided operation
  /// must implement the NestedSymbolTable trait.
  NestedSymbolTable &getSymbolTable(Operation *op) {
    auto it = symbolTables.find(op);
    if (it == symbolTables.end())
      return nullptr;
    return *it->second.get();
  }

private:
  // This maps operations to their symbol tables. Used to be able to look up
  // any symbol table from any operation.
  llvm::DenseMap<Operation *, std::shared_ptr<NestedSymbolTable>> symbolTables;
};

} // namespace ibis
} // namespace circt

#endif // CIRCT_DIALECT_IBIS_NESTEDSYMBOLTABLE_H