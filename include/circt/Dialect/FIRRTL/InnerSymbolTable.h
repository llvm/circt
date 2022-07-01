//===- InnerSymbolTable.h - Inner Symbol Table -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the InnerSymbolTable and related classes, used for
// managing and tracking "inner symbols".
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_INNERSYMBOLTABLE_H
#define CIRCT_DIALECT_FIRRTL_INNERSYMBOLTABLE_H

#include "circt/Dialect/FIRRTL/FIRRTLAttributes.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace firrtl {

/// A table of inner symbols and their resolutions.
class InnerSymbolTable {
public:
  /// Build an inner symbol table for the given operation.  The operation must
  /// have the InnerSymbolTable trait.
  explicit InnerSymbolTable(Operation *op);

  /// Non-copyable
  InnerSymbolTable(const InnerSymbolTable &) = delete;
  InnerSymbolTable &operator=(InnerSymbolTable &) = delete;

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Names never include the @ on them.
  Operation *lookup(StringRef name) const;
  template <typename T>
  T lookup(StringRef name) const {
    return dyn_cast_or_null<T>(lookup(name));
  }

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists. Names never include the @ on them.
  Operation *lookup(StringAttr name) const;
  template <typename T>
  T lookup(StringAttr name) const {
    return dyn_cast_or_null<T>(lookup(name));
  }

  /// Return an InnerRef to the given operation which must be within this table.
  hw::InnerRefAttr getInnerRef(Operation *op);

  /// Return an InnerRef for the given inner symbol, which must be valid.
  hw::InnerRefAttr getInnerRef(StringRef name) {
    return getInnerRef(lookup(name));
  }

  /// Return an InnerRef for the given inner symbol, which must be valid.
  hw::InnerRefAttr getInnerRef(StringAttr name) {
    return getInnerRef(lookup(name));
  }

  /// Get InnerSymbol for an operation.
  static StringAttr getInnerSymbol(Operation *op);

  /// Return the name of the attribute used for inner symbol names.
  static StringRef getInnerSymbolAttrName() { return "inner_sym"; }

private:
  /// This is the operation this table is constructed for, which must have the
  /// InnerSymbolTable trait.
  Operation *innerSymTblOp;

  /// This maps names to operations with that inner symbol.
  DenseMap<StringAttr, Operation *> symbolTable;
};

/// This class represents a collection of InnerSymbolTable's.
class InnerSymbolTableCollection {
public:
  /// Get or create the InnerSymbolTable for the specified operation.
  InnerSymbolTable &getInnerSymbolTable(Operation *op);

  /// Populate tables in parallel for all InnerSymbolTable operations in the
  /// given InnerRefNamespace operation.
  void populateTables(Operation *innerRefNSOp);

  explicit InnerSymbolTableCollection() = default;
  InnerSymbolTableCollection(const InnerSymbolTableCollection &) = delete;
  InnerSymbolTableCollection &operator=(InnerSymbolTableCollection &) = delete;

private:
  /// This maps Operations to their InnnerSymbolTable's.
  DenseMap<Operation *, std::unique_ptr<InnerSymbolTable>> symbolTables;
};

/// This class represents the namespace in which InnerRef's can be resolved.
struct InnerRefNamespace {
  SymbolTable &symTable;
  InnerSymbolTableCollection &innerSymTables;

  /// Resolve the InnerRef to its target within this namespace, returning null
  /// if no such name exists.
  ///
  /// Note that some InnerRef's target ports and must be handled separately.
  Operation *lookup(hw::InnerRefAttr inner);
  template <typename T>
  T lookup(hw::InnerRefAttr inner) {
    return dyn_cast_or_null<T>(lookup(inner));
  }
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_INNERSYMBOLTABLE_H
