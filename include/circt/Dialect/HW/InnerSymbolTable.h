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

#ifndef CIRCT_DIALECT_HW_INNERSYMBOLTABLE_H
#define CIRCT_DIALECT_HW_INNERSYMBOLTABLE_H

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/StringRef.h"

namespace circt {
namespace hw {

namespace detail {
struct InnerSymbolTableCollection;
}

/// The target of an inner symbol, the entity the symbol is a handle for.
class InnerSymTarget {
public:
  /// Default constructor, invalid.
  InnerSymTarget() { assert(!*this); }

  /// Target an operation.
  explicit InnerSymTarget(Operation *op) : InnerSymTarget(op, 0) {}

  /// Target an operation and a field (=0 means the op itself).
  InnerSymTarget(Operation *op, size_t fieldID)
      : op(op), portIdx(invalidPort), fieldID(fieldID) {}

  /// Target a port, and optionally a field (=0 means the port itself).
  InnerSymTarget(size_t portIdx, Operation *op, size_t fieldID = 0)
      : op(op), portIdx(portIdx), fieldID(fieldID) {}

  InnerSymTarget(const InnerSymTarget &) = default;
  InnerSymTarget(InnerSymTarget &&) = default;

  // Accessors:

  /// Return the target's fieldID.
  auto getField() const { return fieldID; }

  /// Return the target's base operation.  For ports, this is the module.
  Operation *getOp() const { return op; }

  /// Return the target's port, if valid.  Check "isPort()".
  auto getPort() const {
    assert(isPort());
    return portIdx;
  }

  // Classification:

  /// Return if this targets a field (nonzero fieldID).
  bool isField() const { return fieldID != 0; }

  /// Return if this targets a port.
  bool isPort() const { return portIdx != invalidPort; }

  /// Returns if this targets an operation only (not port or field).
  bool isOpOnly() const { return !isPort() && !isField(); }

  /// Return a target to the specified field within the given base.
  /// FieldID is relative to the specified base target.
  static InnerSymTarget getTargetForSubfield(const InnerSymTarget &base,
                                             size_t fieldID) {
    if (base.isPort())
      return InnerSymTarget(base.portIdx, base.op, base.fieldID + fieldID);
    return InnerSymTarget(base.op, base.fieldID + fieldID);
  }

private:
  auto asTuple() const { return std::tie(op, portIdx, fieldID); }
  Operation *symbolTableOp = nullptr;
  Operation *op = nullptr;
  size_t portIdx = 0;
  size_t fieldID = 0;
  static constexpr size_t invalidPort = ~size_t{0};

public:
  // Operators are defined below.

  // Comparison operators:
  bool operator==(const InnerSymTarget &rhs) const {
    return asTuple() == rhs.asTuple();
  }

  // Assignment operators:
  InnerSymTarget &operator=(InnerSymTarget &&) = default;
  InnerSymTarget &operator=(const InnerSymTarget &) = default;

  /// Check if this target is valid.
  operator bool() const { return op; }
};

/// A table of inner symbols and their resolutions.
class InnerSymbolTable {

public:
  /// Construct an InnerSymbolTable, checking for verification failure.
  /// Emits diagnostics describing encountered issues.
  static FailureOr<InnerSymbolTable> get(Operation *op);

  /// Non-copyable
  InnerSymbolTable(const InnerSymbolTable &) = delete;
  InnerSymbolTable &operator=(const InnerSymbolTable &) = delete;

  // Moveable
  InnerSymbolTable(InnerSymbolTable &&) = default;
  InnerSymbolTable &operator=(InnerSymbolTable &&) = default;

  /// Look up a symbol with the specified name, returning empty InnerSymTarget
  /// if no such name exists. Names never include the @ on them.
  InnerSymTarget lookup(StringRef name) const;
  InnerSymTarget lookup(StringAttr name) const;

  /// Lookup an InnerRefAttr path, returning empty InnerSymTarget if no such
  /// name exists.
  InnerSymTarget lookup(hw::InnerRefAttr path) const;
  InnerSymTarget lookup(llvm::ArrayRef<StringAttr> path) const;

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists or doesn't target just an operation.
  Operation *lookupOp(StringRef name) const;
  template <typename T>
  T lookupOp(StringRef name) const {
    return dyn_cast_or_null<T>(lookupOp(name));
  }

  /// Look up a symbol with the specified name, returning null if no such
  /// name exists or doesn't target just an operation.
  Operation *lookupOp(StringAttr name) const;
  template <typename T>
  T lookupOp(StringAttr name) const {
    return dyn_cast_or_null<T>(lookupOp(name));
  }

  /// Returns the name of an InnerSymbol operation.
  static StringAttr getName(Operation *innerSymbolOp);

  /// Return a pointer to the symbol table which this table is nested within.
  /// If this is a top-level symbol table (nested in the InnerRefNamespace),
  /// returns nullptr.
  InnerSymbolTable *getParentSymbolTable() const;

  /// Dumps this symbol table to the provided output stream.
  void dump(llvm::raw_ostream &os, int indent = 0) const;

  /// Return the operation this symbol table is associated with.
  Operation *getInnerSymTblOp() const { return innerSymTblOp; }

  /// Get InnerSymbol for an operation.
  static StringAttr getInnerSymbol(Operation *op);

  /// Get InnerSymbol for a target.
  static StringAttr getInnerSymbol(const InnerSymTarget &target);

  /// Get the symbol name for an InnerSymbolTable operation. These operations
  /// may define symbols as either builtin MLIR symbols or inner_sym's.
  static StringAttr getISTSymbol(Operation *op);

  /// Return the name of the attribute used for inner symbol names.
  static StringRef getInnerSymbolAttrName() { return "inner_sym"; }

  using InnerSymCallbackFn =
      llvm::function_ref<LogicalResult(StringAttr, const InnerSymTarget &,
                                       /*currentSymbolTableOp*/ Operation *)>;

  /// Callback type for walking nested symbol tables. The `currentSymbolTable`
  /// operation is provided as an Operation* value. A `InnerSymbolTable*` is
  /// not provided, since this callback is also used in
  /// `static InnerSymbolTable::walkSymbols`, which doesn't guarantee
  /// construction of the symbol table.
  using InnerSymTableCallbackFn = llvm::function_ref<LogicalResult(
      StringAttr, /*currentSymbolTableOp*/ Operation *,
      /*thisSymbolTable*/ Operation *)>;

  /// Walk the given IST operation and invoke the callback for all encountered
  /// inner symbols and symbol tables.
  /// This variant allows callbacks that return LogicalResult OR void,
  /// and wraps the underlying implementation.
  template <typename FuncTy,
            typename RetTy = typename std::invoke_result_t<
                FuncTy, StringAttr, const InnerSymTarget &, Operation *>>
  static RetTy walkSymbols(
      Operation *op, FuncTy &&symCallback,
      std::optional<InnerSymTableCallbackFn> symTblCallback = std::nullopt) {
    if constexpr (std::is_void_v<RetTy>)
      return (void)walkSymbols(
          op,
          InnerSymCallbackFn([&](StringAttr name, const InnerSymTarget &target,
                                 Operation *currentIST) {
            std::invoke(std::forward<FuncTy>(symCallback), name, target,
                        currentIST);
            return success();
          }),
          symTblCallback);
    else
      return walkSymbols(
          op,
          InnerSymCallbackFn([&](StringAttr name, const InnerSymTarget &target,
                                 Operation *currentIST) {
            return std::invoke(std::forward<FuncTy>(symCallback), name, target,
                               currentIST);
          }),
          symTblCallback);
  }

  /// Walk the given IST operation and invoke the symCallback for all
  /// encountered inner symbols and inner symbol tables. This variant is the
  /// underlying implementation. If callback returns failure, the walk is
  /// aborted and failure is returned. A successful walk with no failures
  /// returns success.
  /// This function guarantees that symTblCallback is called before any
  /// symCallback call on operations that reside within a given symbol table.
  static LogicalResult walkSymbols(
      Operation *op, InnerSymCallbackFn symCallback,
      std::optional<InnerSymTableCallbackFn> symTblCallback = std::nullopt);

private:
  /// Construct an empty inner symbol table; used by InnerSymbolTable::get to
  /// avoid calling the InnerSymbolTable(Operation*) constructor.
  explicit InnerSymbolTable(){};

  /// Like walkSymbols, but also carries the currently active InnerSymbolTable
  /// operation as an argument.
  static LogicalResult
  walkSymbolsInner(Operation *op, InnerSymCallbackFn symCallback,
                   std::optional<InnerSymTableCallbackFn> symTblCallback,
                   Operation *currentIST);

  using SymbolTableTy = DenseMap<StringAttr, InnerSymTarget>;
  using NestedSymbolTableTy =
      DenseMap<StringAttr, std::unique_ptr<InnerSymbolTable>>;

  /// Walk the symbols of the symbol table op and populate the symbol table.
  static LogicalResult buildSymbolTable(InnerSymbolTable &ist,
                                        Operation *istOp);

  /// Build an inner symbol table for the given operation.  The operation must
  /// have the InnerSymbolTable trait.
  explicit InnerSymbolTable(Operation *op);

  /// Construct an inner symbol table for the given operation,
  /// with pre-populated table contents.
  explicit InnerSymbolTable(Operation *op, SymbolTableTy &&table)
      : innerSymTblOp(op), symbolTable(table){};

  /// This is the operation this table is constructed for, which must have the
  /// InnerSymbolTable trait.
  Operation *innerSymTblOp;

  /// This maps inner symbol names to their targets.
  SymbolTableTy symbolTable;

  /// This maps inner symbol names to nested symbol tables.
  NestedSymbolTableTy nestedSymbolTables;
};

/// This class represents the namespace in which InnerRef's can be resolved.
struct InnerRefNamespace {
  /// Construct an InnerRefNamespace from the given operation.
  explicit InnerRefNamespace(Operation *op);
  ~InnerRefNamespace();

  /// Non-copyable
  InnerRefNamespace(const InnerRefNamespace &) = delete;
  InnerRefNamespace &operator=(const InnerRefNamespace &) = delete;

  // Moveable
  InnerRefNamespace(InnerRefNamespace &&) = default;
  InnerRefNamespace &operator=(InnerRefNamespace &&) = default;

  /// Construct an InnerRefNamespace, checking for verification failure.
  /// Emits diagnostics describing encountered issues.
  static FailureOr<InnerRefNamespace> get(Operation *op);

  /// Resolve the InnerRef to its target within this namespace, returning
  /// empty target if no such name exists.
  InnerSymTarget lookup(hw::InnerRefAttr inner) const;

  /// Dump this namespace to the provided output stream.
  void dump(llvm::raw_ostream &os) const;

  /// Resolve the InnerRef to its target within this namespace, returning
  /// empty target if no such name exists or it's not an operation.
  /// Template type can be used to limit results to specified op type.
  Operation *lookupOp(hw::InnerRefAttr inner) const;
  template <typename T>
  T lookupOp(hw::InnerRefAttr inner) const {
    return dyn_cast_or_null<T>(lookupOp(inner));
  }

private:
  /// Private constructor to be used for verification purposes.
  InnerRefNamespace();

  /// Collection of InnerSymbolTable's for all top-level InnerSymbolTable
  /// operations in the namespace.
  std::unique_ptr<detail::InnerSymbolTableCollection> innerSymTables;
};

/// Printing InnerSymTarget's.
template <typename OS>
OS &operator<<(OS &os, const InnerSymTarget &target) {
  if (!target)
    return os << "<invalid target>";

  if (target.isField())
    os << "field " << target.getField() << " of ";

  if (target.isPort())
    os << "port " << target.getPort() << " on ";
  else
    os << "op ";

  if (auto symName = InnerSymbolTable::getName(target.getOp()))
    os << "@" << symName.getValue() << "";
  else
    os << *target.getOp();

  return os;
}

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_INNERSYMBOLTABLE_H
