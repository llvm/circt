//===- Namespace.h - A symbol table for FIRRTL ops --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the FIRRTL CircuitNamespace, a symbol table for
// `firrtl.circuit` operations that allows for name collision resolution.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_NAMESPACE_H
#define CIRCT_DIALECT_FIRRTL_NAMESPACE_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "llvm/ADT/StringSet.h"

namespace circt {
namespace firrtl {

/// A namespace that is used to store existing names and generate new names in
/// some scope within the IR. This exists to work around limitations of
/// SymbolTables. This acts as a base class providing facilities common to all
/// namespaces implementations.
class NamespaceBase {
public:
  NamespaceBase() {}
  NamespaceBase(const NamespaceBase &other) = default;
  NamespaceBase(NamespaceBase &&other) : internal(std::move(other.internal)) {}

  NamespaceBase &operator=(const NamespaceBase &other) = default;
  NamespaceBase &operator=(NamespaceBase &&other) {
    internal = std::move(other.internal);
    return *this;
  }

  /// Empty the namespace.
  void clear() { internal.clear(); }

  /// Return a unique name, derived from the input `name`, and add the new name
  /// to the internal namespace.  There are two possible outcomes for the
  /// returned name:
  ///
  /// 1. The original name is returned.
  /// 2. The name is given a `_<n>` suffix where `<n>` is a number starting from
  ///    `_0` and incrementing by one each time.
  StringRef newName(const Twine &name) {
    // Special case the situation where there is no name collision to avoid
    // messing with the SmallString allocation below.
    llvm::SmallString<64> tryName;
    auto inserted = internal.insert(name.toStringRef(tryName));
    if (inserted.second)
      return inserted.first->getKey();

    // Try different suffixes until we get a collision-free one.
    size_t i = 0;
    if (tryName.empty())
      name.toVector(tryName); // toStringRef may leave tryName unfilled
    tryName.push_back('_');
    size_t baseLength = tryName.size();
    for (;;) {
      tryName.resize(baseLength);
      Twine(i++).toVector(tryName); // append integer to tryName
      auto inserted = internal.insert(tryName);
      if (inserted.second)
        return inserted.first->getKey();
    }
  }

protected:
  llvm::StringSet<> internal;
};

/// The namespace of a `CircuitOp`, generally inhabited by modules.
struct CircuitNamespace : public NamespaceBase {
  CircuitNamespace() {}
  CircuitNamespace(CircuitOp circuit) { add(circuit); }

  /// Populate the namespace from a circuit operation. This namespace will be
  /// composed of any operation in the first level of the circuit that contains
  /// a symbol.
  void add(CircuitOp circuit) {
    for (auto &op : *circuit.getBody())
      if (auto symbol = op.getAttrOfType<mlir::StringAttr>(
              SymbolTable::getSymbolAttrName()))
        internal.insert(symbol.getValue());
  }
};

/// The namespace of a `FModuleLike` operation, generally inhabited by its ports
/// and declarations.
struct ModuleNamespace : public NamespaceBase {
  ModuleNamespace() {}
  ModuleNamespace(FModuleLike module) { add(module); }

  /// Populate the namespace from a module-like operation. This namespace will
  /// be composed of the `inner_sym`s of the module's ports and declarations.
  void add(FModuleLike module) {
    for (auto portSymbol : module.getPortSymbolsAttr().getAsRange<StringAttr>())
      if (!portSymbol.getValue().empty())
        internal.insert(portSymbol.getValue());
    module.walk([&](Operation *op) {
      auto attr = op->getAttrOfType<StringAttr>("inner_sym");
      if (attr)
        internal.insert(attr.getValue());
    });
  }
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_NAMESPACE_H
