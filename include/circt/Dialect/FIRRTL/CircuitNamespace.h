//===- CircuitNamespace.h - A symbol table for firrtl.circuit ---*- C++ -*-===//
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

#ifndef CIRCT_DIALECT_FIRRTL_CIRCUITNAMESPACE_H
#define CIRCT_DIALECT_FIRRTL_CIRCUITNAMESPACE_H

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "llvm/ADT/StringSet.h"

namespace circt {
namespace firrtl {

/// A namespace that is used to store existing names and generate names.  This
/// exists to work around limitations of SymbolTables.
class CircuitNamespace {
  llvm::StringSet<> internal;

public:
  /// Construct a new namespace from a circuit op.  This namespace will be
  /// composed of any operation in the first level of the circuit that contains
  /// a symbol.
  CircuitNamespace(CircuitOp circuit) {
    for (auto &op : *circuit.getBody())
      if (auto symbol = op.getAttrOfType<mlir::StringAttr>(
              SymbolTable::getSymbolAttrName()))
        internal.insert(symbol.getValue());
  }

  /// Return a unique name, derived from the input `name`, and add the new name
  /// to the internal namespace.  There are two possible outcomes for the
  /// returned name:
  ///
  /// 1. The original name is returned.
  /// 2. The name is given a `_<n>` suffix where `<n>` is a number starting from
  ///    `_0` and incrementing by one each time.
  std::string newName(const Twine &name) {
    // Special case the situation where there is no name collision to avoid
    // messing with the SmallString allocation below.
    llvm::SmallString<64> tryName;
    StringRef workingName = name.toStringRef(tryName);
    if (internal.insert(workingName).second)
      return workingName.str();

    // Try different suffixes until we get a collision-free one.
    size_t i = 0;
    if (tryName.empty())
      name.toVector(tryName); // toStringRef may leave tryName unfilled
    tryName.push_back('_');
    size_t baseLength = tryName.size();
    do {
      tryName.resize(baseLength);
      Twine(i++).toVector(tryName); // append integer to tryName
    } while (!internal.insert(tryName).second);
    return std::string(tryName);
  }
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_CIRCUITNAMESPACE_H
