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
#include "circt/Support/Namespace.h"

namespace circt {
namespace firrtl {

/// The namespace of a `CircuitOp`, generally inhabited by modules.
struct CircuitNamespace : public Namespace {
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
struct ModuleNamespace : public Namespace {
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
