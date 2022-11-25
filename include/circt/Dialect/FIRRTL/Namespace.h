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
    for (auto &op : *circuit.getBodyBlock())
      if (auto symbol = op.getAttrOfType<mlir::StringAttr>(
              SymbolTable::getSymbolAttrName()))
        nextIndex.insert({symbol.getValue(), 0});
  }
};

/// The namespace of a `FModuleLike` operation, generally inhabited by its ports
/// and declarations.
struct ModuleNamespace : public Namespace {
  ModuleNamespace() {}
  ModuleNamespace(FModuleLike module) : module(module) { add(module); }

  /// Populate the namespace from a module-like operation. This namespace will
  /// be composed of the `inner_sym`s of the module's ports and declarations.
  void add(FModuleLike module) {
    addPorts(module);
    addBody(module);
  }

  /// Populate the namespace with the ports of a module-like operation.
  void addPorts(FModuleLike module) {
    for (auto portSymbol : module.getPortSymbolsAttr())
      if (portSymbol)
        static_cast<void>(portSymbol.cast<hw::InnerSymAttr>().walkSymbols(
            [&](StringAttr sName) {
              nextIndex.insert({sName.getValue(), 0});
              return success();
            }));
  }

  void addPorts(ArrayRef<PortInfo> ports) {
    for (auto port : ports)
      if (port.sym)
        static_cast<void>(port.sym.cast<hw::InnerSymAttr>().walkSymbols(
            [&](StringAttr symName) {
              nextIndex.insert({symName.getValue(), 0});
              return success();
            }));
  }

  /// Populate the namespace with the body of a module-like operation.
  void addBody(FModuleLike module) {
    module.walk([&](Operation *op) {
      auto attr = getInnerSymName(op);
      if (attr)
        nextIndex.insert({attr.getValue(), 0});
    });
  }

  /// The module associated with this namespace.
  FModuleLike module;
};

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_NAMESPACE_H
