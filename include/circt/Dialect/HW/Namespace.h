//===- Namespace.h - A symbol table for HW ops ------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements HW namespace which are symbol tables for HW operations
// that automatically resolve name collisions.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_NAMESPACE_H
#define CIRCT_DIALECT_HW_NAMESPACE_H

#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Namespace.h"

namespace circt {
namespace hw {

struct ModuleNamespace : Namespace {
  ModuleNamespace() = default;
  ModuleNamespace(hw::HWModuleOp module) { add(module); }

  /// Populate the namespace from a module-like operation. This namespace will
  /// be composed of the `inner_sym`s of the module's ports and declarations.
  void add(hw::HWModuleOp module) {
    for (auto port : module.getAllPorts())
      if (port.sym && !port.sym.empty())
        nextIndex.insert({port.sym.getSymName().getValue(), 0});
    module.walk([&](Operation *op) {
      auto attr = op->getAttrOfType<StringAttr>("inner_sym");
      if (attr)
        nextIndex.insert({attr.getValue(), 0});
    });
  }
};

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_NAMESPACE_H
