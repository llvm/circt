//===- InnerSymbolNamespace.h - Inner Symbol Table Namespace ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the InnerSymbolNamespace, which tracks the names
// used by inner symbols within an InnerSymbolTable.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_HW_INNERSYMBOLNAMESPACE_H
#define CIRCT_DIALECT_HW_INNERSYMBOLNAMESPACE_H

#include "circt/Dialect/HW/InnerSymbolTable.h"
#include "circt/Support/Namespace.h"

namespace circt {
namespace hw {

struct InnerSymbolNamespace : Namespace {
  InnerSymbolNamespace() = default;
  InnerSymbolNamespace(Operation *module) { add(module); }

  /// Populate the namespace from a module-like operation. This namespace will
  /// be composed of the `inner_sym`s of the module's ports and declarations.
  void add(Operation *module) {
    hw::InnerSymbolTable::walkSymbols(module, [&](StringAttr name,
                                                  const InnerSymTarget &target,
                                                  Operation * /*currentIST*/) {
      nextIndex.insert({name.getValue(), 0});
    });
  }
};

struct InnerSymbolNamespaceCollection {

  InnerSymbolNamespace &get(Operation *op) {
    return collection.try_emplace(op, op).first->second;
  }

  InnerSymbolNamespace &operator[](Operation *op) { return get(op); }

private:
  DenseMap<Operation *, InnerSymbolNamespace> collection;
};

} // namespace hw
} // namespace circt

#endif // CIRCT_DIALECT_HW_INNERSYMBOLNAMESPACE_H
