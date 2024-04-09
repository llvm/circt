//===- NestedSymbolTable.cpp - NestedSymbolTable and NestedRef verification ==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements NestedSymbolTable and verification for NestedRef's.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Ibis/InnerSymbolTable.h"
#include "circt/Dialect/Ibis/NESTEDOpInterfaces.h"
#include "mlir/IR/Threading.h"
#include "llvm/Support/Debug.h"

using namespace circt;
using namespace ibis;

Operation *NestedSymbolNamespace::lookup(Operation *from,
                                         ArrayRef<StringAttr> path,
                                         StringRef name) const {
  Operation *targetOp = nullptr;
  bool first = true;
  while (!targetOp) {
    if (!first) {
      if (from->getParentOp() == nullptr) {
        return nullptr;
      }
      from = from->getParentOp();
    }
    NestedSymbolTable &fromSymTable = getSymbolTable(from);
    targetOp = fromSymTable.lookup(path, name);
    first = false;
  }
  return targetOp;
}
