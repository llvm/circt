//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_FIRRTL_LAYERSET_H
#define CIRCT_DIALECT_FIRRTL_LAYERSET_H

#include "circt/Support/LLVM.h"
#include "mlir/IR/BuiltinAttributes.h"

namespace circt {
namespace firrtl {

/// Compares two `SymbolRefAttr` lexicographically, returning true if LHS should
/// be ordered before RHS.
struct LayerSetCompare {
  bool operator()(SymbolRefAttr lhs, SymbolRefAttr rhs) const {
    auto cmp = lhs.getRootReference().compare(rhs.getRootReference());
    if (cmp == -1)
      return true;
    if (cmp == 1)
      return false;
    auto lhsNested = lhs.getNestedReferences();
    auto rhsNested = rhs.getNestedReferences();
    auto lhsNestedSize = lhsNested.size();
    auto rhsNestedSize = rhsNested.size();
    auto e = std::min(lhsNestedSize, rhsNestedSize);
    for (unsigned i = 0; i < e; ++i) {
      auto cmp = lhsNested[i].getAttr().compare(rhsNested[i].getAttr());
      if (cmp == -1)
        return true;
      if (cmp == 1)
        return false;
    }
    return lhsNestedSize < rhsNestedSize;
  }
};

using LayerSet = SmallSet<SymbolRefAttr, 4, LayerSetCompare>;

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_LAYERSET_H
