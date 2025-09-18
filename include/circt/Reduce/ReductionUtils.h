//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_REDUCE_REDUCTIONUTILS_H
#define CIRCT_REDUCE_REDUCTIONUTILS_H

#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Support/LLVM.h"

namespace circt {
// Forward declarations.
struct Reduction;

namespace reduce {

/// Starting at the given `op`, traverse through it and its operands and erase
/// operations that have no more uses.
void pruneUnusedOps(Operation *initialOp, Reduction &reduction);

/// A helper struct that scans a root operation and all its nested operations
/// for `InnerRefAttr`s.
struct InnerSymbolUses {
  InnerSymbolUses(Operation *root);

  InnerSymbolUses() = default;
  InnerSymbolUses(const InnerSymbolUses &) = default;
  InnerSymbolUses(InnerSymbolUses &&) = default;

  InnerSymbolUses &operator=(const InnerSymbolUses &) = default;
  InnerSymbolUses &operator=(InnerSymbolUses &&) = default;

  /// Check whether an op is involved in an `InnerRefAttr`. Considers both the
  /// `sym_name` and the `inner_sym` attributes on the given op.
  bool hasUses(Operation *op) const;
  /// Check if the given `InnerRefAttr` is used.
  bool hasUses(hw::InnerRefAttr inner) const;
  /// Check if the given module name is involved in an `InnerRefAttr`.
  bool hasUses(StringAttr mod) const;
  /// Check if the given module and inner symbol name is involved in an
  /// `InnerRefAttr`.
  bool hasUses(StringAttr mod, StringAttr sym) const;

private:
  /// Module and inner symbol name pairs used in `InnerRefAttr`s.
  DenseSet<std::pair<StringAttr, StringAttr>> uses;
  /// Module names used in `InnerRefAttr`s.
  DenseSet<StringAttr> moduleUses;
};

} // namespace reduce
} // namespace circt

#endif // CIRCT_REDUCE_REDUCTIONUTILS_H
