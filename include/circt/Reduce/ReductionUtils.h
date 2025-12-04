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

  /// Check whether an op is targeted by an inner ref. Considers both the
  /// `sym_name` and the `inner_sym` attributes on the given op.
  bool hasInnerRef(Operation *op) const;
  /// Check if the given inner ref is used.
  bool hasInnerRef(hw::InnerRefAttr innerRef) const;
  /// Check if the given symbol name is targeted by an inner ref.
  bool hasInnerRef(StringAttr symbol) const;
  /// Check if the given symbol and inner symbol name pair is targeted by an
  /// inner ref.
  bool hasInnerRef(StringAttr symbol, StringAttr innerSym) const;

  /// Check whether the given symbol is targeted by a symbol ref.
  bool hasSymbolRef(Operation *op) const;
  /// Check whether the given symbol name is targeted by a symbol ref.
  bool hasSymbolRef(StringAttr symbol) const;

  /// Check whether the given symbol is targeted by a symbol ref or inner ref.
  bool hasRef(Operation *op) const;
  /// Check whether the given symbol name is targeted by a symbol ref or inner
  /// ref.
  bool hasRef(StringAttr symbol) const;

private:
  /// Symbol and inner symbol name pairs used in inner refs.
  DenseSet<std::pair<StringAttr, StringAttr>> innerRefs;
  /// Symbol names used in inner refs.
  DenseSet<StringAttr> innerRefModules;
  /// Symbol names used in symbol or inner refs.
  DenseSet<StringAttr> symbolRefs;
};

} // namespace reduce
} // namespace circt

#endif // CIRCT_REDUCE_REDUCTIONUTILS_H
