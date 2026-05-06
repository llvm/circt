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

//===----------------------------------------------------------------------===//
// Layer Verification Utilities
//===----------------------------------------------------------------------===//

/// Get the ambient layers active at the given op.
LayerSet getAmbientLayersAt(Operation *op);

/// Get the ambient layer requirements at the definition site of the value.
LayerSet getAmbientLayersFor(Value value);

/// Get the effective layer requirements for the given value.
/// The effective layers for a value is the union of
///   - the ambient layers for the cannonical storage location.
///   - any explicit layer annotations in the value's type.
LayerSet getLayersFor(Value value);

/// Check that the source layer is compatible with the destination layer.
/// Either the source and destination are identical, or the source-layer
/// is a parent of the destination. For example `A` is compatible with `A.B.C`,
/// because any definition valid in `A` is also valid in `A.B.C`.
bool isLayerCompatibleWith(mlir::SymbolRefAttr srcLayer,
                           mlir::SymbolRefAttr dstLayer);

/// Check that the source layer is present in the destination layers.
bool isLayerCompatibleWith(SymbolRefAttr srcLayer, const LayerSet &dstLayers);

/// Check that the source layers are all present in the destination layers.
/// True if all source layers are present in the destination.
/// Outputs the set of source layers that are missing in the destination.
bool isLayerSetCompatibleWith(const LayerSet &src, const LayerSet &dst,
                              SmallVectorImpl<SymbolRefAttr> &missing);

LogicalResult checkLayerCompatibility(
    Operation *op, const LayerSet &src, const LayerSet &dst,
    const Twine &errorMsg,
    const Twine &noteMsg = Twine("missing layer requirements"));

} // namespace firrtl
} // namespace circt

#endif // CIRCT_DIALECT_FIRRTL_LAYERSET_H
