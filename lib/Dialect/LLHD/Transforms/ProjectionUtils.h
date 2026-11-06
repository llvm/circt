//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helpers to reason about `llhd.sig.*` projections: the portion of a signal's
// storage they select, and whether two projections provably target disjoint
// storage.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_SUPPORT_PROJECTIONUTILS_H
#define CIRCT_SUPPORT_PROJECTIONUTILS_H

#include "circt/Dialect/LLHD/LLHDTypes.h"
#include "circt/Support/LLVM.h"
#include "mlir/IR/Value.h"

#include <utility>
#include <variant>

namespace circt {
namespace llhd {

/// Portion of a signal selected by a projection, with a statically known
/// offset and width in bits.
struct ConstProjectionRange {
  uint64_t offset = 0;
  uint64_t width = 1;
};

/// Portion of a signal selected by a projection with a dynamic index. The
/// exact offset is unknown.
struct DynProjectionRange {
  Value index;
};

/// Portion of the signal operand that is selected by a projection.
using ProjectionRange = std::variant<ConstProjectionRange, DynProjectionRange>;

/// Helper to check if a projection range is dynamic or not.
inline bool isDynamic(const ProjectionRange &range) {
  return std::holds_alternative<DynProjectionRange>(range);
}

/// Return the type stored in a signal or projection.
inline Type getStoredType(Value slot) {
  return cast<RefType>(slot.getType()).getNestedType();
}

/// Check if an op is one of the supported projection types.
bool isProjection(const Operation *op);

/// Calculate a projection's range.
/// Returns nullopt if the range could not be calculated for this projection.
/// Asserts if the provided Operation is not a (supported) projection.
std::optional<ProjectionRange> getProjectionRange(Operation *op);

/// Check whether two projections provably describe disjoint storage.
bool isProvablyDisjoint(const ProjectionRange &a, const ProjectionRange &b);

/// Given a signal or a projection, walk back the projection chain until you
/// find the root signal declaration. A nullptr is returned if at any point of
/// the use-def chain we encounter an Operation that is not a (supported)
/// projection.
Operation *getRootSignal(Value sig);

/// Given a signal or a projection, find the root signal declaration by
/// traversing the projection chain. Optionally, if the projection chain is
/// fully constant, return the portion of the original signal that this
/// operation probes or drives. A null Operation is returned if the use-def
/// chain does not bottom-out in a signal declaration, and a null range is
/// returned if a dynamic index is found.
std::pair<Operation *, std::optional<ProjectionRange>>
getRootSignalAndRange(Value sig);

} // namespace llhd
} // namespace circt

#endif // CIRCT_SUPPORT_PROJECTIONUTILS_H
