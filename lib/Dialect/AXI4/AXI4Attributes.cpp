//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Attributes.h"
#include "circt/Dialect/AXI4/AXI4Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"

using namespace circt;
using namespace axi4;
using namespace mlir;

#include "circt/Dialect/AXI4/AXI4Enums.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "circt/Dialect/AXI4/AXI4Attributes.cpp.inc"

LogicalResult
BurstSpecAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      BurstKind kind, uint32_t len) {
  // AXI4 permits 1-16 beats for 'fixed', 1-256 for 'incr', and only 2, 4, 8,
  // or 16 for 'wrap'.
  switch (kind) {
  case BurstKind::Fixed:
    if (len == 0 || len > 16)
      return emitError() << "'fixed' burst 'len' must be between 1 and 16, got "
                         << len;
    break;
  case BurstKind::Incr:
    if (len == 0 || len > 256)
      return emitError() << "'incr' burst 'len' must be between 1 and 256, got "
                         << len;
    break;
  case BurstKind::Wrap:
    if (len < 2 || len > 16 || !llvm::isPowerOf2_32(len))
      return emitError() << "'wrap' burst 'len' must be 2, 4, 8, or 16, got "
                         << len;
    break;
  }
  return success();
}

/// Ordering for burst_set construction
bool BurstSpecAttr::compareCanonical(BurstSpecAttr lhs, BurstSpecAttr rhs) {
  if (lhs.getKind() != rhs.getKind())
    return lhs.getKind() < rhs.getKind();
  return lhs.getLen() < rhs.getLen();
}

LogicalResult BurstSetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                   ArrayRef<BurstSpecAttr> burstSpecs) {
  if (burstSpecs.empty())
    return emitError() << "'burst_set' must be non-empty";
  return success();
}

LogicalResult WindowAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 uint64_t base, uint64_t last,
                                 BurstSetAttr burstSpecs) {
  if (last < base)
    return emitError() << "window 'last' address 0x"
                       << llvm::utohexstr(last, /*LowerCase=*/true)
                       << " must not be less than 'base' address 0x"
                       << llvm::utohexstr(base, /*LowerCase=*/true);
  return success();
}

SmallVector<WindowAttr> WindowSetAttr::normalize(MLIRContext *ctx,
                                                 ArrayRef<WindowAttr> windows) {
  // Split the address space at every point where the specs could change (i.e.
  // the start and end points of the windows).
  // Then, for each of those segments, accumulate all the specs that cover it,
  // and create a window accordingly.

  SmallVector<uint64_t> cuts;
  for (WindowAttr window : windows) {
    cuts.push_back(window.getBase());
    if (window.getLast() != UINT64_MAX)
      cuts.push_back(window.getLast() + 1);
  }
  llvm::sort(cuts);
  cuts.erase(llvm::unique(cuts), cuts.end());

  SmallVector<WindowAttr> normalized;
  SmallVector<BurstSpecAttr> specs;
  for (size_t i = 0; i < cuts.size(); ++i) {
    uint64_t lo = cuts[i];
    uint64_t hi = i + 1 < cuts.size() ? cuts[i + 1] - 1 : UINT64_MAX;

    // Collect the set of specs that cover this segment
    specs.clear();
    for (WindowAttr window : windows)
      if (window.getBase() <= lo && lo <= window.getLast())
        llvm::append_range(specs, window.getBurstSpecs().getBurstSpecs());

    // Nothing covers the segment - a gap between windows.
    if (specs.empty())
      continue;

    // BurstSetAttr::get sorts and de-duplicates the union for us.
    auto burstSpecs = BurstSetAttr::get(ctx, specs);

    // Merge into the previous window where they are contiguous and share
    // capabilities.
    if (!normalized.empty() && normalized.back().getLast() + 1 == lo &&
        normalized.back().getBurstSpecs() == burstSpecs)
      lo = normalized.pop_back_val().getBase();
    normalized.push_back(WindowAttr::get(ctx, lo, hi, burstSpecs));
  }
  return normalized;
}

LogicalResult
WindowSetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      ArrayRef<WindowAttr> windows) {
  if (windows.empty())
    return emitError() << "'window_set' must be non-empty";
  return success();
}

void AXI4Dialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "circt/Dialect/AXI4/AXI4Attributes.cpp.inc"
      >();
}
