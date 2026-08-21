//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AXI4 ops.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AXI4/AXI4Ops.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/MathExtras.h"

using namespace circt;
using namespace axi4;
using namespace mlir;

//===----------------------------------------------------------------------===//
// XbarOp
//===----------------------------------------------------------------------===//

namespace {
/// A width field of a port type, named for diagnostics.
struct WidthField {
  llvm::StringLiteral name;
  uint32_t (PortType::*get)() const;
};
} // namespace

// The port width fields, the ones an xbar carries through unchanged first,
// followed by the ID widths it widens.
static constexpr WidthField kWidths[] = {
    {"addr_width", &PortType::getAddrWidth},
    {"data_width", &PortType::getDataWidth},
    {"user_width", &PortType::getUserWidth},
    {"write_id_width", &PortType::getWriteIdWidth},
    {"read_id_width", &PortType::getReadIdWidth}};
static constexpr size_t kNumSharedWidths = 3;

/// Verify that `port` agrees with `reference` on each of the widths in
/// `fields`.
static LogicalResult verifyWidthsMatch(XbarOp op, ArrayRef<WidthField> fields,
                                       PortType port, const Twine &portDesc,
                                       PortType reference,
                                       const Twine &referenceDesc) {
  for (const WidthField &field : fields) {
    uint32_t width = (port.*field.get)();
    uint32_t expected = (reference.*field.get)();
    if (width != expected)
      return op.emitOpError()
             << portDesc << "'s '" << field.name << "' (" << width
             << ") must match " << referenceDesc << "'s (" << expected << ")";
  }
  return success();
}

/// The downstream port and window covering `address`, or a null window if no
/// downstream port covers it.
static std::pair<size_t, WindowAttr> findDownstreamWindow(ValueRange downstream,
                                                          uint64_t address) {
  for (auto [i, value] : llvm::enumerate(downstream))
    for (WindowAttr window :
         cast<PortType>(value.getType()).getWindows().getWindows())
      if (window.getBase() <= address && address <= window.getLast())
        return {i, window};
  return {0, {}};
}

LogicalResult XbarOp::verify() {
  ValueRange upstream = getUpstream();
  ValueRange downstream = getDownstream();
  if (upstream.empty())
    return emitOpError("must have at least one upstream port");
  if (downstream.empty())
    return emitOpError("must have at least one downstream port");

  // Make sure all upstream ports agree on widths
  auto upstreamTy = cast<PortType>(upstream.front().getType());
  for (auto [i, value] : llvm::enumerate(upstream.drop_front()))
    if (failed(verifyWidthsMatch(
            *this, kWidths, cast<PortType>(value.getType()),
            "upstream port #" + Twine(i + 1), upstreamTy, "upstream port #0")))
      return failure();

  // Each manager's transactions are tagged with its index downstream.
  uint32_t idBits = llvm::Log2_64_Ceil(upstream.size());

  // Make sure all downstream ports agree on address, data, and user widths
  for (auto [i, value] : llvm::enumerate(downstream)) {
    auto downstreamTy = cast<PortType>(value.getType());
    if (failed(verifyWidthsMatch(
            *this, ArrayRef(kWidths).take_front(kNumSharedWidths), downstreamTy,
            "downstream port #" + Twine(i), upstreamTy, "upstream port #0")))
      return failure();

    // Make sure downstream ports are wide enough to uniquely tag transactions
    // from upstream ports.
    for (const WidthField &field :
         ArrayRef(kWidths).drop_front(kNumSharedWidths)) {
      uint32_t least = (upstreamTy.*field.get)() + idBits;
      if ((downstreamTy.*field.get)() < least)
        return emitOpError()
               << "downstream port #" << i << "'s '" << field.name
               << "' must be at least " << least << " to tag transactions from "
               << upstream.size() << " managers, got "
               << (downstreamTy.*field.get)();
    }
  }

  // Downstream windows must not overlap (so routing is unambiguous)
  for (auto [i, value] : llvm::enumerate(downstream)) {
    auto windows = cast<PortType>(value.getType()).getWindows();
    for (auto [j, other] : llvm::enumerate(downstream.take_front(i)))
      if (windows.overlaps(cast<PortType>(other.getType()).getWindows()))
        return emitOpError() << "downstream ports #" << j << " and #" << i
                             << " have overlapping windows";
  }

  // Every window a manager can access must be routed downstream, to a port
  // supporting at least the bursts the manager issues there.
  // We've already verified no overlap, so we can just check the existence of a
  // supporting window.
  for (auto [i, value] : llvm::enumerate(upstream)) {
    auto managerTy = cast<PortType>(value.getType());
    for (WindowAttr window : managerTy.getWindows().getWindows()) {
      // Walk through addresses, skipping the ones we know are covered
      // Begin at the window's start
      for (uint64_t address = window.getBase();;) {
        // Make sure it's supported
        auto [j, covering] = findDownstreamWindow(downstream, address);
        if (!covering)
          return emitOpError()
                 << "address 0x" << llvm::utohexstr(address, /*LowerCase=*/true)
                 << ", in upstream port #" << i
                 << "'s windows, is not covered by any downstream port";
        if (!covering.getBurstSpecs().contains(window.getBurstSpecs()))
          return emitOpError()
                 << "downstream port #" << j
                 << " does not support all the bursts upstream port #" << i
                 << " issues at address 0x"
                 << llvm::utohexstr(address, /*LowerCase=*/true)
                 << "; upstream requires " << window.getBurstSpecs()
                 << ", downstream supports " << covering.getBurstSpecs();
        // If we know that every remaining address in the upstream window is
        // covered by this downstream window, we're done
        if (covering.getLast() >= window.getLast())
          break;
        // Otherwise, skip ahead to the next address that we don't already know
        // is covered (the address directly after the end of the covering
        // downstream window)
        address = covering.getLast() + 1;
      }
    }
  }

  return success();
}

//===----------------------------------------------------------------------===//
// TableGen generated logic.
//===----------------------------------------------------------------------===//

// Provide the autogenerated implementation guts for the Op classes.
#define GET_OP_CLASSES
#include "circt/Dialect/AXI4/AXI4.cpp.inc"
