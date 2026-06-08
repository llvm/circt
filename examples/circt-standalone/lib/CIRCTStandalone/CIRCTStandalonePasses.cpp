//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Replace all wire names with foo.
//
//===----------------------------------------------------------------------===//

#include "CIRCTStandalone/CIRCTStandalonePasses.h"
#include "circt/Dialect/HW/HWOps.h"

#include <string>

namespace circt {
namespace standalone {
#define GEN_PASS_DEF_RENAMEWIRES
#include "CIRCTStandalone/CIRCTStandalonePasses.h.inc"

namespace {
// A test pass that simply replaces all wire names with foo_<n>.
struct RenameWires : public impl::RenameWiresBase<RenameWires> {
  using impl::RenameWiresBase<RenameWires>::RenameWiresBase;

  void runOnOperation() override {
    size_t nWires = 0; // Counts the number of wires modified.
    getOperation().walk(
        [&](hw::WireOp wire) { // Walk over every wire in the module.
          wire.setName("foo_" + std::to_string(nWires++)); // Rename said wire.
        });
  }
};
} // namespace

} // namespace standalone
} // namespace circt
