//===- FooWires.cpp - Replace all wire names with foo ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Replace all wire names with foo.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "mlir/Pass/Pass.h"

using namespace circt;
using namespace hw;

namespace {
// A test pass that simply replaces all wire names with foo_<n>
struct FooWiresPass : FooWiresBase<FooWiresPass> {
  void runOnOperation() override;
};
} // namespace

void FooWiresPass::runOnOperation() {
  size_t nWires = 0; // Counts the number of wires modified
  getOperation().walk(
      [&](hw::WireOp wire) { // Walk over every wire in the module
        wire.setName("foo_" + std::to_string(nWires++)); // Rename said wire
      });
}

std::unique_ptr<mlir::Pass> circt::hw::createFooWiresPass() {
  return std::make_unique<FooWiresPass>();
}
