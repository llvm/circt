//===- DropName.cpp - Drop Names  -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the DropName pass.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/Passes.h"

using namespace circt;
using namespace firrtl;

namespace {
struct DropNamesPass : public DropNameBase<DropNamesPass> {
  void runOnOperation() override {
    size_t changedNames = 0;
    getOperation()->walk([&changedNames](FNamableOp op) {
      if (!op.hasDroppableName()) {
        ++changedNames;
        op.dropName();
      }
    });

    numNamesConverted += changedNames;
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass> circt::firrtl::createDropNamesPass() {
  return std::make_unique<DropNamesPass>();
}
