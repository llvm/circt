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
#include "circt/Dialect/FIRRTL/FIRRTLUtils.h"
#include "circt/Dialect/FIRRTL/Passes.h"

using namespace circt;
using namespace firrtl;

namespace {
struct DropNamesPass : public DropNameBase<DropNamesPass> {
  DropNamesPass(PreserveValues::PreserveMode preserveMode) {
    this->preserveMode = preserveMode;
  }

  void runOnOperation() override {
    if (preserveMode == PreserveValues::None) {
      // Drop all names.
      numNamesConverted += dropNamesIf([](FNamableOp) { return true; });
    } else if (preserveMode == PreserveValues::Named) {
      // Drop the name if it isn't considered meaningful.
      numNamesConverted += dropNamesIf(
          [](FNamableOp op) { return isUselessName(op.getName()); });
    }
  }

private:
  size_t dropNamesIf(llvm::function_ref<bool(FNamableOp)> pred) {
    size_t changedNames = 0;
    auto droppableNameAttr =
        NameKindEnumAttr::get(&getContext(), NameKindEnum::DroppableName);
    getOperation()->walk([&](FNamableOp op) {
      if (pred(op) && !op.hasDroppableName()) {
        ++changedNames;
        op.setNameKindAttr(droppableNameAttr);
      }
    });
    return changedNames;
  }
};

} // end anonymous namespace

std::unique_ptr<mlir::Pass>
circt::firrtl::createDropNamesPass(PreserveValues::PreserveMode mode) {
  return std::make_unique<DropNamesPass>(mode);
}
