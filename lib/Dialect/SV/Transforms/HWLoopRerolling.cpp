//===- HWLoopRerolling.cpp - HW Memory Implementation Pass
//-------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass converts generated FIRRTL memory modules to
// simulation models.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/Namespace.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqAttributes.h"
#include "circt/Support/LoopRerolling.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Path.h"

using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// HWLoopRerollingPass Pass
//===----------------------------------------------------------------------===//

struct HWLoopRerollingPass
    : public sv::HWLoopRerollingBase<HWLoopRerollingPass> {
  void runOnOperation() override;
};

void HWLoopRerollingPass::runOnOperation() {
  auto topModule = getOperation();
  for (auto &op : topModule.getBodyBlock()->getOperations()) {
    auto orOp = dyn_cast<comb::OrOp>(&op);
    if (!orOp)
      continue;
    unsigned size = orOp->getNumOperands();
    // If the loop is small enough.
    if (size <= 4)
      continue;

    ImplicitLocOpBuilder builder(orOp.getLoc(), orOp);
    // Unpacked array.
    auto unpacked = hw::UnpackedArrayType::get(orOp.getType(), size);
    auto wire = builder.create<sv::WireOp>(unpacked);

    unsigned start = 0;
    auto fn = [&](unsigned idx) {
      // Assign.
    };

    auto always_comb = builder.create<sv::AlwaysCombOp>(orOp.getLoc());
    while (start < size) {
      if (start == size - 1) {
        fn(start);
        break;
      }

      builder.setInsertionPointToEnd(always_comb.getBodyBlock());
      circt::LoopReroller reroller(builder);
      auto value = orOp.getOperand(start);
      auto next = orOp.getOperand(start + 1);
      if (failed(reroller.unifyTwoValues(value, next))) {
        fn(start++);
        continue;
      }
    }
  }
}

std::unique_ptr<Pass> circt::sv::createHWLoopRerollingPass() {
  auto pass = std::make_unique<HWLoopRerollingPass>();
  return pass;
}
