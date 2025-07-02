//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
namespace llhd {
#define GEN_PASS_DEF_SELECTTOMUXPASS
#include "circt/Dialect/LLHD/Transforms/LLHDPasses.h.inc"
} // namespace llhd
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace circt::llhd;

namespace {
struct SelectToMuxPass
    : public llhd::impl::SelectToMuxPassBase<SelectToMuxPass> {
  void runOnOperation() override;
};
} // namespace

void SelectToMuxPass::runOnOperation() {
  getOperation().walk([](Operation *op) {
    OpBuilder builder(op);

    // arith.constant
    if (auto constOp = dyn_cast<arith::ConstantOp>(op)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValueAttr())) {
        constOp.replaceAllUsesWith(
            builder.create<hw::ConstantOp>(constOp.getLoc(), intAttr)
                .getResult());
        constOp.erase();
      }
      return;
    }

    // arith.select
    if (auto selectOp = dyn_cast<arith::SelectOp>(op)) {
      selectOp.replaceAllUsesWith(
          builder
              .create<comb::MuxOp>(selectOp.getLoc(), selectOp.getCondition(),
                                   selectOp.getTrueValue(),
                                   selectOp.getFalseValue())
              .getResult());
      selectOp.erase();
      return;
    }
  });
}
