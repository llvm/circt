//===- LowerSizeOf.cpp - Elaborate sizeof intrinsics -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the LowerSizeOf pass for lowering FIRRTL sizeof intrinsic
// operations to constants.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "circt/Dialect/FIRRTL/FIRRTLTypes.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "mlir/Pass/Pass.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_LOWERSIZEOF
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;

namespace {
class LowerSizeOfPass
    : public circt::firrtl::impl::LowerSizeOfBase<LowerSizeOfPass> {
  using Base::Base;

  void runOnOperation() override {
    SmallVector<SizeOfIntrinsicOp> sizeofOps;
    getOperation().walk([&](SizeOfIntrinsicOp op) { sizeofOps.push_back(op); });

    if (sizeofOps.empty()) {
      markAllAnalysesPreserved();
      return;
    }

    for (SizeOfIntrinsicOp op : sizeofOps) {
      auto width = getBitWidth(op.getInput().getType());
      if (!width) {
        op.emitError(
              "failed to elaborate sizeof intrinsic: unable to determine "
              "operand width")
                .attachNote(op.getInput().getLoc())
            << "operand of type " << op.getInput().getType()
            << " defined here has unknown width";
        return signalPassFailure();
      }

      OpBuilder builder(op);
      auto constant = ConstantOp::create(builder, op.getLoc(), op.getType(),
                                         APInt(32, *width));
      op.getResult().replaceAllUsesWith(constant);
      op.erase();
    }
  }
};
} // namespace
