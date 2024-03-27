//===- LowerSeqShiftReg.cpp - seq.shiftreg lowering -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace circt;
using namespace seq;

namespace {

struct ShiftRegLowering : public OpConversionPattern<seq::ShiftRegOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::ShiftRegOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Value in = adaptor.getInput();
    auto baseName = op.getName();
    for (size_t i = 0; i < op.getNumElements(); ++i) {
      StringAttr name;
      if (baseName.has_value())
        name = rewriter.getStringAttr(baseName.value() + "_sh" + Twine(i + 1));
      in = rewriter.create<seq::CompRegClockEnabledOp>(
          op.getLoc(), in, adaptor.getClk(), adaptor.getClockEnable(),
          adaptor.getReset(), adaptor.getResetValue(), name,
          op.getPowerOnValue());
    }

    op.replaceAllUsesWith(in);
    rewriter.eraseOp(op);
    return success();
  }
};

#define GEN_PASS_DEF_LOWERSEQSHIFTREG
#include "circt/Dialect/Seq/SeqPasses.h.inc"

struct LowerSeqShiftRegPass
    : public impl::LowerSeqShiftRegBase<LowerSeqShiftRegPass> {
  void runOnOperation() override;
};

} // namespace

void LowerSeqShiftRegPass::runOnOperation() {
  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);

  target.addIllegalOp<seq::ShiftRegOp>();
  target.addLegalDialect<seq::SeqDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<ShiftRegLowering>(&ctxt);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::seq::createLowerSeqShiftRegPass() {
  return std::make_unique<LowerSeqShiftRegPass>();
}
