//===- LowerSeqCompRegCE.cpp - seq.compreg.ce lowering --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
namespace seq {
#define GEN_PASS_DEF_LOWERSEQCOMPREGCE
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

using namespace circt;
using namespace seq;

namespace {

/// Lowers seq.compreg.ce to a seq.compreg with the clock enable signal
/// built into the next logic, i.e. `next := mux(clock_enable, next, current)`
struct CompRegCELowering
    : public OpConversionPattern<seq::CompRegClockEnabledOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(seq::CompRegClockEnabledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    // compreg.ce and compreg have the same inputs of the same types
    // other than the additional clockEnable, so not much conversion
    // is needed, we just incorporate the clock enable into the input
    auto mux =
        comb::MuxOp::create(rewriter, op.getLoc(), adaptor.getClockEnable(),
                            adaptor.getInput(), op.getResult());

    // Extract name if there is one
    StringAttr name;
    if (op.getName().has_value())
      name = rewriter.getStringAttr(op.getName().value());

    // Create the new compreg to replace the compreg.ce
    auto compreg = seq::CompRegOp::create(
        rewriter, op.getLoc(), mux, adaptor.getClk(), adaptor.getReset(),
        adaptor.getResetValue(), name, adaptor.getInitialValue());

    rewriter.replaceOp(op, compreg);
    return success();
  };

  struct LowerSeqCompRegCEPass
      : public circt::seq::impl::LowerSeqCompRegCEBase<LowerSeqCompRegCEPass> {
    void runOnOperation() override;
  };

} // namespace

void LowerSeqCompRegCEPass::runOnOperation() {
  MLIRContext &ctxt = getContext();
  ConversionTarget target(ctxt);

  // In theory all compreg.ce ops should be gone after this
  target.addIllegalOp<seq::CompRegClockEnabledOp>();
  target.addLegalDialect<seq::SeqDialect, hw::HWDialect, comb::CombDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<CompRegCELowering>(&ctxt);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::seq::createLowerSeqCompRegCEPass() {
  return std::make_unique<LowerSeqCompRegCEPass>();
}
