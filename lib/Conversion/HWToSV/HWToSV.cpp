//===- HWToSV.cpp - HW To SV Conversion Pass ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to SV Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToSV.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/TypeSwitch.h"

namespace circt {
#define GEN_PASS_DEF_LOWERHWTOSV
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace sv;

static sv::EventControl hwToSvEventControl(hw::EventControl ec) {
  switch (ec) {
  case hw::EventControl::AtPosEdge:
    return sv::EventControl::AtPosEdge;
  case hw::EventControl::AtNegEdge:
    return sv::EventControl::AtNegEdge;
  case hw::EventControl::AtEdge:
    return sv::EventControl::AtEdge;
  }
  llvm_unreachable("Unknown event control kind");
}

namespace {
struct HWToSVPass : public circt::impl::LowerHWToSVBase<HWToSVPass> {
  void runOnOperation() override;
};

struct TriggeredOpConversionPattern : public OpConversionPattern<TriggeredOp> {
  using OpConversionPattern<TriggeredOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TriggeredOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto alwaysOp = AlwaysOp::create(
        rewriter, op.getLoc(),
        llvm::SmallVector<sv::EventControl>{hwToSvEventControl(op.getEvent())},
        llvm::SmallVector<Value>{op.getTrigger()});
    rewriter.mergeBlocks(op.getBodyBlock(), alwaysOp.getBodyBlock(),
                         operands.getInputs());
    rewriter.eraseOp(op);
    return success();
  }
};

struct AttachOpConversionPattern : public OpConversionPattern<hw::AttachOp> {
  using OpConversionPattern<hw::AttachOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hw::AttachOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Value> inoutValues(operands.getSignals().begin(),
                                   operands.getSignals().end());

    // Assuming all the necessary macro symbols have already been declared, and
    // these declarations will be processed in the LowerToHW phase, we will
    // refrain from declaring macro symbols here to avoid modifying the
    // top-level module and redefining symbols.
    sv::IfDefOp::create(
        rewriter, loc, "SYNTHESIS",
        [&]() {
          SmallVector<Value> values;
          values.reserve(inoutValues.size());
          for (auto inoutValue : inoutValues)
            values.push_back(
                sv::ReadInOutOp::create(rewriter, loc, inoutValue));

          for (size_t i1 = 0, e = inoutValues.size(); i1 != e; ++i1)
            for (size_t i2 = 0; i2 != e; ++i2)
              if (i1 != i2)
                sv::AssignOp::create(rewriter, loc, inoutValues[i1],
                                     values[i2]);
        },
        [&]() {
          sv::IfDefOp::create(
              rewriter, loc, "VERILATOR",
              [&]() {
                sv::VerbatimOp::create(
                    rewriter, loc,
                    "`error \"Verilator does not support alias and thus "
                    "cannot arbitrarily connect bidirectional wires and "
                    "ports\"");
              },
              [&]() { sv::AliasOp::create(rewriter, loc, inoutValues); });
        });

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void HWToSVPass::runOnOperation() {
  MLIRContext &context = getContext();
  hw::HWModuleOp module = getOperation();

  ConversionTarget target(context);
  RewritePatternSet patterns(&context);

  target.addIllegalOp<TriggeredOp, hw::AttachOp>();
  target.addLegalDialect<sv::SVDialect>();

  patterns.add<TriggeredOpConversionPattern, AttachOpConversionPattern>(
      &context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// HW to SV Conversion Pass
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<hw::HWModuleOp>> circt::createLowerHWToSVPass() {
  return std::make_unique<HWToSVPass>();
}
