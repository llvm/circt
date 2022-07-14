//===- LowerSeqToSV.cpp - Seq to SV lowering ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform translate Seq ops to SV.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/SV/SVAttributes.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Seq/SeqPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace seq;

namespace {
struct SeqToSVPass : public LowerSeqToSVBase<SeqToSVPass> {
  void runOnOperation() override;
};
} // anonymous namespace

namespace {
/// Lower CompRegOp to `sv.reg` and `sv.alwaysff`. Use a posedge clock and
/// synchronous reset.
struct CompRegLower : public OpConversionPattern<CompRegOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CompRegOp reg, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = reg.getLoc();

    auto svReg = rewriter.create<sv::RegOp>(loc, reg.getResult().getType(),
                                            reg.getNameAttr());
    svReg->setDialectAttrs(reg->getDialectAttrs());

    // If the seq::CompRegOp has an inner_sym attribute, set this for the
    // sv::RegOp inner_sym attribute.
    if (reg.getSymName().hasValue())
      svReg.setInnerSymAttr(reg.getSymNameAttr());

    if (auto attribute = circt::sv::getSVAttributes(reg))
      circt::sv::setSVAttributes(svReg, attribute);

    auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);
    if (reg.getReset() && reg.getResetValue()) {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.getClk(), ResetType::SyncReset,
          sv::EventControl::AtPosEdge, reg.getReset(),
          [&]() { rewriter.create<sv::PAssignOp>(loc, svReg, reg.getInput()); },
          [&]() {
            rewriter.create<sv::PAssignOp>(loc, svReg, reg.getResetValue());
          });
    } else {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.getClk(), [&]() {
            rewriter.create<sv::PAssignOp>(loc, svReg, reg.getInput());
          });
    }

    rewriter.replaceOp(reg, {regVal});
    return success();
  }
};
} // namespace

void SeqToSVPass::runOnOperation() {
  ModuleOp top = getOperation();
  MLIRContext &ctxt = getContext();

  ConversionTarget target(ctxt);
  target.addIllegalDialect<SeqDialect>();
  target.addLegalDialect<sv::SVDialect>();
  RewritePatternSet patterns(&ctxt);
  patterns.add<CompRegLower>(&ctxt);

  if (failed(applyPartialConversion(top, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::seq::createSeqLowerToSVPass() {
  return std::make_unique<SeqToSVPass>();
}
