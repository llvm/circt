//===- VerifToSV.cpp - HW To SV Conversion Pass ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Verif to SV Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSV.h"
#include "../PassDetail.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace sv;
using namespace verif;

namespace {
struct VerifToSVPass : public LowerVerifToSVBase<VerifToSVPass> {
  void runOnOperation() override;
};

struct PrintOpConversionPattern : public OpConversionPattern<PrintOp> {
  using OpConversionPattern<PrintOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PrintOp op, OpAdaptor operands,
                  ConversionPatternRewriter &rewriter) const override {

    // Printf's will be emitted to stdout (32'h8000_0001 in IEEE Std 1800-2012).
    Value fdStdout = rewriter.create<hw::ConstantOp>(
        op.getLoc(), APInt(32, 0x80000001, false));

    auto fstrOp =
        dyn_cast_or_null<FormatVerilogStringOp>(op.getString().getDefiningOp());
    if (!fstrOp)
      return op->emitOpError() << "expected FormatVerilogStringOp as the "
                                  "source of the formatted string";

    rewriter.replaceOpWithNewOp<sv::FWriteOp>(
        op, fdStdout, fstrOp.getFormatString(), fstrOp.getSubstitutions());
    return success();
  }
};

} // namespace

void VerifToSVPass::runOnOperation() {
  MLIRContext &context = getContext();
  hw::HWModuleOp module = getOperation();

  ConversionTarget target(context);
  RewritePatternSet patterns(&context);

  target.addIllegalOp<PrintOp>();
  target.addLegalDialect<sv::SVDialect, hw::HWDialect>();
  patterns.add<PrintOpConversionPattern>(&context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// Verif to SV Conversion Pass
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<hw::HWModuleOp>>
circt::createLowerVerifToSVPass() {
  return std::make_unique<VerifToSVPass>();
}
