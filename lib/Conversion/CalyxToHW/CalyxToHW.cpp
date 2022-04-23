//===- CalyxToHW.cpp - Translate Calyx into HW ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main Calyx to HW Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CalyxToHW.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "circt/Dialect/HW/HWDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace circt::calyx;
using namespace circt::hw;

/// ConversionPatterns.

struct ConvertProgramOp : public OpConversionPattern<ProgramOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProgramOp program, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    ModuleOp mod = program->getParentOfType<ModuleOp>();
    rewriter.inlineRegionBefore(program.body(), &mod.getBodyRegion().front());
    rewriter.eraseBlock(&mod.getBodyRegion().back());
    return success();
  }
};

struct ConvertComponentOp : public OpConversionPattern<ComponentOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ComponentOp component, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<hw::PortInfo> hwPortInfo;
    auto calyxPortInfo = component.getPortInfo();
    for (auto portInfo : calyxPortInfo)
      hwPortInfo.push_back(
          {portInfo.name, hwDirection(portInfo.direction), portInfo.type});

    auto hwMod = rewriter.create<HWModuleOp>(
        component.getLoc(), component.getNameAttr(), hwPortInfo);

    rewriter.inlineRegionBefore(component.body(), hwMod.getBodyRegion(),
                                hwMod.getBodyRegion().begin());
    rewriter.eraseOp(component);
    rewriter.eraseBlock(&hwMod.getBodyRegion().back());

    return success();
  }

private:
  hw::PortDirection hwDirection(calyx::Direction dir) const {
    switch (dir) {
    case calyx::Direction::Input:
      return hw::PortDirection::INPUT;
    case calyx::Direction::Output:
      return hw::PortDirection::OUTPUT;
    }
  }
};

/// Pass entrypoint.

namespace {
class CalyxToHWPass : public CalyxToHWBase<CalyxToHWPass> {
public:
  void runOnOperation() override;

private:
  LogicalResult runOnProgram(ProgramOp program);
};
} // end anonymous namespace

void CalyxToHWPass::runOnOperation() {
  ModuleOp mod = getOperation();
  for (auto program : llvm::make_early_inc_range(mod.getOps<ProgramOp>()))
    if (failed(runOnProgram(program)))
      return signalPassFailure();
}

LogicalResult CalyxToHWPass::runOnProgram(ProgramOp program) {
  MLIRContext &context = getContext();

  ConversionTarget target(context);
  // target.addIllegalDialect<CalyxDialect>();
  target.addIllegalOp<ProgramOp>();
  target.addIllegalOp<ComponentOp>();
  target.addLegalDialect<HWDialect>();

  RewritePatternSet patterns(&context);
  patterns.add<ConvertProgramOp>(&context);
  patterns.add<ConvertComponentOp>(&context);

  return applyPartialConversion(program, target, std::move(patterns));
}

std::unique_ptr<mlir::Pass> circt::createCalyxToHWPass() {
  return std::make_unique<CalyxToHWPass>();
}
