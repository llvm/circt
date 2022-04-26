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
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace circt::calyx;
using namespace circt::hw;
using namespace circt::sv;

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
    SmallVector<hw::PortInfo> hwInputInfo;
    auto inputPortInfo = component.getInputPortInfo();
    for (auto portInfo : inputPortInfo)
      hwInputInfo.push_back(
          {portInfo.name, hwDirection(portInfo.direction), portInfo.type});

    auto hwMod = rewriter.create<HWModuleOp>(
        component.getLoc(), component.getNameAttr(), hwInputInfo);

    rewriter.inlineRegionBefore(component.body(), hwMod.getBodyRegion(),
                                hwMod.getBodyRegion().begin());
    rewriter.eraseOp(component);
    rewriter.eraseBlock(&hwMod.getBodyRegion().back());

    ConversionPatternRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(hwMod.getBodyBlock());

    // TODO: keep a map from component block arg number to output index.
    SmallVector<Value> hwOutputTemps;
    auto outputPortInfo = component.getOutputPortInfo();
    for (auto portInfo : outputPortInfo)
      hwOutputTemps.push_back(
          rewriter.create<ConstantOp>(component.getLoc(), portInfo.type, 0));
    rewriter.create<OutputOp>(component.getLoc(), hwOutputTemps);

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

struct ConvertWiresOp : public OpConversionPattern<WiresOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(WiresOp wires, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    HWModuleOp hwMod = wires->getParentOfType<HWModuleOp>();
    rewriter.inlineRegionBefore(wires.body(), hwMod.getBodyRegion(),
                                hwMod.getBodyRegion().end());
    rewriter.eraseOp(wires);
    rewriter.mergeBlockBefore(&hwMod.getBodyRegion().getBlocks().back(),
                              &hwMod.getBodyBlock()->back());
    return success();
  }
};

struct ConvertControlOp : public OpConversionPattern<ControlOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ControlOp control, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    assert(control.getBody()->empty() && "calyx control must be structural");
    rewriter.eraseOp(control);
    return success();
  }
};

struct ConvertCellOp : public OpInterfaceConversionPattern<CellInterface> {
  using OpInterfaceConversionPattern::OpInterfaceConversionPattern;

  LogicalResult
  matchAndRewrite(CellInterface cell, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    assert(operands.empty() && "calyx cells do not have operands");

    SmallVector<Value> portWires;
    for (auto port : cell.getPortInfo()) {
      auto wire =
          rewriter.create<sv::WireOp>(cell.getLoc(), port.type, port.name);
      auto wireRead = rewriter.create<sv::ReadInOutOp>(cell.getLoc(), wire);
      portWires.push_back(wireRead);
    }

    rewriter.replaceOp(cell, portWires);

    return success();
  }
};

struct ConvertAssignOp : public OpConversionPattern<calyx::AssignOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(calyx::AssignOp assign, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Special case for assigning to an output. Add source of assign to output,
    // erase the dest, remove the block argument.
    if (auto arg = assign.dest().dyn_cast<BlockArgument>()) {
      llvm::errs() << "assign to arg " << arg.getArgNumber() << ' ';
      arg.getOwner()->getParentOp()->dump();
      auto hwMod = cast<HWModuleOp>(arg.getOwner()->getParentOp());
    }

    // General case converts to SV directly.
    rewriter.replaceOpWithNewOp<sv::AssignOp>(assign, assign.dest(),
                                              assign.src());

    return success();
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
  target.addIllegalOp<WiresOp>();
  target.addIllegalOp<ControlOp>();
  target.addIllegalOp<calyx::AssignOp>();

  target.addLegalDialect<HWDialect>();
  target.addLegalDialect<SVDialect>();

  RewritePatternSet patterns(&context);
  patterns.add<ConvertProgramOp>(&context);
  patterns.add<ConvertComponentOp>(&context);
  patterns.add<ConvertWiresOp>(&context);
  patterns.add<ConvertControlOp>(&context);
  patterns.add<ConvertCellOp>(&context);
  patterns.add<ConvertAssignOp>(&context);

  return applyPartialConversion(program, target, std::move(patterns));
}

std::unique_ptr<mlir::Pass> circt::createCalyxToHWPass() {
  return std::make_unique<CalyxToHWPass>();
}
