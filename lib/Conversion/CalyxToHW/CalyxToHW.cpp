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
#include "circt/Dialect/HW/HWTypes.h"
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
    for (auto portInfo : component.getPortInfo())
      hwInputInfo.push_back(
          {portInfo.name, hwDirection(portInfo.direction), portInfo.type});

    auto hwMod = rewriter.create<HWModuleOp>(
        component.getLoc(), component.getNameAttr(), hwInputInfo);

    rewriter.eraseOp(hwMod.getBodyBlock()->getTerminator());
    ConversionPatternRewriter::InsertionGuard g(rewriter);
    rewriter.setInsertionPointToEnd(hwMod.getBodyBlock());

    SmallVector<Value> argValues;
    SmallVector<Value> outputWires;
    size_t portIdx = 0;
    for (auto portInfo : component.getPortInfo()) {
      switch (portInfo.direction) {
      case calyx::Direction::Input:
        assert(hwMod.getArgument(portIdx).getType() == portInfo.type);
        argValues.push_back(hwMod.getArgument(portIdx));
        break;
      case calyx::Direction::Output:
        auto wire = rewriter.create<sv::WireOp>(component.getLoc(),
                                                portInfo.type, portInfo.name);
        auto wireRead =
            rewriter.create<sv::ReadInOutOp>(component.getLoc(), wire);
        argValues.push_back(wireRead);
        outputWires.push_back(wireRead);
        break;
      }
      ++portIdx;
    }

    rewriter.mergeBlocks(component.getBody(), hwMod.getBodyBlock(), argValues);
    rewriter.create<OutputOp>(component.getLoc(), outputWires);
    rewriter.eraseOp(component);

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
    auto dest = adaptor.dest();

    // To make life easy in ConvertComponentOp, we read from the output wires so
    // the dialect conversion block argument mapping would work without a type
    // converter. This means assigns to ComponentOp outputs will try to assign
    // to a read from a wire, so we need to map to the wire.
    if (auto readInOut = dyn_cast<ReadInOutOp>(adaptor.dest().getDefiningOp()))
      dest = readInOut.input();

    rewriter.replaceOpWithNewOp<sv::AssignOp>(assign, dest, adaptor.src());

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
