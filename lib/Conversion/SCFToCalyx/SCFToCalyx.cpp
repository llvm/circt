//===- SCFToCalyx.cpp - SCF to Calyx pass entry point -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SCF to Calyx conversion pass implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SCFToCalyx/SCFToCalyx.h"
#include "../PassDetail.h"
#include "circt/Dialect/Calyx/CalyxOps.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include <variant>

using namespace llvm;
using namespace mlir;

namespace circt {

struct ModuleOpConversion : public OpRewritePattern<mlir::ModuleOp> {
  ModuleOpConversion(MLIRContext *context, calyx::ProgramOp *programOpOutput)
      : OpRewritePattern<mlir::ModuleOp>(context),
        programOpOutput(programOpOutput) {
    assert(programOpOutput->getOperation() == nullptr &&
           "this function will set programOpOutput post module conversion");
  }

  LogicalResult matchAndRewrite(mlir::ModuleOp moduleOp,
                                PatternRewriter &rewriter) const override {
    if (!moduleOp.getOps<calyx::ProgramOp>().empty())
      return failure();

    rewriter.updateRootInPlace(moduleOp, [&] {
      // Create ProgramOp
      rewriter.setInsertionPointAfter(moduleOp);
      auto programOp = rewriter.create<calyx::ProgramOp>(moduleOp.getLoc());

      // Inline the module body region
      rewriter.inlineRegionBefore(moduleOp.getBodyRegion(),
                                  programOp.getBodyRegion(),
                                  programOp.getBodyRegion().end());

      // Inlining the body region also removes ^bb0 from the module body
      // region, so recreate that, before finally inserting the programOp
      auto moduleBlock = rewriter.createBlock(&moduleOp.getBodyRegion());
      rewriter.setInsertionPointToStart(moduleBlock);
      rewriter.insert(programOp);
      *programOpOutput = programOp;
    });
    return success();
  }

private:
  calyx::ProgramOp *programOpOutput = nullptr;
};

//===----------------------------------------------------------------------===//
// Pass driver
//===----------------------------------------------------------------------===//
class SCFToCalyxPass : public SCFToCalyxBase<SCFToCalyxPass> {
public:
  SCFToCalyxPass() : SCFToCalyxBase<SCFToCalyxPass>() {}
  void runOnOperation() override;

  LogicalResult mainFuncIsDefined(mlir::ModuleOp moduleOp,
                                  StringRef topLevelFunction) {
    if (SymbolTable::lookupSymbolIn(moduleOp, topLevelFunction) == nullptr) {
      moduleOp.emitError("Main function '" + topLevelFunction +
                         "' not found in module.");
      return failure();
    }
    return success();
  }

  //// Creates a new Calyx program with the contents of the source module
  /// inlined within.
  /// Furthermore, this function performs validation on the input function, to
  /// ensure that we've implemented the capabilities necessary to convert it.
  LogicalResult createProgram(calyx::ProgramOp *programOpOut) {
    auto createModuleConvTarget = [&]() {
      ConversionTarget target(getContext());
      target.addLegalDialect<calyx::CalyxDialect>();
      target.addLegalDialect<scf::SCFDialect>();
      target.addIllegalDialect<hw::HWDialect>();
      target.addIllegalDialect<comb::CombDialect>();

      // For loops should have been lowered to while loops
      target.addIllegalOp<scf::ForOp>();

      // Only accept std operations which we've added lowerings for
      target.addIllegalDialect<StandardOpsDialect>();
      target.addLegalOp<AddIOp, SubIOp, CmpIOp, ShiftLeftOp,
                        UnsignedShiftRightOp, SignedShiftRightOp, AndOp, XOrOp,
                        OrOp, ZeroExtendIOp, TruncateIOp, CondBranchOp,
                        BranchOp, ReturnOp, ConstantOp, IndexCastOp>();
      return target;
    };

    // Program legalization - the partial conversion driver will not run unless
    // some pattern is provided - provide a dummy pattern.
    struct DummyPattern : public OpRewritePattern<mlir::ModuleOp> {
      using OpRewritePattern::OpRewritePattern;
      LogicalResult matchAndRewrite(mlir::ModuleOp,
                                    PatternRewriter &) const override {
        return failure();
      }
    };
    RewritePatternSet legalizePatterns(&getContext());
    legalizePatterns.add<DummyPattern>(&getContext());
    auto target = createModuleConvTarget();
    DenseSet<Operation *> legalizedOps;
    if (applyPartialConversion(getOperation(), target,
                               std::move(legalizePatterns))
            .failed())
      return failure();

    // Program conversion
    RewritePatternSet patterns(&getContext());
    patterns.add<ModuleOpConversion>(&getContext(), programOpOut);
    return applyOpPatternsAndFold(getOperation(), std::move(patterns));
  }

private:
};

void SCFToCalyxPass::runOnOperation() {
  std::string topLevelFunction =
      topLevelComponent.empty() ? std::string("main") : topLevelComponent;

  if (failed(mainFuncIsDefined(getOperation(), topLevelFunction))) {
    signalPassFailure();
    return;
  }

  /// Start conversion
  calyx::ProgramOp programOp;
  if (failed(createProgram(&programOp))) {
    signalPassFailure();
    return;
  }
  assert(programOp.getOperation() != nullptr &&
         "programOp should have been set during module "
         "conversion, if module conversion succeeded.");
}

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSCFToCalyxPass() {
  return std::make_unique<SCFToCalyxPass>();
}

} // namespace circt
