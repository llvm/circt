//===- IndexSwitchToIf.cpp - Index switch to if-else pass ---*-C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Contains the definitions of the SCF IndexSwitch to If-Else pass.
//
//===----------------------------------------------------------------------===//

#include "circt/Transforms/Passes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLExtras.h"

namespace circt {
#define GEN_PASS_DEF_INDEXSWITCHTOIF
#include "circt/Transforms/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;

struct SwitchToIfConversion : public OpConversionPattern<scf::IndexSwitchOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IndexSwitchOp switchOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = switchOp.getLoc();

    Region &defaultRegion = switchOp.getDefaultRegion();
    bool hasResults = !switchOp.getResultTypes().empty();

    SmallVector<Value> finalResults;
    scf::IfOp prevIfOp = nullptr;

    rewriter.setInsertionPointAfter(switchOp);
    auto switchCases = switchOp.getCases();
    Value switchOperand = adaptor.getArg();
    if (!switchOperand)
      return rewriter.notifyMatchFailure(switchOp,
                                         "missing converted switch operand");
    for (size_t i = 0; i < switchCases.size(); i++) {
      auto caseValueInt = switchCases[i];
      if (prevIfOp)
        rewriter.setInsertionPointToStart(&prevIfOp.getElseRegion().front());

      Value caseValue =
          arith::ConstantIndexOp::create(rewriter, loc, caseValueInt);
      Value cond = arith::CmpIOp::create(
          rewriter, loc, arith::CmpIPredicate::eq, switchOperand, caseValue);

      auto ifOp = scf::IfOp::create(rewriter, loc, switchOp.getResultTypes(),
                                    cond, /*hasElseRegion=*/true);

      Region &caseRegion = switchOp.getCaseRegions()[i];
      rewriter.eraseBlock(&ifOp.getThenRegion().front());
      rewriter.inlineRegionBefore(caseRegion, ifOp.getThenRegion(),
                                  ifOp.getThenRegion().end());

      if (i + 1 == switchCases.size()) {
        rewriter.eraseBlock(&ifOp.getElseRegion().front());
        rewriter.inlineRegionBefore(defaultRegion, ifOp.getElseRegion(),
                                    ifOp.getElseRegion().end());
      }

      if (prevIfOp && hasResults) {
        rewriter.setInsertionPointToEnd(&prevIfOp.getElseRegion().front());
        scf::YieldOp::create(rewriter, loc, ifOp.getResults());
      }

      if (i == 0 && hasResults)
        llvm::append_range(finalResults, ifOp.getResults());

      prevIfOp = ifOp;
    }

    if (hasResults)
      rewriter.replaceOp(switchOp, finalResults);
    else
      rewriter.eraseOp(switchOp);

    return success();
  }
};

namespace {

struct IndexSwitchToIfPass
    : public circt::impl::IndexSwitchToIfBase<IndexSwitchToIfPass> {
public:
  void runOnOperation() override {
    auto *ctx = &getContext();
    RewritePatternSet patterns(ctx);
    ConversionTarget target(*ctx);

    target.addLegalDialect<scf::SCFDialect>();
    target.addLegalDialect<arith::ArithDialect>();
    target.addLegalOp<ModuleOp, func::FuncOp, func::ReturnOp>();
    target.addIllegalOp<scf::IndexSwitchOp>();

    patterns.add<SwitchToIfConversion>(ctx);

    if (applyPartialConversion(getOperation(), target, std::move(patterns))
            .failed()) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

namespace circt {
std::unique_ptr<mlir::Pass> createIndexSwitchToIfPass() {
  return std::make_unique<IndexSwitchToIfPass>();
}
} // namespace circt
