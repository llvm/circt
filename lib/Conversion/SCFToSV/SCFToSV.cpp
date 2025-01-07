//===- SCFToSV.cpp - Sim to SV lowering -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/SCFToSV.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "lower-scf-to-sv"

namespace circt {
#define GEN_PASS_DEF_SCFTOSV
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace mlir;

struct IfOpConversionPattern : public OpConversionPattern<scf::IfOp> {
  using OpConversionPattern<scf::IfOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const final {
    if (!llvm::all_of(op.getResults(), [](auto res) {
          return llvm::isa<IntegerType>(res.getType());
        })) {
      op.emitError("Cannot convert non-integer result to SV");
      return failure();
    }

    if (adaptor.getThenRegion().getBlocks().size() > 1 ||
        adaptor.getElseRegion().getBlocks().size() > 1) {
      op.emitError("Cannot convert scf.if region with more than one block.");
      return failure();
    }

    SmallVector<sv::LogicOp> resultDecls;
    auto assignResults = [&](scf::YieldOp yieldOp) {
      assert(resultDecls.size() == yieldOp.getNumOperands());
      for (auto [decl, yield] : llvm::zip(resultDecls, yieldOp.getOperands()))
        rewriter.create<sv::BPAssignOp>(yieldOp.getLoc(), decl.getResult(),
                                        yield);
    };

    for (auto res : op.getResults())
      resultDecls.push_back(
          rewriter.create<sv::LogicOp>(op.getLoc(), res.getType()));

    auto svIfOp =
        rewriter.create<sv::IfOp>(op.getLoc(), adaptor.getCondition());
    if (!adaptor.getThenRegion().empty()) {
      auto yield =
          cast<scf::YieldOp>(adaptor.getThenRegion().front().getTerminator());
      rewriter.mergeBlocks(&adaptor.getThenRegion().front(),
                           svIfOp.getThenBlock());
      rewriter.setInsertionPointToEnd(svIfOp.getThenBlock());
      assignResults(yield);
      rewriter.eraseOp(yield);
    }

    if (!adaptor.getElseRegion().empty()) {
      auto yield =
          cast<scf::YieldOp>(adaptor.getElseRegion().front().getTerminator());
      auto dest = rewriter.createBlock(&svIfOp.getElseRegion());
      rewriter.eraseOp(adaptor.getElseRegion().front().getTerminator());
      rewriter.mergeBlocks(&adaptor.getElseRegion().front(), dest);
      rewriter.setInsertionPointToEnd(svIfOp.getElseBlock());
      assignResults(yield);
      rewriter.eraseOp(yield);
    }

    SmallVector<Value> reads;
    for (auto decl : resultDecls)
      reads.push_back(
          rewriter.createOrFold<sv::ReadInOutOp>(op.getLoc(), decl));

    rewriter.replaceOp(op, reads);
    return success();
  }
};

namespace {
struct SCFToSVPass : public circt::impl::SCFToSVBase<SCFToSVPass> {
  void runOnOperation() override {
    auto context = &getContext();
    ConversionTarget target(*context);
    RewritePatternSet patterns(context);
    target.addIllegalDialect<scf::SCFDialect>();
    target.addLegalDialect<sv::SVDialect>();
    patterns.add<IfOpConversionPattern>(context);

    auto result =
        applyPartialConversion(getOperation(), target, std::move(patterns));
    if (failed(result))
      signalPassFailure();
  }
};
} // namespace
