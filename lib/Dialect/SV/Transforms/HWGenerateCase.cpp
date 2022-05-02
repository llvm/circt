//===- HWGenerateCase.cpp - Generate case statements ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This pass implements a transformation from a linear mux tree to a case
// statement.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace circt;
using namespace circt::comb;
using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
struct MuxChainCaseOpConverter : public OpRewritePattern<MuxOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(MuxOp op,
                                PatternRewriter &rewriter) const final;
};
} // namespace

LogicalResult
MuxChainCaseOpConverter::matchAndRewrite(MuxOp op,
                                         PatternRewriter &rewriter) const {

  SmallVector<Location> locationsFound;
  SmallVector<std::pair<hw::ConstantOp, Value>, 4> valuesFound;
  Value defaultValue, indexValue;
  if (!getLinearMuxChainsComparison(op, /*isFalseSide*/ true, indexValue,
                                    defaultValue, locationsFound, valuesFound))
    return failure();

  // If the size of mux chains is not large enough to generate case, then stop
  // here.
  if (valuesFound.size() <= 5)
    return failure();

  // Create a register to store in case conditions.
  auto reg = rewriter.create<sv::RegOp>(op.getLoc(), op.getType());
  auto regRead = rewriter.create<sv::ReadInOutOp>(op.getLoc(), reg);

  // If the op is not in a procedural region, then create always_comb.
  if (!op->getParentOp()->hasTrait<sv::ProceduralRegion>()) {
    auto alwaysComb = rewriter.create<sv::AlwaysCombOp>(op.getLoc());
    rewriter.setInsertionPointToEnd(alwaysComb.getBodyBlock());
  }

  using sv::CasePattern;

  auto *context = op.getContext();
  // Create the case itself.
  // NOTE: We have to use `priority` to preserve the semantics of mux chains.
  rewriter.create<sv::CaseOp>(
      FusedLoc::get(context, locationsFound), CaseStmtType::CaseStmt,
      sv::ValidationQualifierTypeEnum::ValidationQualifierPriority, indexValue,
      valuesFound.size() + 1, [&](size_t caseIdx) -> CasePattern {
        // Use a default pattern for the last value, even if we are complete.
        // This avoids tools thinking they need to insert a latch due to
        // potentially incomplete case coverage.
        bool isDefault = caseIdx == valuesFound.size();
        sv::CasePattern thePattern =
            isDefault
                ? CasePattern(indexValue.getType().getIntOrFloatBitWidth(),
                              CasePattern::DefaultPatternTag(), context)
                : CasePattern(valuesFound[caseIdx].first.value(), context);
        rewriter.create<sv::BPAssignOp>(
            op.getLoc(), reg,
            isDefault ? defaultValue : valuesFound[caseIdx].second);
        return thePattern;
      });

  rewriter.replaceOp(op, {regRead});
  return success();
}

struct HWGenerateCasePass : public sv::HWGenerateCaseBase<HWGenerateCasePass> {
  HWGenerateCasePass() {}
  void runOnOperation() override;
};

void HWGenerateCasePass::runOnOperation() {
  auto module = getOperation();
  MLIRContext *ctx = module.getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<MuxChainCaseOpConverter>(ctx);

  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns))))
    markAllAnalysesPreserved();
}

//===----------------------------------------------------------------------===//
// Pass Creation
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::Pass> circt::sv::createHWGenerateCasePass() {
  return std::make_unique<HWGenerateCasePass>();
}
