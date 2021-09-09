//===- SeqPasses.cpp - Implement Seq passes -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace seq;

namespace circt {
namespace seq {
#define GEN_PASS_CLASSES
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace seq
} // namespace circt

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
  matchAndRewrite(CompRegOp reg, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const final {
    Location loc = reg.getLoc();

    auto svReg = rewriter.create<sv::RegOp>(loc, reg.getResult().getType());
    DictionaryAttr regAttrs = reg->getAttrDictionary();
    if (!regAttrs.empty())
      svReg->setAttrs(regAttrs);
    auto regVal = rewriter.create<sv::ReadInOutOp>(loc, svReg);

    if (reg.reset() && reg.resetValue()) {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.clk(), ResetType::SyncReset,
          sv::EventControl::AtPosEdge, reg.reset(),
          [&]() { rewriter.create<sv::PAssignOp>(loc, svReg, reg.input()); },
          [&]() {
            rewriter.create<sv::PAssignOp>(loc, svReg, reg.resetValue());
          });
    } else {
      rewriter.create<sv::AlwaysFFOp>(
          loc, sv::EventControl::AtPosEdge, reg.clk(),
          [&]() { rewriter.create<sv::PAssignOp>(loc, svReg, reg.input()); });
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

namespace circt {
namespace seq {
std::unique_ptr<OperationPass<ModuleOp>> createSeqLowerToSVPass() {
  return std::make_unique<SeqToSVPass>();
}
} // namespace seq
} // namespace circt

namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Dialect/Seq/SeqPasses.h.inc"
} // namespace

void circt::seq::registerSeqPasses() { registerPasses(); }
