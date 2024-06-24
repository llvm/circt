//===- VerifToSMT.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/VerifToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/SMT/SMTOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTVERIFTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a verif::AssertOp operation with an i1 operand to a smt::AssertOp.
struct VerifAssertOpConversion : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value cond = typeConverter->materializeTargetConversion(
        rewriter, op.getLoc(), smt::BoolType::get(getContext()),
        adaptor.getProperty());
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, cond);
    return success();
  }
};

/// Lower a verif::LecOp operation to a miter circuit encoded in SMT.
/// More information on miter circuits can be found, e.g., in this paper:
/// Brand, D., 1993, November. Verification of large synthesized designs. In
/// Proceedings of 1993 International Conference on Computer Aided Design
/// (ICCAD) (pp. 534-537). IEEE.
struct LogicEquivalenceCheckingOpConversion
    : OpConversionPattern<verif::LogicEquivalenceCheckingOp> {
  using OpConversionPattern<
      verif::LogicEquivalenceCheckingOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::LogicEquivalenceCheckingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto *firstOutputs = adaptor.getFirstCircuit().front().getTerminator();
    auto *secondOutputs = adaptor.getSecondCircuit().front().getTerminator();

    if (firstOutputs->getNumOperands() == 0) {
      // Trivially equivalent
      Value trueVal =
          rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
      rewriter.replaceOp(op, trueVal);
      return success();
    }

    smt::SolverOp solver =
        rewriter.create<smt::SolverOp>(loc, rewriter.getI1Type(), ValueRange{});
    rewriter.createBlock(&solver.getBodyRegion());

    // First, convert the block arguments of the miter bodies.
    if (failed(rewriter.convertRegionTypes(&adaptor.getFirstCircuit(),
                                           *typeConverter)))
      return failure();
    if (failed(rewriter.convertRegionTypes(&adaptor.getSecondCircuit(),
                                           *typeConverter)))
      return failure();

    // Second, create the symbolic values we replace the block arguments with
    SmallVector<Value> inputs;
    for (auto arg : adaptor.getFirstCircuit().getArguments())
      inputs.push_back(rewriter.create<smt::DeclareFunOp>(loc, arg.getType()));

    // Third, inline the blocks
    // Note: the argument value replacement does not happen immediately, but
    // only after all the operations are already legalized.
    // Also, it has to be ensured that the original argument type and the type
    // of the value with which is is to be replaced match. The value is looked
    // up (transitively) in the replacement map at the time the replacement
    // pattern is committed.
    rewriter.mergeBlocks(&adaptor.getFirstCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.mergeBlocks(&adaptor.getSecondCircuit().front(), solver.getBody(),
                         inputs);
    rewriter.setInsertionPointToEnd(solver.getBody());

    // Fourth, convert the yielded values back to the source type system (since
    // the operations of the inlined blocks will be converted by other patterns
    // later on and we should make sure the IR is well-typed after each pattern
    // application), and build the 'assert'.
    SmallVector<Value> outputsDifferent;
    for (auto [out1, out2] :
         llvm::zip(firstOutputs->getOperands(), secondOutputs->getOperands())) {
      Value o1 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out1);
      Value o2 = typeConverter->materializeTargetConversion(
          rewriter, loc, typeConverter->convertType(out1.getType()), out2);
      outputsDifferent.emplace_back(
          rewriter.create<smt::DistinctOp>(loc, o1, o2));
    }

    rewriter.eraseOp(firstOutputs);
    rewriter.eraseOp(secondOutputs);

    Value toAssert;
    if (outputsDifferent.size() == 1)
      toAssert = outputsDifferent[0];
    else
      toAssert = rewriter.create<smt::OrOp>(loc, outputsDifferent);

    rewriter.create<smt::AssertOp>(loc, toAssert);

    // Fifth, check for satisfiablility and report the result back.
    Value falseVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(false));
    Value trueVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getBoolAttr(true));
    auto checkOp = rewriter.create<smt::CheckOp>(loc, rewriter.getI1Type());
    rewriter.createBlock(&checkOp.getSatRegion());
    rewriter.create<smt::YieldOp>(loc, falseVal);
    rewriter.createBlock(&checkOp.getUnknownRegion());
    rewriter.create<smt::YieldOp>(loc, falseVal);
    rewriter.createBlock(&checkOp.getUnsatRegion());
    rewriter.create<smt::YieldOp>(loc, trueVal);
    rewriter.setInsertionPointAfter(checkOp);
    rewriter.create<smt::YieldOp>(loc, checkOp->getResults());

    rewriter.replaceOp(op, solver->getResults());
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Convert Verif to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertVerifToSMTPass
    : public circt::impl::ConvertVerifToSMTBase<ConvertVerifToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateVerifToSMTConversionPatterns(TypeConverter &converter,
                                                 RewritePatternSet &patterns) {
  patterns.add<VerifAssertOpConversion, LogicEquivalenceCheckingOpConversion>(
      converter, patterns.getContext());
}

void ConvertVerifToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<verif::VerifDialect>();
  target.addLegalDialect<smt::SMTDialect, arith::ArithDialect>();
  target.addLegalOp<UnrealizedConversionCastOp>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);
  populateVerifToSMTConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
