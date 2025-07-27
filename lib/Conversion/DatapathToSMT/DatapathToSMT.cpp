//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/DatapathToSMT.h"
#include "circt/Conversion/HWToSMT.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "mlir/Dialect/SMT/IR/SMTOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
#define GEN_PASS_DEF_CONVERTDATAPATHTOSMT
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace datapath;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

// Lower to an SMT assertion that summing the results is equivalent to summing
// the compress inputs
// d:2 = compress(a, b, c) ->
// assert(d#0 + d#1 == a + b + c)
struct CompressOpConversion : OpConversionPattern<CompressOp> {
  using OpConversionPattern<CompressOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ValueRange operands = adaptor.getOperands();
    ValueRange results = op.getResults();

    // Sum operands
    Value operandRunner = operands[0];
    for (Value operand : operands.drop_front())
      operandRunner =
          smt::BVAddOp::create(rewriter, op.getLoc(), operandRunner, operand);

    // Create free variables
    SmallVector<Value, 2> newResults;
    newResults.reserve(results.size());
    for (Value result : results) {
      auto declareFunOp = smt::DeclareFunOp::create(
          rewriter, op.getLoc(), typeConverter->convertType(result.getType()));
      newResults.push_back(declareFunOp.getResult());
    }

    // Sum the free variables
    Value resultRunner = newResults.front();
    for (auto freeVar : llvm::drop_begin(newResults, 1))
      resultRunner =
          smt::BVAddOp::create(rewriter, op.getLoc(), resultRunner, freeVar);

    // Assert sum operands == sum results (free variables)
    auto premise =
        smt::EqOp::create(rewriter, op.getLoc(), operandRunner, resultRunner);
    // Encode via an assertion (could be relaxed to an assumption).
    smt::AssertOp::create(rewriter, op.getLoc(), premise);

    if (newResults.size() != results.size())
      return rewriter.notifyMatchFailure(op, "expected same number of results");

    rewriter.replaceOp(op, newResults);
    return success();
  }
};

// Lower to an SMT assertion that summing the results is equivalent to the
// product of the partial_product inputs
// c:<N> = partial_product(a, b) ->
// assert(c#0 + ... + c#<N-1> == a * b)
struct PartialProductOpConversion : OpConversionPattern<PartialProductOp> {
  using OpConversionPattern<PartialProductOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(PartialProductOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    ValueRange operands = adaptor.getOperands();
    ValueRange results = op.getResults();

    // Multiply the operands
    auto mulResult =
        smt::BVMulOp::create(rewriter, op.getLoc(), operands[0], operands[1]);

    // Create free variables
    SmallVector<Value, 2> newResults;
    newResults.reserve(results.size());
    for (Value result : results) {
      auto declareFunOp = smt::DeclareFunOp::create(
          rewriter, op.getLoc(), typeConverter->convertType(result.getType()));
      newResults.push_back(declareFunOp.getResult());
    }

    // Sum the free variables
    Value resultRunner = newResults.front();
    for (auto freeVar : llvm::drop_begin(newResults, 1))
      resultRunner =
          smt::BVAddOp::create(rewriter, op.getLoc(), resultRunner, freeVar);

    // Assert product of operands == sum results (free variables)
    auto premise =
        smt::EqOp::create(rewriter, op.getLoc(), mulResult, resultRunner);
    // Encode via an assertion (could be relaxed to an assumption).
    smt::AssertOp::create(rewriter, op.getLoc(), premise);

    if (newResults.size() != results.size())
      return rewriter.notifyMatchFailure(op, "expected same number of results");

    rewriter.replaceOp(op, newResults);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert Datapath to SMT pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertDatapathToSMTPass
    : public circt::impl::ConvertDatapathToSMTBase<ConvertDatapathToSMTPass> {
  void runOnOperation() override;
};
} // namespace

void circt::populateDatapathToSMTConversionPatterns(
    TypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<CompressOpConversion, PartialProductOpConversion>(
      converter, patterns.getContext());
}

void ConvertDatapathToSMTPass::runOnOperation() {
  ConversionTarget target(getContext());
  target.addIllegalDialect<datapath::DatapathDialect>();
  target.addLegalDialect<smt::SMTDialect>();

  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  populateHWToSMTTypeConverter(converter);
  populateDatapathToSMTConversionPatterns(converter, patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
