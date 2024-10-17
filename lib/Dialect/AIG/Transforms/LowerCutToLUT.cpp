//===- LowerCutToLUT.cpp --------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass lowers `aig.cut` to `comb.truth_table` with k inputs where
// k is the size of the cut (= operand inputs size).
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/AIG/AIGOps.h"
#include "circt/Dialect/AIG/AIGPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "aig-lower-cut-to-lut"

namespace circt {
namespace aig {
#define GEN_PASS_DEF_LOWERCUTTOLUT
#include "circt/Dialect/AIG/AIGPasses.h.inc"
} // namespace aig
} // namespace circt

using namespace circt;
using namespace aig;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

struct CutToLUTPattern : OpRewritePattern<CutOp> {
  using OpRewritePattern<CutOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CutOp op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
CutToLUTPattern::matchAndRewrite(CutOp cutOp, PatternRewriter &rewriter) const {
  if (cutOp.getNumResults() == 0) {
    rewriter.eraseOp(cutOp);
    return success();
  }

  auto lutWidth = cutOp.getNumOperands();
  // 2. Lower the cut to a LUT. We can get a truth table by evaluating the cut
  // body with every possible combination of the input values.
  uint32_t tableSize = 1 << lutWidth;
  DenseMap<Value, APInt> mapping;
  auto &body = cutOp.getBodyRegion().front();
  for (uint32_t i = 0; i < lutWidth; i++) {
    APInt value(tableSize, 0);
    for (uint32_t j = 0; j < tableSize; j++) {
      // Make sure the order of the bits is correct.
      value.setBitVal(j, (j >> i) & 1);
    }
    mapping[body.getArgument(i)] = std::move(value);
  }

  // Evaluate the cut body. Update `mapping` along the way.
  for (auto &op : body.getOperations()) {
    if (auto constOp = dyn_cast<hw::ConstantOp>(&op)) {
      mapping[constOp.getResult()] =
          APInt(tableSize, constOp.getValue().getZExtValue());
    } else if (auto AndInverterOp = dyn_cast<aig::AndInverterOp>(&op)) {
      // TODO: Avoid this copy.
      SmallVector<APInt> inputs;
      for (auto input : AndInverterOp.getInputs())
        inputs.push_back(mapping[input]);
      mapping[AndInverterOp.getResult()] = AndInverterOp.evaluate(inputs);
    } else if (auto outputOp = dyn_cast<aig::OutputOp>(&op)) {
      assert(outputOp.getOutputs().size() == 1 && "expected single output");
      auto value = mapping.at(outputOp.getOutputs().front());
      LLVM_DEBUG(llvm::dbgs() << "value: " << value << "\n");
      SmallVector<bool> bits;
      bits.reserve(tableSize);
      for (uint32_t i = 0; i < tableSize; i++)
        bits.push_back(value[i]);
      auto truthTable = rewriter.create<comb::TruthTableOp>(
          op.getLoc(), cutOp.getOperands(), rewriter.getBoolArrayAttr(bits));
      rewriter.replaceOp(cutOp, truthTable);
      return success();
    } else {
      return op.emitError("unsupported operation in Eval: ") << op;
    }
  }

  // It should not reach here.
  return failure();
}

} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to AIG pass
//===----------------------------------------------------------------------===//

static void populateLowerCutToLUTPatterns(RewritePatternSet &patterns) {
  patterns.add<CutToLUTPattern>(patterns.getContext());
}

namespace {
struct LowerCutToLUTPass : public impl::LowerCutToLUTBase<LowerCutToLUTPass> {
  void runOnOperation() override;
};
} // namespace

void LowerCutToLUTPass::runOnOperation() {
  auto i1Type = IntegerType::get(&getContext(), 1);
  auto result = getOperation().walk([&](aig::CutOp cutOp) -> WalkResult {
    // 1. Check if the cut can be lowered to a LUT.

    // Check if the cut has a single output.
    if (cutOp.getNumResults() != 1)
      return cutOp.emitError("expected single output");

    // Check if every type is i1.
    for (auto operand : cutOp.getOperands()) {
      if (operand.getType() != i1Type)
        return cutOp.emitError("expected i1 type");
    }

    for (auto result : cutOp.getResults()) {
      if (result.getType() != i1Type)
        return cutOp.emitError("expected i1 type");
    }

    uint32_t lutWidth = cutOp.getNumOperands();
    if (lutWidth >= 32)
      return cutOp.emitError("Cut width is too large to fit in a LUT");

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return signalPassFailure();

  ConversionTarget target(getContext());
  target.addLegalDialect<comb::CombDialect>();
  target.addIllegalOp<aig::CutOp>();

  RewritePatternSet patterns(&getContext());
  populateLowerCutToLUTPatterns(patterns);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}

std::unique_ptr<mlir::Pass> aig::createLowerCutToLUTPass() {
  return std::make_unique<LowerCutToLUTPass>();
}
