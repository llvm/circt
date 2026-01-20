//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SimplifyTruthTable pass, which simplifies truth
// tables that depend on one or fewer inputs.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Comb/CombPasses.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Support/Naming.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"
#include "llvm/Support/LogicalResult.h"

using namespace circt;
using namespace comb;

namespace circt {
namespace comb {
#define GEN_PASS_DEF_SIMPLIFYTRUTHTABLE
#include "circt/Dialect/Comb/Passes.h.inc"
} // namespace comb
} // namespace circt

namespace {

// Helper to check if operation is trivially recursive
static bool isOpTriviallyRecursive(Operation *op) {
  return llvm::any_of(op->getOperands(), [op](auto operand) {
    return operand.getDefiningOp() == op;
  });
}

// Pattern to simplify truth tables
struct SimplifyTruthTable : public OpRewritePattern<TruthTableOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TruthTableOp op,
                                PatternRewriter &rewriter) const override {
    if (isOpTriviallyRecursive(op))
      return failure();

    const auto inputs = op.getInputs();
    const auto table = op.getLookupTable();
    size_t numInputs = inputs.size();
    size_t tableSize = table.size();

    if (numInputs <= 1)
      return failure();

    // Check if all table entries are the same (constant output)
    bool allSame = llvm::all_equal(table);
    if (allSame) {
      bool firstValue = table[0];
      auto constOp =
          hw::ConstantOp::create(rewriter, op.getLoc(), APInt(1, firstValue));
      replaceOpAndCopyNamehint(rewriter, op, constOp);
      return success();
    }

    // Detect if the truth table depends only on one of the inputs.
    // For each input bit, we test whether flipping only that input bit changes
    // the output value of the truth table at any point.
    SmallVector<bool> dependsOn(numInputs, false);
    int dependentInput = -1;
    unsigned numDependencies = 0;

    for (size_t idx = 0; idx < tableSize; ++idx) {
      bool currentValue = table[idx];

      for (size_t bitPos = 0; bitPos < numInputs; ++bitPos) {
        // Skip if we already know this input matters
        if (dependsOn[bitPos])
          continue;

        // Calculate the index of the entry with the bit in question flipped
        size_t bitPositionInTable = numInputs - 1 - bitPos;
        size_t flippedIdx = idx ^ (1ull << bitPositionInTable);
        bool flippedValue = table[flippedIdx];

        // If flipping this bit changes the output, this input is a dependency
        if (currentValue != flippedValue) {
          dependsOn[bitPos] = true;
          dependentInput = bitPos;
          numDependencies++;

          // Exit early if we already found more than one dependency
          if (numDependencies > 1)
            break;
        }
      }

      // Exit early from outer loop if we found more than one dependency
      if (numDependencies > 1)
        break;
    }

    // Only simplify if exactly one input dependency found
    if (numDependencies != 1)
      return failure();

    // Determine if the truth table is identity or inverted by checking the
    // output when the dependent input is 1 (all other inputs at 0)
    size_t bitPositionInTable = numInputs - 1 - dependentInput;
    size_t idxWhen1 = 1ull << bitPositionInTable;
    bool isIdentity = table[idxWhen1];

    // Replace with the input or a simpler truth table for negation
    Value input = inputs[dependentInput];
    if (isIdentity) {
      // Identity case: just replace with the input directly
      replaceOpAndCopyNamehint(rewriter, op, input);
    } else {
      // Inverted case: replace with a single-input truth table for negation
      // This avoids introducing comb.xor, which is useful for LUT mapping
      replaceOpWithNewOpAndCopyNamehint<TruthTableOp>(
          rewriter, op, ValueRange{input}, ArrayRef<bool>{true, false});
    }
    return success();
  }
};

class SimplifyTruthTablePass
    : public impl::SimplifyTruthTableBase<SimplifyTruthTablePass> {
public:
  using SimplifyTruthTableBase::SimplifyTruthTableBase;
  void runOnOperation() override;
};

} // namespace

void SimplifyTruthTablePass::runOnOperation() {
  Operation *op = getOperation();
  MLIRContext *context = op->getContext();
  RewritePatternSet patterns(context);
  patterns.add<SimplifyTruthTable>(context);
  walkAndApplyPatterns(op, std::move(patterns));
}
