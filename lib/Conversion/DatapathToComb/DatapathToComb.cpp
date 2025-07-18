//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/DatapathToComb.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/Datapath/DatapathOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "datapath-to-comb"

namespace circt {
#define GEN_PASS_DEF_CONVERTDATAPATHTOCOMB
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace circt;
using namespace datapath;

// A wrapper for comb::extractBits that returns a SmallVector<Value>.
static SmallVector<Value> extractBits(OpBuilder &builder, Value val) {
  SmallVector<Value> bits;
  comb::extractBits(builder, val, bits);
  return bits;
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
// Construct a full adder for three 1-bit inputs.
std::pair<Value, Value> fullAdder(ConversionPatternRewriter &rewriter,
                                  Location loc, Value a, Value b, Value c) {
  auto aXorB = rewriter.createOrFold<comb::XorOp>(loc, a, b, true);
  // a ^ b ^ c
  Value sum = rewriter.createOrFold<comb::XorOp>(loc, aXorB, c, true);
  // (a & b) | ((a ^ b) & c)
  auto carry = rewriter.createOrFold<comb::OrOp>(
      loc,
      ArrayRef<Value>{rewriter.createOrFold<comb::AndOp>(loc, a, b, true),
                      rewriter.createOrFold<comb::AndOp>(loc, aXorB, c, true)},
      true);

  return {sum, carry};
}

// Replace compressor by an adder of the inputs and zero for the other results:
// compress(a,b,c,d) -> {a+b+c+d, 0}
// Facilitates use of downstream compression algorithms e.g. Yosys
struct DatapathCompressOpAddConversion : OpConversionPattern<CompressOp> {
  using OpConversionPattern<CompressOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputs = op.getOperands();
    unsigned width = inputs[0].getType().getIntOrFloatBitWidth();
    // Sum all the inputs - set that to result value 0
    auto addOp = rewriter.create<comb::AddOp>(loc, inputs, true);
    // Replace remaining results with zeros
    auto zeroOp = rewriter.create<hw::ConstantOp>(loc, APInt(width, 0));
    SmallVector<Value> results(op.getNumResults() - 1, zeroOp);
    results.push_back(addOp);
    rewriter.replaceOp(op, results);
    return success();
  }
};

// Replace compressor by a wallace tree of full-adders
struct DatapathCompressOpConversion : OpConversionPattern<CompressOp> {
  using OpConversionPattern<CompressOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputs = op.getOperands();
    unsigned width = inputs[0].getType().getIntOrFloatBitWidth();

    SmallVector<SmallVector<Value>> addends;
    for (auto input : inputs) {
      addends.push_back(
          extractBits(rewriter, input)); // Extract bits from each input
    }

    // Wallace tree reduction
    // TODO - implement a more efficient compression algorithm to compete with
    // yosys's `alumacc` lowering - a coarse grained timing model would help to
    // sort the inputs according to arrival time.
    auto falseValue = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));
    auto targetAddends = op.getNumResults();
    rewriter.replaceOp(op, wallaceReduction(falseValue, width, targetAddends,
                                            rewriter, loc, addends));
    return success();
  }

private:
  // Perform Wallace tree reduction on partial products.
  // See https://en.wikipedia.org/wiki/Wallace_tree
  static SmallVector<Value>
  wallaceReduction(Value falseValue, size_t width, size_t targetAddends,
                   ConversionPatternRewriter &rewriter, Location loc,
                   SmallVector<SmallVector<Value>> &addends) {
    SmallVector<SmallVector<Value>> newAddends;
    newAddends.reserve(addends.size());
    // Continue reduction until we have only two rows. The length of
    // `addends` is reduced by 1/3 in each iteration.
    auto numReductionStages = 0;
    while (addends.size() > targetAddends) {
      newAddends.clear();
      ++numReductionStages;
      // Take three rows at a time and reduce to two rows(sum and carry).
      for (unsigned i = 0; i < addends.size(); i += 3) {
        if (i + 2 < addends.size()) {
          // We have three rows to reduce
          auto &row1 = addends[i];
          auto &row2 = addends[i + 1];
          auto &row3 = addends[i + 2];

          assert(row1.size() == width && row2.size() == width &&
                 row3.size() == width);

          SmallVector<Value> sumRow, carryRow;
          sumRow.reserve(width);
          carryRow.reserve(width);
          carryRow.push_back(falseValue);

          // Process each bit position
          for (unsigned j = 0; j < width; ++j) {
            // Full adder logic
            auto [sum, carry] =
                fullAdder(rewriter, loc, row1[j], row2[j], row3[j]);
            sumRow.push_back(sum);
            if (j + 1 < width)
              carryRow.push_back(carry);
          }

          newAddends.push_back(std::move(sumRow));
          newAddends.push_back(std::move(carryRow));
        } else {
          // Add remaining rows as is
          newAddends.append(addends.begin() + i, addends.end());
        }
      }
      std::swap(newAddends, addends);
    }

    LLVM_DEBUG(llvm::dbgs() << "Wallace tree reduction completed in "
                            << numReductionStages << " stages\n");

    assert(addends.size() <= targetAddends);
    SmallVector<Value> carrySave;
    for (auto addend : addends) {
      // Reverse the order of the bits
      std::reverse(addend.begin(), addend.end());
      carrySave.push_back(rewriter.create<comb::ConcatOp>(loc, addend));
    }

    // Pad with zeros
    auto zero = rewriter.create<hw::ConstantOp>(loc, APInt(width, 0));
    while (carrySave.size() < targetAddends)
      carrySave.push_back(zero);

    return carrySave;
  }
};

struct DatapathPartialProductOpConversion
    : OpConversionPattern<PartialProductOp> {
  using OpConversionPattern<PartialProductOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(PartialProductOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto inputs = op.getOperands();
    Value a = inputs[0];
    Value b = inputs[1];
    unsigned width = a.getType().getIntOrFloatBitWidth();

    // Skip a zero width value.
    if (width == 0) {
      rewriter.replaceOpWithNewOp<hw::ConstantOp>(op, op.getType(0), 0);
      return success();
    }

    // Use width as a heuristic to guide partial product implementation
    if (width <= 16)
      return lowerAndArray(rewriter, a, b, op, width);
    else
      return lowerBoothArray(rewriter, a, b, op, width);
  }

private:
  static LogicalResult lowerAndArray(ConversionPatternRewriter &rewriter,
                                     Value a, Value b, PartialProductOp op,
                                     unsigned width) {

    Location loc = op.getLoc();
    // Keep a as a bitvector - multiply by each digit of b
    SmallVector<Value> bBits = extractBits(rewriter, b);

    SmallVector<Value> partialProducts;
    partialProducts.reserve(width);
    // AND Array Construction:
    // partialProducts[i] = ({b[i],..., b[i]} & a) << i
    assert(op.getNumResults() <= width &&
           "Cannot return more results than the operator width");

    for (unsigned i = 0; i < op.getNumResults(); ++i) {
      auto repl = rewriter.create<comb::ReplicateOp>(loc, bBits[i], width);
      auto ppRow = rewriter.create<comb::AndOp>(loc, repl, a);
      auto shiftBy = rewriter.create<hw::ConstantOp>(loc, APInt(width, i));
      auto ppAlign = rewriter.create<comb::ShlOp>(loc, ppRow, shiftBy);
      partialProducts.push_back(ppAlign);
    }

    rewriter.replaceOp(op, partialProducts);
    return success();
  }

  static LogicalResult lowerBoothArray(ConversionPatternRewriter &rewriter,
                                       Value a, Value b, PartialProductOp op,
                                       unsigned width) {
    Location loc = op.getLoc();
    auto zeroFalse = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));
    auto zeroWidth = rewriter.create<hw::ConstantOp>(loc, APInt(width, 0));
    auto oneWidth = rewriter.create<hw::ConstantOp>(loc, APInt(width, 1));
    Value twoA = rewriter.create<comb::ShlOp>(loc, a, oneWidth);

    SmallVector<Value> bBits = extractBits(rewriter, b);

    SmallVector<Value> partialProducts;
    partialProducts.reserve(width);

    // Booth encoding halves array height by grouping three bits at a time:
    // partialProducts[i] = a * (-2*b[2*i+1] + b[2*i] + b[2*i-1]) << 2*i
    // encNeg \approx (-2*b[2*i+1] + b[2*i] + b[2*i-1]) <= 0
    // encOne = (-2*b[2*i+1] + b[2*i] + b[2*i-1]) == +/- 1
    // encTwo = (-2*b[2*i+1] + b[2*i] + b[2*i-1]) == +/- 2
    Value encNegPrev;
    for (unsigned i = 0; i < width; i += 2) {
      // Get Booth bits: b[i+1], b[i], b[i-1] (b[-1] = 0)
      Value bim1 = (i == 0) ? zeroFalse : bBits[i - 1];
      Value bi = (i < width) ? bBits[i] : zeroFalse;
      Value bip1 = (i + 1 < width) ? bBits[i + 1] : zeroFalse;

      // Is the encoding zero or negative (an approximation)
      Value encNeg = bip1;
      // Is the encoding one = b[i] xor b[i-1]
      Value encOne = rewriter.create<comb::XorOp>(loc, bi, bim1, true);
      // Is the encoding two = (bip1 & ~bi & ~bim1) | (~bip1 & bi & bim1)
      Value constOne = rewriter.create<hw::ConstantOp>(loc, APInt(1, 1));
      Value biInv = rewriter.create<comb::XorOp>(loc, bi, constOne, true);
      Value bip1Inv = rewriter.create<comb::XorOp>(loc, bip1, constOne, true);
      Value bim1Inv = rewriter.create<comb::XorOp>(loc, bim1, constOne, true);

      Value andLeft = rewriter.create<comb::AndOp>(
          loc, ValueRange{bip1Inv, bi, bim1}, true);
      Value andRight = rewriter.create<comb::AndOp>(
          loc, ValueRange{bip1, biInv, bim1Inv}, true);
      Value encTwo = rewriter.create<comb::OrOp>(loc, andLeft, andRight, true);

      Value encNegRepl = rewriter.create<comb::ReplicateOp>(loc, encNeg, width);
      Value encOneRepl = rewriter.create<comb::ReplicateOp>(loc, encOne, width);
      Value encTwoRepl = rewriter.create<comb::ReplicateOp>(loc, encTwo, width);

      // Select between 2*a or 1*a or 0*a
      Value selTwoA = rewriter.create<comb::AndOp>(loc, encTwoRepl, twoA);
      Value selOneA = rewriter.create<comb::AndOp>(loc, encOneRepl, a);
      Value magA = rewriter.create<comb::OrOp>(loc, selTwoA, selOneA, true);

      // Conditionally invert the row
      Value ppRow = rewriter.create<comb::XorOp>(loc, magA, encNegRepl, true);

      // No sign-correction in the first row
      if (i == 0) {
        partialProducts.push_back(ppRow);
        encNegPrev = encNeg;
        continue;
      }

      // Insert a sign-correction from the previous row
      assert(i >= 2 && "Expected i to be at least 2 for sign correction");
      // {ppRow, 0, encNegPrev} << 2*(i-1)
      Value withSignCorrection = rewriter.create<comb::ConcatOp>(
          loc, ValueRange{ppRow, zeroFalse, encNegPrev});
      Value ppAlignPre =
          rewriter.create<comb::ExtractOp>(loc, withSignCorrection, 0, width);
      Value shiftBy = rewriter.create<hw::ConstantOp>(loc, APInt(width, i - 2));
      Value ppAlign = rewriter.create<comb::ShlOp>(loc, ppAlignPre, shiftBy);
      partialProducts.push_back(ppAlign);
      encNegPrev = encNeg;

      if (partialProducts.size() == op.getNumResults())
        break;
    }

    // Zero-pad to match the required output width
    while (partialProducts.size() < op.getNumResults())
      partialProducts.push_back(zeroWidth);

    assert(partialProducts.size() == op.getNumResults() &&
           "Expected number of booth partial products to match results");

    rewriter.replaceOp(op, partialProducts);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert Datapath to Comb pass
//===----------------------------------------------------------------------===//

namespace {
struct ConvertDatapathToCombPass
    : public impl::ConvertDatapathToCombBase<ConvertDatapathToCombPass> {
  void runOnOperation() override;
  using ConvertDatapathToCombBase<
      ConvertDatapathToCombPass>::ConvertDatapathToCombBase;
};
} // namespace

static void
populateDatapathToCombConversionPatterns(RewritePatternSet &patterns,
                                         bool lowerCompressToAdd) {
  patterns.add<DatapathPartialProductOpConversion>(patterns.getContext());

  if (lowerCompressToAdd)
    // Lower compressors to simple add operations for downstream optimisations
    patterns.add<DatapathCompressOpAddConversion>(patterns.getContext());
  else
    // Lower compressors to a complete gate-level implementation
    patterns.add<DatapathCompressOpConversion>(patterns.getContext());
}

void ConvertDatapathToCombPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<comb::CombDialect, hw::HWDialect>();
  target.addIllegalDialect<DatapathDialect>();

  RewritePatternSet patterns(&getContext());
  populateDatapathToCombConversionPatterns(patterns, lowerCompressToAdd);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
