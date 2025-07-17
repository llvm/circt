//===- DatapathToComb.cpp--------------------------------------------------===//
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
#include "llvm/ADT/PointerUnion.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "datapath-to-comb"

namespace circt {
#define GEN_PASS_DEF_CONVERTDATAPATHTOCOMB
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

// using namespace mlir;
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

struct DatapathCompressOpAddConversion : OpConversionPattern<CompressOp> {
  using OpConversionPattern<CompressOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputs = op.getOperands();
    unsigned width = inputs[0].getType().getIntOrFloatBitWidth();
    auto addOp = rewriter.create<comb::AddOp>(loc, inputs, true);
    auto zeroOp = rewriter.create<hw::ConstantOp>(loc, APInt(width, 0));

    rewriter.replaceOp(op, {addOp, zeroOp});
    return success();
  }
};

struct DatapathCompressOpConversion : OpConversionPattern<CompressOp> {
  using OpConversionPattern<CompressOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto inputs = op.getOperands();
    unsigned width = inputs[0].getType().getIntOrFloatBitWidth();
    // TODO - implement a more efficient compression algorithm to compete with
    // yosys's `compress` pass.

    auto falseValue = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));
    SmallVector<SmallVector<Value>> partialProducts;
    for (auto input : inputs) {
      partialProducts.push_back(
          extractBits(rewriter, input)); // Extract bits from each input
    }

    // Wallace tree reduction
    rewriter.replaceOp(op, wallaceReduction(falseValue, width, rewriter, loc,
                                            partialProducts));
    return success();
  }

private:
  // Perform Wallace tree reduction on partial products.
  // See https://en.wikipedia.org/wiki/Wallace_tree
  static SmallVector<Value>
  wallaceReduction(Value falseValue, size_t width,
                   ConversionPatternRewriter &rewriter, Location loc,
                   SmallVector<SmallVector<Value>> &partialProducts) {
    SmallVector<SmallVector<Value>> newPartialProducts;
    newPartialProducts.reserve(partialProducts.size());
    // Continue reduction until we have only two rows. The length of
    // `partialProducts` is reduced by 1/3 in each iteration.
    auto numReductionStages = 0;
    while (partialProducts.size() > 2) {
      newPartialProducts.clear();
      ++numReductionStages;
      // Take three rows at a time and reduce to two rows(sum and carry).
      for (unsigned i = 0; i < partialProducts.size(); i += 3) {
        if (i + 2 < partialProducts.size()) {
          // We have three rows to reduce
          auto &row1 = partialProducts[i];
          auto &row2 = partialProducts[i + 1];
          auto &row3 = partialProducts[i + 2];

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

          newPartialProducts.push_back(std::move(sumRow));
          newPartialProducts.push_back(std::move(carryRow));
        } else {
          // Add remaining rows as is
          newPartialProducts.append(partialProducts.begin() + i,
                                    partialProducts.end());
        }
      }

      std::swap(newPartialProducts, partialProducts);
    }

    LLVM_DEBUG(llvm::dbgs() << "Wallace tree reduction completed in "
                            << numReductionStages << " stages\n");

    assert(partialProducts.size() == 2);
    SmallVector<Value> carrySave;
    for (auto partialProduct : partialProducts) {
      // Reverse the order of the bits
      std::reverse(partialProduct.begin(), partialProduct.end());
      carrySave.push_back(rewriter.create<comb::ConcatOp>(loc, partialProduct));
    }
    // Use comb.add for the final addition.
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

    // Extract individual bits from operands
    SmallVector<Value> aBits = extractBits(rewriter, a);
    SmallVector<Value> bBits = extractBits(rewriter, b);

    SmallVector<Value> partialProducts;
    partialProducts.reserve(width);

    // Implement a basic and array
    if (width <= 16)
      lowerAndArray(rewriter, op, partialProducts, aBits, bBits, width);
    else
      lowerBoothArray(rewriter, op, partialProducts, aBits, bBits, width);

    return success();
  }

  void lowerAndArray(ConversionPatternRewriter &rewriter, PartialProductOp op,
                     SmallVector<Value> partialProducts,
                     SmallVector<Value> &aBits, SmallVector<Value> &bBits,
                     unsigned width) const {

    Location loc = op.getLoc();
    Value a = op.getOperand(0);

    // Generate partial products

    for (unsigned i = 0; i < width; ++i) {
      auto repl = rewriter.create<comb::ReplicateOp>(loc, bBits[i], width);
      auto ppRow = rewriter.create<comb::AndOp>(loc, repl, a);
      Value shiftBy = rewriter.create<hw::ConstantOp>(loc, APInt(width, i));
      Value ppAlign = rewriter.create<comb::ShlOp>(loc, ppRow, shiftBy);
      partialProducts.push_back(ppAlign);
      if (partialProducts.size() == op.getNumResults())
        break;
    }
    if (partialProducts.size() != op.getNumResults()) {
      llvm::errs() << "Expected " << op.getNumResults()
                   << " partial products, but got " << partialProducts.size()
                   << " width " << width << "\n";
      assert(false && "Expected width number of booth partial products");
    }
    rewriter.replaceOp(op, partialProducts);
  }

  void lowerBoothArray(ConversionPatternRewriter &rewriter, PartialProductOp op,
                       SmallVector<Value> &partialProducts,
                       SmallVector<Value> &aBits, SmallVector<Value> &bBits,
                       unsigned width) const {
    Location loc = op.getLoc();
    auto zeroFalse = rewriter.create<hw::ConstantOp>(loc, APInt(1, 0));
    auto zeroWidth = rewriter.create<hw::ConstantOp>(loc, APInt(width, 0));
    auto oneWidth = rewriter.create<hw::ConstantOp>(loc, APInt(width, 1));
    Value a = op.getOperand(0);
    Value twoA = rewriter.create<comb::ShlOp>(loc, a, oneWidth);

    // Booth encoding: examine b[i+1:i-1] for each i
    // We need width/2 partial products for radix-2 Booth
    Value cplPrev;
    for (unsigned i = 0; i < width; i += 2) {
      // Get Booth bits: b[i+1], b[i], b[i-1] (b[-1] = 0)
      Value b_i = (i < width) ? bBits[i] : zeroFalse;
      Value b_ip1 = (i + 1 < width) ? bBits[i + 1] : zeroFalse;
      Value b_im1 = (i == 0) ? zeroFalse : bBits[i - 1];

      // Is the encoding zero or negative (an approximation)
      Value cpl = b_ip1;
      // Is the encoding one = b_i xor b_im1
      Value one = rewriter.create<comb::XorOp>(loc, b_i, b_im1, true);
      // Is the encoding two = (b_ip1 & ~b_i & ~b_im1) | (~b_ip1 & b_i & b_im1)
      Value const_one = rewriter.create<hw::ConstantOp>(loc, APInt(1, 1));
      Value b_i_inv = rewriter.create<comb::XorOp>(loc, b_i, const_one, true);
      Value b_ip1_inv =
          rewriter.create<comb::XorOp>(loc, b_ip1, const_one, true);
      Value b_im1_inv =
          rewriter.create<comb::XorOp>(loc, b_im1, const_one, true);
      Value andLeft = rewriter.create<comb::AndOp>(
          loc, ValueRange{b_ip1_inv, b_i, b_im1}, true);
      Value andRight = rewriter.create<comb::AndOp>(
          loc, ValueRange{b_ip1, b_i_inv, b_im1_inv}, true);
      Value two = rewriter.create<comb::OrOp>(loc, andLeft, andRight, true);

      Value cpl_repl = rewriter.create<comb::ReplicateOp>(loc, cpl, width);
      Value one_repl = rewriter.create<comb::ReplicateOp>(loc, one, width);
      Value two_repl = rewriter.create<comb::ReplicateOp>(loc, two, width);

      // Select between 2*a or 1*a or 0*a
      Value selTwoA = rewriter.create<comb::AndOp>(loc, two_repl, twoA);
      Value selOneA = rewriter.create<comb::AndOp>(loc, one_repl, a);
      Value magA = rewriter.create<comb::OrOp>(loc, selTwoA, selOneA, true);

      // Conditionally invert the row
      Value ppRow = rewriter.create<comb::XorOp>(loc, magA, cpl_repl, true);
      if (i == 0) {
        partialProducts.push_back(ppRow);
        cplPrev = cpl;
        continue;
      }
      assert(i >= 2 && "Expected i to be at least 2 for sign correction");
      Value withSignCorrection = rewriter.create<comb::ConcatOp>(
          loc, ValueRange{ppRow, zeroFalse, cplPrev});
      Value ppAlignPre =
          rewriter.create<comb::ExtractOp>(loc, withSignCorrection, 0, width);
      Value shiftBy = rewriter.create<hw::ConstantOp>(loc, APInt(width, i - 2));
      Value ppAlign = rewriter.create<comb::ShlOp>(loc, ppAlignPre, shiftBy);
      partialProducts.push_back(ppAlign);
      cplPrev = cpl;

      if (partialProducts.size() == op.getNumResults())
        break;
    }

    while (partialProducts.size() < op.getNumResults())
      partialProducts.push_back(zeroWidth);

    if (partialProducts.size() != op.getNumResults()) {
      llvm::errs() << "Expected " << op.getNumResults()
                   << " partial products, but got " << partialProducts.size()
                   << "width " << width << "\n";
      assert(false && "Expected width number of booth partial products");
    }
    assert(partialProducts.size() == op.getNumResults() &&
           "Expected width number of booth partial products");

    rewriter.replaceOp(op, partialProducts);
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Convert Comb to Datapath pass
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
    patterns.add<DatapathCompressOpAddConversion>(patterns.getContext());
  else
    patterns.add<DatapathCompressOpConversion>(patterns.getContext());
}

void ConvertDatapathToCombPass::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<comb::CombDialect, hw::HWDialect>();
  target.addIllegalDialect<datapath::DatapathDialect>();

  RewritePatternSet patterns(&getContext());
  populateDatapathToCombConversionPatterns(patterns, lowerCompressToAdd);

  if (failed(mlir::applyPartialConversion(getOperation(), target,
                                          std::move(patterns))))
    return signalPassFailure();
}
