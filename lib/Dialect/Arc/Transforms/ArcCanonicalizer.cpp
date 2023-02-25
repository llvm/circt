//===- ArcCanonicalizer.cpp -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Simulation centric canonicalizations for non-arc operations.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Arc/ArcOps.h"
#include "circt/Dialect/Arc/ArcPasses.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "arc-canonicalizer"

using namespace circt;
using namespace arc;

//===----------------------------------------------------------------------===//
// Canonicalization patterns
//===----------------------------------------------------------------------===//

namespace {

class ZeroCountRaising : public OpRewritePattern<comb::MuxOp> {
public:
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(comb::MuxOp op,
                                PatternRewriter &rewriter) const final;

private:
  using DeltaFunc = std::function<uint32_t(uint32_t, bool)>;
  LogicalResult handleSequenceInitializer(OpBuilder &rewriter, Location loc,
                                          const DeltaFunc &deltaFunc,
                                          bool isLeading, Value falseValue,
                                          Value extractedFrom,
                                          SmallVectorImpl<Value> &arrayElements,
                                          uint32_t &currIndex) const;
};

struct IndexingConstArray : public OpRewritePattern<hw::ArrayGetOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(hw::ArrayGetOp op,
                                PatternRewriter &rewriter) const final;
};

} // namespace

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

static Value zextUsingConcatOp(OpBuilder &builder, Location loc, Value toZext,
                               uint32_t targetWidth) {
  assert(toZext.getType().isSignlessInteger() &&
         "Can only concatenate integers");

  uint32_t bitWidth = toZext.getType().getIntOrFloatBitWidth();
  assert(bitWidth <= targetWidth && "cannot zext to a smaller bitwidth");

  if (bitWidth == targetWidth)
    return toZext;

  Value zero =
      builder.create<hw::ConstantOp>(loc, APInt(targetWidth - bitWidth, 0));
  return builder.create<comb::ConcatOp>(loc, zero, toZext);
}

//===----------------------------------------------------------------------===//
// Canonicalization pattern implementations
//===----------------------------------------------------------------------===//

LogicalResult
IndexingConstArray::matchAndRewrite(hw::ArrayGetOp op,
                                    PatternRewriter &rewriter) const {
  auto constArray = op.getInput().getDefiningOp<hw::AggregateConstantOp>();
  if (!constArray)
    return failure();

  Type elementType = op.getResult().getType();

  if (!elementType.isSignlessInteger())
    return failure();

  uint32_t elementBitWidth = elementType.getIntOrFloatBitWidth();
  uint32_t indexBitWidth = op.getIndex().getType().getIntOrFloatBitWidth();

  if (elementBitWidth < indexBitWidth)
    return failure();

  APInt one(elementBitWidth, 1);
  bool isIdentity = true, isShlOfOne = true;

  for (auto [i, fieldAttr] : llvm::enumerate(constArray.getFields())) {
    APInt elementValue = fieldAttr.cast<IntegerAttr>().getValue();

    if (elementValue != APInt(elementBitWidth, i))
      isIdentity = false;

    if (elementValue != one << APInt(elementBitWidth, i))
      isShlOfOne = false;
  }

  Value optionalZext = op.getIndex();
  if (isIdentity || isShlOfOne)
    optionalZext = zextUsingConcatOp(rewriter, op.getLoc(), op.getIndex(),
                                     elementBitWidth);

  if (isIdentity) {
    rewriter.replaceOp(op, optionalZext);
    return success();
  }

  if (isShlOfOne) {
    Value one =
        rewriter.create<hw::ConstantOp>(op.getLoc(), optionalZext.getType(), 1);
    rewriter.replaceOpWithNewOp<comb::ShlOp>(op, one, optionalZext);
    return success();
  }

  return failure();
}

LogicalResult
ZeroCountRaising::matchAndRewrite(comb::MuxOp op,
                                  PatternRewriter &rewriter) const {
  // We don't want to match on muxes in the middle of a sequence.
  if (llvm::any_of(op.getResult().getUsers(),
                   [](auto user) { return isa<comb::MuxOp>(user); }))
    return failure();

  comb::MuxOp curr = op;
  uint32_t currIndex = -1;
  Value extractedFrom;
  SmallVector<Value> arrayElements;
  std::optional<bool> isLeading = std::nullopt;
  auto deltaFunc = [](uint32_t input, bool isLeading) {
    return isLeading ? --input : ++input;
  };

  while (true) {
    // Muxes not at the end of the sequence must not be used anywhere else as we
    // cannot remove them then.
    if (curr != op && !curr->hasOneUse())
      return failure();

    // We force the condition to be extracts for now as we otherwise have to
    // insert a concat which might be more expensive than what we gain.
    auto ext = curr.getCond().getDefiningOp<comb::ExtractOp>();
    if (!ext)
      return failure();

    if (ext.getResult().getType().getIntOrFloatBitWidth() != 1)
      return failure();

    if (currIndex == -1U)
      extractedFrom = ext.getInput();

    if (extractedFrom != ext.getInput())
      return failure();

    if (currIndex != -1U) {
      if (!isLeading.has_value()) {
        if (ext.getLowBit() == currIndex - 1)
          isLeading = true;
        else if (ext.getLowBit() == currIndex + 1)
          isLeading = false;
        else
          return failure();
      }

      if (ext.getLowBit() != deltaFunc(currIndex, *isLeading))
        return failure();
    }

    currIndex = ext.getLowBit();

    arrayElements.push_back(curr.getTrueValue());
    Value falseValue = curr.getFalseValue();

    curr = curr.getFalseValue().getDefiningOp<comb::MuxOp>();
    if (!curr) {
      // Check for init value patterns
      if (failed(handleSequenceInitializer(
              rewriter, op.getLoc(), deltaFunc, isLeading.value(), falseValue,
              extractedFrom, arrayElements, currIndex)))
        arrayElements.push_back(falseValue);

      break;
    }
  }

  Value extForLzc = rewriter.create<comb::ExtractOp>(
      op.getLoc(), extractedFrom,
      *isLeading ? currIndex : ((int)currIndex - (int)arrayElements.size() + 2),
      arrayElements.size() - 1);
  Value lcz = rewriter.create<arc::ZeroCountOp>(
      op.getLoc(), extForLzc,
      *isLeading ? ZeroCountPredicate::leading : ZeroCountPredicate::trailing);
  Value arrayIndex = rewriter.create<comb::ExtractOp>(
      op.getLoc(), lcz, 0, llvm::Log2_64_Ceil(arrayElements.size()));
  Value array = rewriter.create<hw::ArrayCreateOp>(op.getLoc(), arrayElements);
  rewriter.replaceOpWithNewOp<hw::ArrayGetOp>(op, array, arrayIndex);

  return success();
}

LogicalResult ZeroCountRaising::handleSequenceInitializer(
    OpBuilder &rewriter, Location loc, const DeltaFunc &deltaFunc,
    bool isLeading, Value falseValue, Value extractedFrom,
    SmallVectorImpl<Value> &arrayElements, uint32_t &currIndex) const {
  if (auto concat = falseValue.getDefiningOp<comb::ConcatOp>()) {
    if (concat.getInputs().size() != 2) {
      arrayElements.push_back(falseValue);
      return failure();
    }
    Value nonConstant;
    if (auto constAllSet = concat.getOperand(0).getDefiningOp<hw::ConstantOp>())
      nonConstant = concat.getOperand(1);
    else if (auto constAllSet =
                 concat.getOperand(1).getDefiningOp<hw::ConstantOp>())
      nonConstant = concat.getOperand(0);
    else
      return failure();

    Value indirection = nonConstant;
    bool negated = false;
    if (auto xorOp = nonConstant.getDefiningOp<comb::XorOp>();
        xorOp && xorOp.isBinaryNot()) {
      indirection = xorOp.getOperand(0);
      negated = true;
    }

    auto ext = indirection.getDefiningOp<comb::ExtractOp>();
    if (ext.getInput() == extractedFrom &&
        ext.getResult().getType().getIntOrFloatBitWidth() == 1 &&
        ext.getLowBit() == deltaFunc(currIndex, isLeading)) {
      currIndex = ext.getLowBit();
      Value zero =
          rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 0);
      Value one =
          rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), -1);
      // Value stoppers = rewriter.create<comb::OrOp>(op.getLoc(),
      // extractedFrom, c);
      IRMapping mapping;
      mapping.map(nonConstant, zero);
      auto *clonedZeroConcat = rewriter.clone(*concat, mapping);
      mapping.map(nonConstant, one);
      auto *clonedOneConcat = rewriter.clone(*concat, mapping);

      if (negated)
        arrayElements.push_back(clonedZeroConcat->getResult(0));
      arrayElements.push_back(clonedOneConcat->getResult(0));
      if (!negated)
        arrayElements.push_back(clonedZeroConcat->getResult(0));
      return success();
    }
  }
  return failure();
}

//===----------------------------------------------------------------------===//
// ArcCanonicalizerPass implementation
//===----------------------------------------------------------------------===//

namespace {
struct ArcCanonicalizerPass
    : public ArcCanonicalizerBase<ArcCanonicalizerPass> {
  void runOnOperation() override;
};
} // namespace

void ArcCanonicalizerPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  patterns.add<ZeroCountRaising, IndexingConstArray>(&getContext());

  if (failed(mlir::applyPatternsAndFoldGreedily(getOperation(),
                                                std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> arc::createArcCanonicalizerPass() {
  return std::make_unique<ArcCanonicalizerPass>();
}
