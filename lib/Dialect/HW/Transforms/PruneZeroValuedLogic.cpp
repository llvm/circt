//===- PruneZeroValuedLogic.cpp - Prune zero-valued logic -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform removes zero-valued logic from a `hw.module`.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace llvm;
using namespace mlir;
using namespace circt;
using namespace hw;

namespace {

// Returns true if 't' is zero-width logic.
// For now, this strictly relies on the announced bit-width of the type.
static bool isZeroWidthLogic(Type t) {
  if (!t.isa<IntegerType>())
    return false;
  return t.getIntOrFloatBitWidth() == 0;
}

static bool noI0Type(TypeRange types) {
  return llvm::none_of(types, [](Type type) { return isZeroWidthLogic(type); });
}

static bool noI0TypedValue(ValueRange values) {
  return noI0Type(values.getTypes());
}

static SmallVector<Value> removeI0Typed(ValueRange values) {
  SmallVector<Value> result;
  llvm::copy_if(values, std::back_inserter(result),
                [](Value value) { return !isZeroWidthLogic(value.getType()); });
  return result;
}

class PruneTypeConverter : public mlir::TypeConverter {
public:
  PruneTypeConverter() {
    addConversion([&](Type type, SmallVectorImpl<Type> &results) {
      if (!isZeroWidthLogic(type))
        results.push_back(type);
      return success();
    });
  }
};

struct OutputOpConversionPattern : public OpConversionPattern<OutputOp> {
public:
  using OpConversionPattern<OutputOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (noI0TypedValue(adaptor.getOperands()))
      return failure();

    auto prunedOperands = removeI0Typed(adaptor.getOperands());
    rewriter.replaceOpWithNewOp<OutputOp>(op, prunedOperands);
    return success();
  }
};

// The PruningConversionPattern will aggressively remove any operation which has
// a zero-valued operand. It is therefore implied that any operation which takes
// part of a chain of logic containing i0 values will be removed.
template <typename TOp>
struct PruningConversionPattern : public OpConversionPattern<TOp> {
public:
  using OpConversionPattern<TOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<TOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(TOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (noI0TypedValue(adaptor.getOperands()))
      return failure();

    // Part of i0-typed logic - erase!
    rewriter.eraseOp(op);
    return success();
  }
};

struct ToRemoveArgResNames {
  Operation *op;
  llvm::SmallVector<unsigned> argIndices;
  llvm::SmallVector<unsigned> resIndices;

  static ArrayAttr removeIndices(ArrayAttr attrs, ArrayRef<unsigned> indices) {
    llvm::SmallVector<Attribute> cleanedAttrs;
    for (auto [idx, attr] : llvm::enumerate(attrs))
      if (!llvm::is_contained(indices, idx))
        cleanedAttrs.push_back(attr);
    return ArrayAttr::get(attrs.getContext(), cleanedAttrs);
  }

  void clean() {
    auto argNames = op->getAttrOfType<ArrayAttr>("argNames");
    auto resNames = op->getAttrOfType<ArrayAttr>("resultNames");
    op->setAttr("argNames", removeIndices(argNames, argIndices));
    op->setAttr("resultNames", removeIndices(resNames, resIndices));
  }
};

llvm::SmallVector<ToRemoveArgResNames>
getToRemoveArgResNames(mlir::ModuleOp module) {
  llvm::SmallVector<ToRemoveArgResNames> toRemove;
  for (auto op : module.getOps<HWModuleLike>()) {
    ToRemoveArgResNames toRemoveForOp;
    toRemoveForOp.op = op;
    auto funcIF = cast<FunctionOpInterface>(op.getOperation());
    for (auto [idx, arg] : llvm::enumerate(funcIF.getArgumentTypes())) {
      if (isZeroWidthLogic(arg))
        toRemoveForOp.argIndices.push_back(idx);
    }
    for (auto [idx, res] : llvm::enumerate(funcIF.getResultTypes())) {
      if (isZeroWidthLogic(res))
        toRemoveForOp.resIndices.push_back(idx);
    }

    if (!toRemoveForOp.argIndices.empty() || !toRemoveForOp.resIndices.empty())
      toRemove.push_back(toRemoveForOp);
  }
  return toRemove;
}

template <typename... TOp>
static void addSignatureConversion(ConversionTarget &target,
                                   RewritePatternSet &patterns,
                                   PruneTypeConverter &typeConverter) {
  (mlir::populateFunctionOpInterfaceTypeConversionPattern<TOp>(patterns,
                                                               typeConverter),
   ...);

  target.addDynamicallyLegalOp<TOp...>([&](FunctionOpInterface moduleLikeOp) {
    // Legal if no results and args have i0 values.
    bool legalResults = noI0Type(moduleLikeOp.getResultTypes());
    bool legalArgs = noI0Type(moduleLikeOp.getArgumentTypes());
    return legalResults && legalArgs;
  });
}

template <typename... TOp>
static void addNoI0TypedOperandsLegalizationPattern(ConversionTarget &target) {
  target.addDynamicallyLegalOp<TOp...>(
      [&](auto op) { return noI0TypedValue(op->getOperands()); });
}

// Adds a pattern to prune TOp if it contains a zero-valued operand, as well as
// a dynamic legality check.
template <typename... TOp>
static void addPruningPattern(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              PruneTypeConverter &typeConverter) {
  (patterns.add<PruningConversionPattern<TOp>>(typeConverter,
                                               patterns.getContext()),
   ...);
  (addNoI0TypedOperandsLegalizationPattern<TOp>(target), ...);
}

struct PruneZeroValuedLogicPass
    : public PruneZeroValuedLogicBase<PruneZeroValuedLogicPass> {
  PruneZeroValuedLogicPass() {}
  void runOnOperation() override {
    ModuleOp module = getOperation();
    ConversionTarget target(getContext());
    RewritePatternSet patterns(&getContext());
    PruneTypeConverter typeConverter;

    // Record argument and res names which needs to be cleaned up
    // post-conversion.
    auto toRemoveArgResNames = getToRemoveArgResNames(module);

    // Signature conversion and legalization patterns.
    addSignatureConversion<hw::HWModuleOp, hw::HWModuleExternOp>(
        target, patterns, typeConverter);

    // Generic conversion and legalization patterns for operations that we
    // expect to be using i0 valued logic.
    addPruningPattern<comb::AddOp, comb::AndOp, comb::ICmpOp, comb::ConcatOp,
                      comb::ExtractOp, comb::MuxOp, comb::OrOp, comb::ShlOp,
                      comb::ShrSOp, comb::SubOp, comb::XorOp, seq::CompRegOp>(
        target, patterns, typeConverter);

    // Other patterns.
    patterns.add<OutputOpConversionPattern>(typeConverter,
                                            patterns.getContext());
    addNoI0TypedOperandsLegalizationPattern<hw::OutputOp>(target);

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      return signalPassFailure();

    // Cleanup argument and result names.
    for (auto &toRemove : toRemoveArgResNames)
      toRemove.clean();
  }
};

} // namespace

std::unique_ptr<Pass> circt::hw::createPruneZeroValuedLogicPass() {
  return std::make_unique<PruneZeroValuedLogicPass>();
}
