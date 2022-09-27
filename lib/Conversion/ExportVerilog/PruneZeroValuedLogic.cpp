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

#include "ExportVerilogInternals.h"
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

// Marks the provided 'op' as pruned.
static void pruneZeroWidthOp(Operation *op) {
  op->setAttr("pruned", StringAttr::get(op->getContext(), "Zero Width"));
}

// The NoI0OperandsConversionPattern will aggressively remove any operation
// which has a zero-valued operand.
template <typename TOp>
struct NoI0OperandsConversionPattern : public OpConversionPattern<TOp> {
public:
  using OpConversionPattern<TOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<TOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(TOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (noI0TypedValue(adaptor.getOperands()))
      return failure();

    // Part of i0-typed logic - prune!
    pruneZeroWidthOp(op);
    return failure();
  }
};

template <typename... TOp>
static void addNoI0OperandsLegalizationPattern(ConversionTarget &target) {
  target.addDynamicallyLegalOp<TOp...>([&](auto op) {
    return op->hasAttr("pruned") || noI0TypedValue(op->getOperands());
  });
}

// A generic pruning pattern which prunes any operation which has an operand
// with an i0 typed value. Similarly, an operation is legal if all of its
// operands are not i0 typed.
template <typename TOp>
struct NoI0OperandPruningPattern {
  using ConversionPattern = NoI0OperandsConversionPattern<TOp>;
  static void addLegalizer(ConversionTarget &target) {
    addNoI0OperandsLegalizationPattern<TOp>(target);
  }
};

// Adds a pruning pattern to the conversion target. TPattern is expected to
// provides ConversionPattern definition and an addLegalizer function.
template <typename... TPattern>
static void addPruningPattern(ConversionTarget &target,
                              RewritePatternSet &patterns,
                              PruneTypeConverter &typeConverter) {
  (patterns.add<typename TPattern::ConversionPattern>(typeConverter,
                                                      patterns.getContext()),
   ...);
  (TPattern::addLegalizer(target), ...);
}
} // namespace

LogicalResult ExportVerilog::pruneZeroValuedLogic(hw::HWModuleOp module) {
  ConversionTarget target(*module.getContext());
  RewritePatternSet patterns(module.getContext());
  PruneTypeConverter typeConverter;

  target.addLegalDialect<sv::SVDialect, comb::CombDialect, hw::HWDialect>();

  // Generic conversion and legalization patterns for operations that we
  // expect to be using i0 valued logic.
  addPruningPattern<NoI0OperandPruningPattern<sv::PAssignOp>,
                    NoI0OperandPruningPattern<sv::BPAssignOp>,
                    NoI0OperandPruningPattern<sv::AssignOp>>(target, patterns,
                                                             typeConverter);

  return applyPartialConversion(module, target, std::move(patterns));
}
