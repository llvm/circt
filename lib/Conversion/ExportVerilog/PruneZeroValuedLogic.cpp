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

// Returns true if 'op' is a legal user of an I0 value.
static bool isLegalI0User(Operation *op) {
  return isa<hw::OutputOp, hw::ArrayGetOp, sv::ArrayIndexInOutOp,
             hw::InstanceOp>(op);
}

static bool noI0Type(TypeRange types) {
  return llvm::none_of(
      types, [](Type type) { return ExportVerilog::isZeroBitType(type); });
}

static bool noI0TypedValue(ValueRange values) {
  return noI0Type(values.getTypes());
}

namespace {

class PruneTypeConverter : public mlir::TypeConverter {
public:
  PruneTypeConverter() {
    addConversion([&](Type type, SmallVectorImpl<Type> &results) {
      if (!ExportVerilog::isZeroBitType(type))
        results.push_back(type);
      return success();
    });
  }
};

// The NoI0OperandsConversionPattern will aggressively remove any operation
// which has a zero-width operand.
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

    // Part of i0-typed logic - prune it!
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename... TOp>
static void addNoI0OperandsLegalizationPattern(ConversionTarget &target) {
  target.addDynamicallyLegalOp<TOp...>(
      [&](auto op) { return noI0TypedValue(op->getOperands()); });
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

// The NoI0ResultsConversionPattern will aggressively remove any operation
// which has a zero-width result. Furthermore, it will recursively erase any
// downstream users of the operation.
template <typename TOp>
struct NoI0ResultsConversionPattern : public OpConversionPattern<TOp> {
public:
  using OpConversionPattern<TOp>::OpConversionPattern;
  using OpAdaptor = typename OpConversionPattern<TOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(TOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (noI0TypedValue(op->getResults()))
      return failure();

    // Part of i0-typed logic - prune!
    // If the operation defines a value which is used by a valid user, we
    // replace it with a zero-width constant.
    for (auto res : op->getResults()) {
      for (auto *user : res.getUsers()) {
        if (isLegalI0User(user)) {
          assert(op->getNumResults() == 1 &&
                 "expected single result if using rewriter.replaceOpWith");
          rewriter.replaceOpWithNewOp<hw::ConstantOp>(
              op, APInt(0, 0, /*isSigned=*/false));
          return success();
        }
      }
    }

    // Else, just erase the op.
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename... TOp>
static void addNoI0ResultsLegalizationPattern(ConversionTarget &target) {
  target.addDynamicallyLegalOp<TOp...>(
      [&](auto op) { return noI0TypedValue(op->getResults()); });
}

// A generic pruning pattern which prunes any operation that returns an i0
// value.
template <typename TOp>
struct NoI0ResultPruningPattern {
  using ConversionPattern = NoI0ResultsConversionPattern<TOp>;
  static void addLegalizer(ConversionTarget &target) {
    addNoI0ResultsLegalizationPattern<TOp>(target);
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

template <typename... TOp>
static void addNoI0ResultPruningPattern(ConversionTarget &target,
                                        RewritePatternSet &patterns,
                                        PruneTypeConverter &typeConverter) {
  (patterns.add<typename NoI0ResultPruningPattern<TOp>::ConversionPattern>(
       typeConverter, patterns.getContext()),
   ...);
  (NoI0ResultPruningPattern<TOp>::addLegalizer(target), ...);
}

} // namespace

void ExportVerilog::pruneZeroValuedLogic(hw::HWModuleOp module) {
  ConversionTarget target(*module.getContext());
  RewritePatternSet patterns(module.getContext());
  PruneTypeConverter typeConverter;

  target.addLegalDialect<sv::SVDialect, comb::CombDialect, hw::HWDialect>();
  addPruningPattern<NoI0OperandPruningPattern<sv::PAssignOp>,
                    NoI0OperandPruningPattern<sv::BPAssignOp>,
                    NoI0OperandPruningPattern<sv::AssignOp>>(target, patterns,
                                                             typeConverter);

  addNoI0ResultPruningPattern<
      // SV ops
      sv::WireOp, sv::RegOp, sv::ReadInOutOp,
      // Prune all zero-width combinational logic.
      comb::AddOp, comb::AndOp, comb::ConcatOp, comb::DivSOp, comb::DivUOp,
      comb::ExtractOp, comb::ModSOp, comb::ModUOp, comb::MulOp, comb::MuxOp,
      comb::OrOp, comb::ParityOp, comb::ReplicateOp, comb::ShlOp, comb::ShrSOp,
      comb::ShrUOp, comb::SubOp, comb::XorOp>(target, patterns, typeConverter);

  (void)applyPartialConversion(module, target, std::move(patterns));
}
