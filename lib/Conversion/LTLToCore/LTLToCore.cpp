//===- LTLToCore.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Converts LTL and Verif operations to Core operations
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/LTLToCore.h"
#include "../PassDetail.h"
#include "circt/Conversion/HWToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifDialect.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"

using namespace mlir;
using namespace circt;
using namespace hw;

static verif::ClockEdge ltlToVerifClockEdge(ltl::ClockEdge ce) {
  switch (ce) {
  case ltl::ClockEdge::Pos:
    return verif::ClockEdge::Pos;
  case ltl::ClockEdge::Neg:
    return verif::ClockEdge::Neg;
  case ltl::ClockEdge::Both:
    return verif::ClockEdge::Both;
  }
  llvm_unreachable("Unknown event control kind");
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

// Custom pattern matchers

// Matches and records a boolean attribute
struct I1ValueMatcher {
  Value *what;
  I1ValueMatcher(Value *what) : what(what) {}
  bool match(Value op) const {
    if (!op.getType().isSignlessInteger(1))
      return false;
    *what = op;
    return true;
  }
};

static inline I1ValueMatcher mBool(Value *const val) {
  return I1ValueMatcher(val);
}

// Matches and records an arbitrary op
template <typename OpType, typename... OperandMatchers>
struct BindingRecursivePatternMatcher
    : mlir::detail::RecursivePatternMatcher<OpType, OperandMatchers...> {

  using BaseMatcher =
      mlir::detail::RecursivePatternMatcher<OpType, OperandMatchers...>;
  BindingRecursivePatternMatcher(OpType *bop, OperandMatchers... matchers)
      : BaseMatcher(matchers...), opBind(bop) {}

  bool match(Operation *op) {
    if (BaseMatcher::match(op)) {
      *opBind = llvm::cast<OpType>(op);
      return true;
    }
    return false;
  }

  OpType *opBind;
};

template <typename OpType, typename... Matchers>
static inline auto mOpWithBind(OpType *op, Matchers... matchers) {
  return BindingRecursivePatternMatcher<OpType, Matchers...>(op, matchers...);
}

struct HasBeenResetOpConversion : OpConversionPattern<verif::HasBeenResetOp> {
  using OpConversionPattern<verif::HasBeenResetOp>::OpConversionPattern;

  // HasBeenReset generates a 1 bit register that is set to one once the reset
  // has been raised and lowered at at least once.
  LogicalResult
  matchAndRewrite(verif::HasBeenResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1 = rewriter.getI1Type();
    // Generate the constant used to set the register value
    Value constZero = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 0);

    // Generate the constant used to enegate the
    Value constOne = rewriter.create<hw::ConstantOp>(op.getLoc(), i1, 1);

    // Create a backedge for the register to be used in the OrOp
    circt::BackedgeBuilder bb(rewriter, op.getLoc());
    circt::Backedge reg = bb.get(rewriter.getI1Type());

    // Generate an or between the reset and the register's value to store
    // whether or not the reset has been active at least once
    Value orReset =
        rewriter.create<comb::OrOp>(op.getLoc(), adaptor.getReset(), reg);

    // This register should not be reset, so we give it dummy reset and resetval
    // operands to fit the build signature
    Value reset, resetval;

    // Finally generate the register to set the backedge
    reg.setValue(rewriter.create<seq::CompRegOp>(
        op.getLoc(), orReset,
        rewriter.createOrFold<seq::ToClockOp>(op.getLoc(), adaptor.getClock()),
        rewriter.getStringAttr("hbr"), reset, resetval, constZero,
        InnerSymAttr{} // inner_sym
        ));

    // We also need to consider the case where we are currently in a reset cycle
    // in which case our hbr register should be down-
    // Practically this means converting it to (and hbr (not reset))
    Value notReset =
        rewriter.create<comb::XorOp>(op.getLoc(), adaptor.getReset(), constOne);
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, reg, notReset);

    return success();
  }
};

struct AssertLikeOp {
  // Assert and assume are disable by creating a disjunction between the
  // disable condition and the assertion condition.
  // Coverops are disabled instead visa a negative conjunction.
  static Value visit(Value input, Value disable, Location loc,
                     ConversionPatternRewriter &rewriter,
                     bool isCover = false) {
    // Replace the ltl::DisableOp with an OR op as it represents a disabling
    // implication: (implies (not condition) input) is equivalent to
    // (or (not (not condition)) input) which becomes (or condition input)
    // for assertions and assumptions, otherwised with an NOT cond AND pred
    if (isCover) {
      Value constOne =
          rewriter.create<hw::ConstantOp>(loc, rewriter.getI1Type(), 1);
      Value notDisable = rewriter.create<comb::XorOp>(loc, disable, constOne);
      return rewriter.createOrFold<comb::AndOp>(loc, notDisable, input);
    }

    return rewriter.createOrFold<comb::OrOp>(loc, disable, input);
  }
};

struct AssertOpConversionPattern : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value ltlClock, disableCond, disableInput, disabledProperty;
    ltl::ClockOp clockOp;
    ltl::DisableOp disableOp;

    // Look for the Assert Property pattern
    bool matchedProperty = matchPattern(
        op.getProperty(),
        mOpWithBind<ltl::ClockOp>(
            &clockOp,
            mOpWithBind<ltl::DisableOp>(&disableOp, mBool(&disableInput),
                                        mBool(&disableCond)),
            mBool(&ltlClock)));

    if (!matchedProperty)
      return rewriter.notifyMatchFailure(op,
                                         " unsupported assert-like pattern!");

    // Then visit the disable op
    disabledProperty = AssertLikeOp::visit(disableInput, disableCond,
                                           disableOp.getLoc(), rewriter);
    Value emptyDisable;

    // If the clock op matches this pattern, fold it into a clocked_assert
    rewriter.replaceOpWithNewOp<verif::ClockedAssertOp>(
        op, disabledProperty, ltlToVerifClockEdge(clockOp.getEdge()), ltlClock,
        emptyDisable, op.getLabelAttr());

    // Erase Converted Ops
    rewriter.eraseOp(clockOp);

    return success();
  }
};

struct AssumeOpConversionPattern : OpConversionPattern<verif::AssumeOp> {
  using OpConversionPattern<verif::AssumeOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(verif::AssumeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value ltlClock, disableCond, disableInput, disabledProperty;
    ltl::ClockOp clockOp;
    ltl::DisableOp disableOp;

    // Look for the Assert Property pattern
    bool matchedProperty = matchPattern(
        op.getProperty(),
        mOpWithBind<ltl::ClockOp>(
            &clockOp,
            mOpWithBind<ltl::DisableOp>(&disableOp, mBool(&disableInput),
                                        mBool(&disableCond)),
            mBool(&ltlClock)));

    if (!matchedProperty)
      return rewriter.notifyMatchFailure(op,
                                         " unsupported assert-like pattern!");

    // Then visit the disable op
    disabledProperty = AssertLikeOp::visit(disableInput, disableCond,
                                           disableOp.getLoc(), rewriter);

    Value emptyDisable;

    // If the clock op matches this pattern, fold it into a clocked_assert
    rewriter.replaceOpWithNewOp<verif::ClockedAssumeOp>(
        op, disabledProperty, ltlToVerifClockEdge(clockOp.getEdge()), ltlClock,
        emptyDisable, op.getLabelAttr());

    // Erase Converted Ops
    rewriter.eraseOp(clockOp);

    return success();
  }
};

struct CoverOpConversionPattern : OpConversionPattern<verif::CoverOp> {
  using OpConversionPattern<verif::CoverOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(verif::CoverOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Value ltlClock, disableCond, disableInput, disabledProperty;
    ltl::ClockOp clockOp;
    ltl::DisableOp disableOp;

    // Look for the Assert Property pattern
    bool matchedProperty = matchPattern(
        op.getProperty(),
        mOpWithBind<ltl::ClockOp>(
            &clockOp,
            mOpWithBind<ltl::DisableOp>(&disableOp, mBool(&disableInput),
                                        mBool(&disableCond)),
            mBool(&ltlClock)));

    if (!matchedProperty)
      return rewriter.notifyMatchFailure(op,
                                         " unsupported assert-like pattern!");

    // Then visit the disable op
    disabledProperty = AssertLikeOp::visit(disableInput, disableCond,
                                           disableOp.getLoc(), rewriter, true);
    Value emptyDisable;

    // If the clock op matches this pattern, fold it into a clocked_assert
    rewriter.replaceOpWithNewOp<verif::ClockedCoverOp>(
        op, disabledProperty, ltlToVerifClockEdge(clockOp.getEdge()), ltlClock,
        emptyDisable, op.getLabelAttr());

    // Erase Converted Ops
    rewriter.eraseOp(clockOp);

    return success();
  }

  struct ClockedAssertOpConversionPattern
      : OpConversionPattern<verif::ClockedAssertOp> {
    using OpConversionPattern<verif::ClockedAssertOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(verif::ClockedAssertOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // Then visit the disable op
      Value disabledProperty = AssertLikeOp::visit(
          adaptor.getProperty(), adaptor.getDisable(), op.getLoc(), rewriter);
      Value emptyDisable;

      // If the clock op matches this pattern, fold it into a clocked_assert
      rewriter.replaceOpWithNewOp<verif::ClockedAssertOp>(
          op, disabledProperty, adaptor.getEdge(), adaptor.getClock(),
          emptyDisable, adaptor.getLabelAttr());

      return success();
    }
  };

  struct ClockedAssumeOpConversionPattern
      : OpConversionPattern<verif::ClockedAssumeOp> {
    using OpConversionPattern<verif::ClockedAssumeOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(verif::ClockedAssumeOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // Then visit the disable op
      Value disabledProperty = AssertLikeOp::visit(
          adaptor.getProperty(), adaptor.getDisable(), op.getLoc(), rewriter);
      Value emptyDisable;

      // If the clock op matches this pattern, fold it into a clocked_assert
      rewriter.replaceOpWithNewOp<verif::ClockedAssumeOp>(
          op, disabledProperty, adaptor.getEdge(), adaptor.getClock(),
          emptyDisable, adaptor.getLabelAttr());

      return success();
    }
  };

  struct ClockedCoverOpConversionPattern
      : OpConversionPattern<verif::ClockedCoverOp> {
    using OpConversionPattern<verif::ClockedCoverOp>::OpConversionPattern;
    LogicalResult
    matchAndRewrite(verif::ClockedCoverOp op, OpAdaptor adaptor,
                    ConversionPatternRewriter &rewriter) const override {
      // Then visit the disable op
      Value disabledProperty =
          AssertLikeOp::visit(adaptor.getProperty(), adaptor.getDisable(),
                              op.getLoc(), rewriter, true);
      Value emptyDisable;

      // If the clock op matches this pattern, fold it into a clocked_assert
      rewriter.replaceOpWithNewOp<verif::ClockedCoverOp>(
          op, disabledProperty, adaptor.getEdge(), adaptor.getClock(),
          emptyDisable, adaptor.getLabelAttr());

      return success();
    }
  };
};

} // namespace

//===----------------------------------------------------------------------===//
// Lower LTL To Core pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerLTLToCorePass : public LowerLTLToCoreBase<LowerLTLToCorePass> {
  LowerLTLToCorePass() = default;
  void runOnOperation() override;
};
} // namespace

// Simply applies the conversion patterns defined above
void LowerLTLToCorePass::runOnOperation() {

  // Set target dialects: We don't want to see any ltl or verif that might
  // come from an AssertProperty left in the result
  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalDialect<ltl::LTLDialect>();
  target.addIllegalDialect<verif::VerifDialect>();

  // Create type converters, mostly just to convert an ltl property to a bool
  mlir::TypeConverter converter;

  // Convert the ltl property type to a built-in type
  converter.addConversion([](IntegerType type) { return type; });
  converter.addConversion([](ltl::PropertyType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([](ltl::SequenceType type) {
    return IntegerType::get(type.getContext(), 1);
  });

  // Basic materializations
  converter.addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });

  converter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });

  // Create the operation rewrite patters
  RewritePatternSet patterns(&getContext());
  patterns.add<AssertOpConversionPattern, AssumeOpConversionPattern,
               CoverOpConversionPattern, HasBeenResetOpConversion>(
      converter, patterns.getContext());

  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
