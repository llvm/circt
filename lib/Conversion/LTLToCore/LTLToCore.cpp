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
#include "circt/Conversion/HWToSV.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/LTL/LTLDialect.h"
#include "circt/Dialect/LTL/LTLOps.h"
#include "circt/Dialect/SV/SVDialect.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"

namespace circt {
#define GEN_PASS_DEF_LOWERLTLTOCORE
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace hw;

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
struct HasBeenResetOpConversion : OpConversionPattern<verif::HasBeenResetOp> {
  using OpConversionPattern<verif::HasBeenResetOp>::OpConversionPattern;

  // HasBeenReset generates a 1 bit register that is set to one once the reset
  // has been raised and lowered at at least once.
  LogicalResult
  matchAndRewrite(verif::HasBeenResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto i1 = rewriter.getI1Type();
    // Generate the constant used to set the register value
    Value constZero = seq::createConstantInitialValue(
        rewriter, op->getLoc(), rewriter.getIntegerAttr(i1, 0));

    // Generate the constant used to negate the reset value
    Value constOne = hw::ConstantOp::create(rewriter, op.getLoc(), i1, 1);

    // Create a backedge for the register to be used in the OrOp
    circt::BackedgeBuilder bb(rewriter, op.getLoc());
    circt::Backedge reg = bb.get(rewriter.getI1Type());

    // Generate an or between the reset and the register's value to store
    // whether or not the reset has been active at least once
    Value orReset =
        comb::OrOp::create(rewriter, op.getLoc(), adaptor.getReset(), reg);

    // This register should not be reset, so we give it dummy reset and resetval
    // operands to fit the build signature
    Value reset, resetval;

    // Finally generate the register to set the backedge
    reg.setValue(seq::CompRegOp::create(
        rewriter, op.getLoc(), orReset,
        rewriter.createOrFold<seq::ToClockOp>(op.getLoc(), adaptor.getClock()),
        rewriter.getStringAttr("hbr"), reset, resetval, constZero,
        InnerSymAttr{} // inner_sym
        ));

    // We also need to consider the case where we are currently in a reset cycle
    // in which case our hbr register should be down-
    // Practically this means converting it to (and hbr (not reset))
    Value notReset = comb::XorOp::create(rewriter, op.getLoc(),
                                         adaptor.getReset(), constOne);
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, reg, notReset);

    return success();
  }
};

struct LTLImplicationConversion
    : public OpConversionPattern<ltl::ImplicationOp> {
  using OpConversionPattern<ltl::ImplicationOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::ImplicationOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Can only lower boolean implications to comb ops
    if (!isa<IntegerType>(op.getAntecedent().getType()) ||
        !isa<IntegerType>(op.getConsequent().getType()))
      return failure();
    /// A -> B = !A || B
    auto loc = op.getLoc();
    auto notA = comb::createOrFoldNot(loc, adaptor.getAntecedent(), rewriter);
    auto orOp =
        comb::OrOp::create(rewriter, loc, notA, adaptor.getConsequent());
    rewriter.replaceOp(op, orOp);
    return success();
  }
};

struct LTLNotConversion : public OpConversionPattern<ltl::NotOp> {
  using OpConversionPattern<ltl::NotOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::NotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Can only lower boolean nots to comb ops
    if (!isa<IntegerType>(op.getInput().getType()))
      return failure();
    auto loc = op.getLoc();
    auto inverted = comb::createOrFoldNot(loc, adaptor.getInput(), rewriter);
    rewriter.replaceOp(op, inverted);
    return success();
  }
};

struct LTLAndOpConversion : public OpConversionPattern<ltl::AndOp> {
  using OpConversionPattern<ltl::AndOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::AndOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Can only lower boolean ands to comb ops
    if (!isa<IntegerType>(op->getOperandTypes()[0]) ||
        !isa<IntegerType>(op->getOperandTypes()[1]))
      return failure();
    auto loc = op.getLoc();
    // Explicit twoState value to disambiguate builders
    auto andOp =
        comb::AndOp::create(rewriter, loc, adaptor.getOperands(), false);
    rewriter.replaceOp(op, andOp);
    return success();
  }
};

struct LTLOrOpConversion : public OpConversionPattern<ltl::OrOp> {
  using OpConversionPattern<ltl::OrOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::OrOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Can only lower boolean ors to comb ops
    if (!isa<IntegerType>(op->getOperandTypes()[0]) ||
        !isa<IntegerType>(op->getOperandTypes()[1]))
      return failure();
    auto loc = op.getLoc();
    // Explicit twoState value to disambiguate builders
    auto orOp = comb::OrOp::create(rewriter, loc, adaptor.getOperands(), false);
    rewriter.replaceOp(op, orOp);
    return success();
  }
};


template <typename OpTy>
struct AssertLikeConversion : public OpConversionPattern<OpTy> {
  int &nameID;
  AssertLikeConversion(TypeConverter &typeConverter, MLIRContext *context,
                       int &nameID)
      : OpConversionPattern<OpTy>(typeConverter, context), nameID(nameID) {}

  FailureOr<Value> getSeqClock(Value clkI1) const {
    if (isa<seq::ClockType>(clkI1.getType()))
      return clkI1;
    if (auto from = clkI1.getDefiningOp<seq::FromClockOp>())
      return from.getInput();
    return failure();
  }

  FailureOr<Value> lowerImplicationWithDelay(OpBuilder &builder, Location loc,
                                             Value antecedent, ltl::DelayOp delay,
                                             Value clkI1) const {
    if (!antecedent.getType().isInteger(1))
      return failure();
    if (!delay.getInput().getType().isInteger(1))
      return failure();

    int64_t start = static_cast<int64_t>(delay.getDelay());
    if (start < 1)
      return failure();

    auto lengthAttr = delay.getLength();
    if (!lengthAttr)
      return failure();
    int64_t length = static_cast<int64_t>(*lengthAttr);
    if (length < 0)
      return failure();
    int64_t end = start + length;

    auto clkOr = getSeqClock(clkI1);
    if (failed(clkOr))
      return failure();
    Value clk = *clkOr;

    auto nextName = [&](StringRef prefix) {
      return builder.getStringAttr((prefix + "_" + Twine(nameID++)).str());
    };

    auto createCompReg = [&](Location loc, Value input, Value clk,
                             StringRef prefix) {
      auto nameAttr = nextName(prefix);
      auto i1 = builder.getI1Type();
      Value initZero = seq::createConstantInitialValue(
          builder, loc, builder.getIntegerAttr(i1, 0));
      return seq::CompRegOp::create(builder, loc, input, clk,
                                    /*reset=*/Value(), /*rstValue=*/Value(),
                                    nameAttr, /*initialValue=*/initZero)
          .getResult();
    };

    auto shift = [&](Location loc, Value input, Value clk, int64_t cycles,
                     StringRef prefix) {
      Value cur = input;
      for (int64_t i = 0; i < cycles; ++i) {
        cur = createCompReg(loc, cur, clk, prefix);
      }
      return cur;
    };

    // Exact delay case: antecedent |-> ##start consequent
    // If the antecedent was true `start` cycles ago, the consequent MUST be true now.
    if (length == 0) {
      // 1. Remember the past: shift the antecedent through `start` registers.
      Value delayed = shift(loc, antecedent, clk, start, "ltl_delay");
      
      // 2. Boolean implication: A -> B is logically equivalent to NOT(A) OR B.
      Value notDelayed = comb::createOrFoldNot(loc, delayed, builder);
      auto orOp = comb::OrOp::create(builder, loc, ValueRange{notDelayed, delay.getInput()}, false);
      return orOp.getResult();
    }

    // Bounded window case: antecedent |-> ##[start:end] consequent
    // A "pending token" is created when antecedent fires. It shifts forward each cycle.
    // During the window [start, end], if the consequent is observed, the token is destroyed.
    // If the token survives to the end of the window, the property was violated.
    Value notRight = comb::createOrFoldNot(loc, delay.getInput(), builder);

    // Age 1 pending token is the antecedent delayed by 1 cycle.
    Value pending = shift(loc, antecedent, clk, 1, "ltl_delay");
    Value prev = pending;

    for (int64_t age = 1; age <= end; ++age) {
      Value nextVal;
      if (age == 1) {
        // Just the initial delayed antecedent
        nextVal = prev;
      } else if (age < start) {
        // Window not open yet: shift the token forward untouched
        nextVal = prev;
      } else {
        // Window is open: erase the token if consequent is true right now (AND NOT(consequent))
        auto andOp = comb::AndOp::create(builder, loc, ValueRange{prev, notRight}, false);
        nextVal = andOp.getResult();
      }
      prev = createCompReg(loc, nextVal, clk, "ltl_window");
    }

    // Violation occurs if a token reaches the deadline (end) AND the consequent is still false.
    auto viol = comb::AndOp::create(builder, loc, ValueRange{prev, notRight}, false);
    
    // Asserts expect '1' for success, so we invert the violation signal.
    Value notViol = comb::createOrFoldNot(loc, viol.getResult(), builder);
    return notViol;
  }

  FailureOr<Value> lowerClockedProperty(OpBuilder &builder, Location loc,
                                        Value prop, Value clkI1) const {
    if (auto imp = prop.getDefiningOp<ltl::ImplicationOp>()) {
      auto delay = imp.getConsequent().getDefiningOp<ltl::DelayOp>();
      if (!delay)
        return failure();
      return lowerImplicationWithDelay(builder, loc, imp.getAntecedent(), delay,
                                       clkI1);
    }
    if (prop.getType().isInteger(1))
      return prop;
    return failure();
  }

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto prop = op.getProperty();
    auto clkOp = prop.template getDefiningOp<ltl::ClockOp>();
    if (!clkOp)
      return failure();

    auto lowered = lowerClockedProperty(rewriter, op->getLoc(), clkOp.getInput(),
                                        clkOp.getClock());
    if (failed(lowered))
      return op->emitError("unsupported LTL delay pattern for LowerLTLToCore");

    rewriter.modifyOpInPlace(op, [&]() { op->setOperand(0, *lowered); });
    if (clkOp->use_empty())
      rewriter.eraseOp(clkOp);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Lower LTL To Core pass
//===----------------------------------------------------------------------===//

namespace {
struct LowerLTLToCorePass
    : public circt::impl::LowerLTLToCoreBase<LowerLTLToCorePass> {
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
  target.addLegalDialect<verif::VerifDialect>();
  target.addIllegalOp<verif::HasBeenResetOp>();

  auto isLegal = [](Operation *op) {
    auto hasNonAssertUsers = std::any_of(
        op->getUsers().begin(), op->getUsers().end(), [](Operation *user) {
          return !isa<verif::AssertOp, verif::ClockedAssertOp>(user);
        });
    auto hasIntegerResultTypes =
        std::all_of(op->getResultTypes().begin(), op->getResultTypes().end(),
                    [](Type type) { return isa<IntegerType>(type); });
    // If there are users other than asserts, we can't map it to comb (unless
    // the return type is already integer anyway)
    if (hasNonAssertUsers && !hasIntegerResultTypes)
      return true;

    // Otherwise illegal if operands are i1
    return std::any_of(
        op->getOperands().begin(), op->getOperands().end(),
        [](Value operand) { return !isa<IntegerType>(operand.getType()); });
  };
  target.addDynamicallyLegalOp<ltl::ImplicationOp>(isLegal);
  target.addDynamicallyLegalOp<ltl::NotOp>(isLegal);
  target.addDynamicallyLegalOp<ltl::AndOp>(isLegal);
  target.addDynamicallyLegalOp<ltl::OrOp>(isLegal);

  target.addDynamicallyLegalOp<verif::AssertOp, verif::AssumeOp>([](Operation *op) {
    Value prop = op->getOperand(0);
    return !prop.getDefiningOp<ltl::ClockOp>();
  });

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
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });

  converter.addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs, mlir::Location loc) -> mlir::Value {
        if (inputs.size() != 1)
          return Value();
        return UnrealizedConversionCastOp::create(builder, loc, resultType,
                                                  inputs[0])
            ->getResult(0);
      });

  // Create the operation rewrite patters
  RewritePatternSet patterns(&getContext());
  int nameID = 0;
  patterns.add<HasBeenResetOpConversion, LTLImplicationConversion,
               LTLNotConversion, LTLAndOpConversion, LTLOrOpConversion>(
      converter, patterns.getContext());
  patterns.add<AssertLikeConversion<verif::AssertOp>,
               AssertLikeConversion<verif::AssumeOp>>(converter,
                                                      patterns.getContext(),
                                                      nameID);

  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();

  // Clean up remaining unrealized casts by changing assert argument types
  getOperation().walk([&](Operation *op) {
    if (!isa<verif::AssertOp, verif::ClockedAssertOp>(op))
      return;
    Value prop = op->getOperand(0);
    if (auto cast = prop.getDefiningOp<UnrealizedConversionCastOp>()) {
      // Make sure that the cast is from an i1, not something random that was in
      // the input
      if (auto intType = dyn_cast<IntegerType>(cast.getOperandTypes()[0]);
          intType && intType.getWidth() == 1)
        op->setOperand(0, cast.getInputs()[0]);
    }
  });
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
