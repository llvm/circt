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
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "circt/Support/Namespace.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace hw;

static sv::EventControl LTLToSVEventControl(ltl::ClockEdge ce) {
  switch (ce) {
  case ltl::ClockEdge::Pos:
    return sv::EventControl::AtPosEdge;
  case ltl::ClockEdge::Neg:
    return sv::EventControl::AtNegEdge;
  case ltl::ClockEdge::Both:
    return sv::EventControl::AtEdge;
  }
  llvm_unreachable("Unknown event control kind");
}

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// Lower a ltl::DisableOp operation to Core operations
struct DisableOpConversion : OpConversionPattern<ltl::DisableOp> {
  using OpConversionPattern<ltl::DisableOp>::OpConversionPattern;

  // DisableOp translates to an implication in the form of
  // ~condition -> input
  LogicalResult
  matchAndRewrite(ltl::DisableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Replace the ltl::DisableOp with an OR op as it represents a disabling
    // implication: (implies (not condition) input) is equivalent to
    // (or (not (not condition)) input) which becomes (or condition input)
    rewriter.replaceOpWithNewOp<comb::OrOp>(op, adaptor.getCondition(),
                                            adaptor.getInput());

    return success();
  }
};

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
        reset, resetval, llvm::StringRef("hbr"), constZero));

    // We also need to consider the case where we are currently in a reset cycle
    // in which case our hbr register should be down-
    // Practically this means converting it to (and hbr (not reset))
    Value notReset =
        rewriter.create<comb::XorOp>(op.getLoc(), adaptor.getReset(), constOne);
    rewriter.replaceOpWithNewOp<comb::AndOp>(op, reg, notReset);

    return success();
  }
};

/*struct ClockOpConversionPattern : OpConversionPattern<ltl::ClockOp> {
  using OpConversionPattern<ltl::ClockOp>::OpConversionPattern;
  LogicalResult
  matchAndRewrite(ltl::ClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op);
    return success();
  }
};

struct AssertOpConversionPattern : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Retrieve predecessor clock and use it to generate the alwaysop
    return llvm::TypeSwitch<Operation *, LogicalResult>(
               adaptor.getProperty().getDefiningOp())
        .Case<mlir::UnrealizedConversionCastOp>([&](auto cast) {
          auto inputs = cast->getOperand(0);
          if (auto clockop = dyn_cast<ltl::ClockOp>(inputs.getDefiningOp())) {
            // Generate the parenting sv.always posedge clock from the ltl
            // clock, containing the generated sv.assert
            rewriter.replaceOpWithNewOp<sv::AlwaysOp>(
                clockop, LTLToSVEventControl(clockop.getEdge()),
                clockop.getClock(), [&] {
                  // Generate the sv assertion using the input to the
                  // parenting clock
                  rewriter.replaceOpWithNewOp<sv::AssertOp>(
                      op, clockop.getInput(),
                      sv::DeferAssertAttr::get(getContext(),
                                               sv::DeferAssert::Immediate),
                      op.getLabelAttr());
                });

            return success();
          }
          return rewriter.notifyMatchFailure(
              op, "verif.assert property is not associated to a clock! " +
                      inputs.getDefiningOp()->getName().getStringRef());
        })
        .Case<ltl::ClockOp>([&](auto clockop) {
          // Generate the parenting sv.always posedge clock from the ltl
          // clock, containing the generated sv.assert
          rewriter.replaceOpWithNewOp<sv::AlwaysOp>(
              clockop, LTLToSVEventControl(clockop.getEdge()),
              clockop.getClock(), [&] {
                // Generate the sv assertion using the input to the
                // parenting clock
                rewriter.replaceOpWithNewOp<sv::AssertOp>(
                    op, clockop.getInput(),
                    sv::DeferAssertAttr::get(getContext(),
                                             sv::DeferAssert::Immediate),
                    op.getLabelAttr());
              });

          return success();
        })
        .Default([&](auto e) {
          return rewriter.notifyMatchFailure(
              op, "verif.assert property is not associated to a clock! " +
                      e->getName().getStringRef());
        });
  }
};*/

struct LowerClockRelatedOpPatterns : OpConversionPattern<ltl::ClockOp> {
  using OpConversionPattern<ltl::ClockOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ltl::ClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Match for a single verif.assert user, and replace the structure.
    // Folding will recursively apply this pattern if multiple instances are
    // found.
    auto users = op.getResult().getUsers();
    if (users.empty())
      return rewriter.notifyMatchFailure(op, "No users found");

    for (auto *user : op.getResult().getUsers()) {
      auto fusedLoc =
          mlir::FusedLoc::get(getContext(), {op.getLoc(), user->getLoc()});

      // Dispatch to the clock-specific patterns.
      LogicalResult res =
          llvm::TypeSwitch<Operation *, LogicalResult>(user)
              .Case<verif::AssertOp>([&](auto assert) {
                return rewriteAssertOp(fusedLoc, assert, adaptor, rewriter);
              })
              .Default([&](auto) {
                return rewriter.notifyMatchFailure(
                    op, "User of clock is not a verif.assert");
              });
      if (failed(res))
        return res;
    }

    // Clock can be removed as all users should have been handled by now.
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult rewriteAssertOp(Location loc, verif::AssertOp assertOp,
                                OpAdaptor &adaptor,
                                ConversionPatternRewriter &rewriter) const {
    // Generate the parenting sv.always posedge clock from the ltl
    // clock, containing the generated sv.assert
    rewriter.replaceOpWithNewOp<sv::AlwaysOp>(
        assertOp, hwToSvEventControl(adaptor.getEdge()), adaptor.getClock(),
        [&] {
          // Generate the sv assertion using the input to the parenting
          // clock
          rewriter.create<sv::AssertOp>(
              loc, adaptor.getInput(),
              sv::DeferAssertAttr::get(getContext(),
                                       sv::DeferAssert::Immediate),
              assertOp.getLabelAttr());
        });

    return success();
  }
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

  // Set target dialects: We don't want to see any ltl or verif that might come
  // from an AssertProperty left in the result
  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addIllegalDialect<ltl::LTLDialect>();
  target.addIllegalDialect<verif::VerifDialect>();

  // Create type converters, mostly just to convert an ltl property to a bool
  mlir::TypeConverter converter;

  // Convert the ltl property type to a built-in type
  converter.addConversion([](IntegerType type) { return type; });
  converter.addConversion([](ltl::PropertyType type) {
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
  patterns.add<HasBeenResetOpConversion, AssertOpConversionPattern,
               ClockOpConversionPattern, DisableOpConversion>(
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
