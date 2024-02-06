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

  // HasBeenReset generates a 1 bit register that is set to one if the reset
  // value is found to be active at any point
  LogicalResult
  matchAndRewrite(verif::HasBeenResetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Generate the constant used to set the register value
    Value constOne = rewriter.create<hw::ConstantOp>(
        op.getLoc(), mlir::IntegerType::get(op.getContext(), 1), 1);

    // Create a backedge for the register to be used in the mux
    circt::BackedgeBuilder bb(rewriter, op.getLoc());
    circt::Backedge reg = bb.get(rewriter.getI1Type());

    // Generate a multiplexer to select the register's value
    Value mux = rewriter.replaceOpWithNewOp<comb::MuxOp>(op, adaptor.getReset(),
                                                         constOne, reg);

    // Finally generate the register to set the backedge
    reg.setValue(rewriter.create<seq::CompRegOp>(
        op.getLoc(), mux, adaptor.getClock().getDefiningOp()->getOperand(0),
        llvm::StringRef("hbr")));

    return success();
  }
};

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
        assertOp, (sv::EventControl)adaptor.getEdge(), adaptor.getClock(), [&] {
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

// Applies all of the conversion patterns needed for this lowering
void circt::populateLTLToCoreConversionPatterns(
    circt::LTLToCoreTypeConverter &converter,
    mlir::RewritePatternSet &patterns) {
  patterns.add<HasBeenResetOpConversion, LowerClockRelatedOpPatterns,
               DisableOpConversion>(converter, patterns.getContext());
}

// Creates the type conversions and materializations needed for this pass to
// work
circt::LTLToCoreTypeConverter::LTLToCoreTypeConverter() {

  // Convert the ltl property type to a built-in type
  addConversion([](Type type) { return type; });
  addConversion([](ltl::PropertyType type) {
    return IntegerType::get(type.getContext(), 1);
  });

  // Basic materializations
  addTargetMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });

  addSourceMaterialization(
      [&](mlir::OpBuilder &builder, mlir::Type resultType,
          mlir::ValueRange inputs,
          mlir::Location loc) -> std::optional<mlir::Value> {
        if (inputs.size() != 1)
          return std::nullopt;
        return inputs[0];
      });
}

// Simply applies the conversion patterns defined above
void LowerLTLToCorePass::runOnOperation() {

  // Set target dialects: We don't want to see any ltl or verif that might come
  // from an AssertProperty left in the result
  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalOp<hw::ConstantOp>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalOp<comb::OrOp>();
  target.addLegalOp<comb::MuxOp>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalOp<sv::AssertOp>();
  target.addLegalOp<sv::AlwaysOp>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addLegalOp<seq::CompRegOp>();
  target.addIllegalOp<verif::HasBeenResetOp>();
  target.addIllegalOp<verif::AssertOp>();
  target.addIllegalOp<ltl::DisableOp>();

  // Create type converters, mostly just to convert an ltl property to a bool
  LTLToCoreTypeConverter converter;

  // Create the operation rewrite patters
  RewritePatternSet patterns(&getContext());
  populateLTLToCoreConversionPatterns(converter, patterns);

  // Apply the conversions
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    return signalPassFailure();
}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
