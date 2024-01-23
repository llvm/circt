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

  // DisableOp translates to an implication in to the form of
  // ~condition -> input
  // As it is assumed that the inpu
  LogicalResult
  matchAndRewrite(ltl::DisableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Generate the constant used to negate the
    /*Value constOne = rewriter.create<hw::ConstantOp>(
        op.getLoc(), mlir::IntegerAttr::get(
                         getContext(), llvm::APSInt(llvm::StringRef("1"))));

    // Generate the antecedant of the disable implication
    Value antecedent = rewriter.create<comb::XorOp>(
        op.getLoc(), adaptor.getCondition(), constOne);*/

    // Replace the ltl::DisableOp with the OR part of the implication
    rewriter.replaceOpWithNewOp<comb::OrOp>(op, adaptor.getCondition(),
                                            adaptor.getInput());

    return success();
  }
};

/// Lower a ltl::DisableOp operation to Core operations
struct ClockOpConversion : OpConversionPattern<ltl::ClockOp> {
  using OpConversionPattern<ltl::ClockOp>::OpConversionPattern;

  // LTL clock gets ignored and is modified via it's user operation
  LogicalResult
  matchAndRewrite(ltl::ClockOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // rewriter.replaceOp(op, adaptor.getClock().getDefiningOp());
    rewriter.eraseOp(op);
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

struct VerifAssertOpConversion : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  // A verif assertion is translated together with it's parenting ltl clock
  // This will yield an sv.always containing an sv.assert
  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    // Start by making sure that the input is associated to a clock
    auto ltlclk = dyn_cast<ltl::ClockOp>(adaptor.getProperty().getDefiningOp());

    // If it's not clocked then we can't generate a proper sv assertion
    if (!ltlclk)
      op->emitError("Verif assertions must be clocked!!");

    // Finish by genrating the parenting sv.always posedge clock from the ltl
    // clock, containing the generated sv.assert
    rewriter.replaceOpWithNewOp<sv::AlwaysOp>(
        op, (sv::EventControl)ltlclk.getEdge(), ltlclk.getClock(), [&] {
          // Generate the sv assertion using the input to the parenting clock
          rewriter.create<sv::AssertOp>(
              op.getLoc(), ltlclk.getInput(),
              sv::DeferAssertAttr::get(getContext(),
                                       sv::DeferAssert::Immediate),
              adaptor.getLabelAttr());
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
  LowerLTLToCorePass() {}
  void runOnOperation() override;
};
} // namespace

void circt::populateLTLToCoreConversionPatterns(
    TypeConverter &converter, mlir::RewritePatternSet &patterns) {
  patterns.add<HasBeenResetOpConversion, VerifAssertOpConversion,
               ClockOpConversion, DisableOpConversion>(converter,
                                                       patterns.getContext());
}

// Simply applies the conversion patterns defined above
void LowerLTLToCorePass::runOnOperation() {

  // Set target dialects: We don't want to see any ltl verif left in the result
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
  target.addIllegalOp<ltl::ClockOp>();

  // Create type converters, mostly just to convert an ltl property to a bool
  TypeConverter converter;
  converter.addConversion([](ltl::PropertyType type) {
    return IntegerType::get(type.getContext(), 1);
  });
  converter.addConversion([](Type type) { return type; });

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
