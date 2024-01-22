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
        op.getLoc(), mlir::UnitAttr(), adaptor.getCondition(), constOne);*/

    // Replace the ltl::DisableOp with the OR part of the implication
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
        op.getLoc(), mlir::IntegerAttr::get(
                         getContext(), llvm::APSInt(llvm::StringRef("1"))));

    // Create a backedge for the register to be used in the mux
    circt::BackedgeBuilder bb(rewriter, op.getLoc());
    circt::Backedge reg = bb.get(rewriter.getNoneType());

    // Generate a multiplexer to select the register's value
    Value mux = rewriter.create<comb::MuxOp>(op.getLoc(), adaptor.getReset(),
                                             constOne, reg);

    // Finally generate the register to set the backedge
    reg.setValue(rewriter.create<seq::CompRegOp>(
        op.getLoc(), mux, adaptor.getClock(), llvm::StringRef("hbr")));

    // Get rid of the old operation
    rewriter.eraseOp(op);
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
    Value prop = adaptor.getProperty();
    auto ltlclk = dyn_cast<ltl::ClockOp>(prop.getDefiningOp());

    // If it's not clocked then we can't generate a proper sv assertion
    if (!ltlclk)
      op->emitError("Verif assertions must be clocked!!");

    // Finish by genrating the parenting sv.always posedge clock from the ltl
    // clock, containing the generated sv.assert
    rewriter.create<sv::AlwaysOp>(
        op.getLoc(), (sv::EventControl)ltlclk.getEdge(), ltlclk.getClock(),
        [&] {
          // Generate the sv assertion using the input to the parenting clock
          rewriter.create<sv::AssertOp>(
              op.getLoc(), ltlclk.getInput(),
              sv::DeferAssertAttr::get(getContext(),
                                       sv::DeferAssert::Immediate),
              adaptor.getLabelAttr());
        });

    // Get rid of the two old operations
    rewriter.eraseOp(ltlclk);
    rewriter.eraseOp(op);

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
  patterns.add<DisableOpConversion, HasBeenResetOpConversion,
               VerifAssertOpConversion>(converter, patterns.getContext());
}

// Simply applies the conversion patterns defined above
void LowerLTLToCorePass::runOnOperation() {

  // Set target dialects: We don't want to see any ltl verif left in the result
  ConversionTarget target(getContext());
  target.addLegalDialect<HWDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<sv::SVDialect>();
  target.addLegalDialect<seq::SeqDialect>();
  target.addIllegalOp<verif::HasBeenResetOp>();
  target.addIllegalOp<verif::AssertOp>();
  target.addIllegalOp<ltl::DisableOp>();
  target.addIllegalOp<ltl::ClockOp>();

  // Create the operation rewrite patters
  RewritePatternSet patterns(&getContext());
  TypeConverter converter;
  converter.addConversion([](Type type) { return type; });
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
