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
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
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

  LogicalResult
  matchAndRewrite(ltl::DisableOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    /*rewriter.replaceOpWithNewOp<smt::ConstantOp>(
        op, smt::BitVectorAttr::get(
                getContext(), adaptor.getValue().getZExtValue(),
                smt::BitVectorType::get(getContext(),
                                        op.getType().getIntOrFloatBitWidth())));*/
    return success();
  }
};

struct VerifAssertOpConversion : OpConversionPattern<verif::AssertOp> {
  using OpConversionPattern<verif::AssertOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(verif::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    /*Value solver = op->getParentOfType<func::FuncOp>()
                       .getFunctionBody()
                       .front()
                       .getArgument(0);
    Value constOne = rewriter.create<smt::ConstantOp>(
        op.getLoc(),
        smt::BitVectorAttr::get(getContext(), 0,
                                smt::BitVectorType::get(getContext(), 1)));
    Value cond = rewriter.create<smt::EqOp>(op.getLoc(), adaptor.getProperty(),
                                            constOne);
    rewriter.replaceOpWithNewOp<smt::AssertOp>(op, solver, cond);*/
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

void LowerLTLToCorePass::runOnOperation() {}

// Basic default constructor
std::unique_ptr<mlir::Pass> circt::createLowerLTLToCorePass() {
  return std::make_unique<LowerLTLToCorePass>();
}
