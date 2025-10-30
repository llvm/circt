//===- CombToLLVM.cpp - Comb to LLVM Conversion Patterns ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements Comb to LLVM conversion patterns.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToLLVM.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace circt;

namespace {

//===----------------------------------------------------------------------===//
// Comb Operation Conversion Patterns
//===----------------------------------------------------------------------===//

/// Convert a comb::ParityOp to the LLVM dialect.
/// This is the only Comb operation that doesn't have a Comb-to-Arith pattern.
struct CombParityOpConversion : public ConvertToLLVMPattern {
  explicit CombParityOpConversion(MLIRContext *ctx,
                                  LLVMTypeConverter &typeConverter)
      : ConvertToLLVMPattern(comb::ParityOp::getOperationName(), ctx,
                             typeConverter) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    auto parityOp = cast<comb::ParityOp>(op);

    auto popCount =
        LLVM::CtPopOp::create(rewriter, op->getLoc(), parityOp.getInput());
    rewriter.replaceOpWithNewOp<LLVM::TruncOp>(
        op, IntegerType::get(rewriter.getContext(), 1), popCount);

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population Functions
//===----------------------------------------------------------------------===//

void circt::populateCombToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                                 RewritePatternSet &patterns) {
  // Only add patterns for operations that don't have Comb-to-Arith patterns
  // Most Comb operations are handled by the Comb-to-Arith + Arith-to-LLVM
  // pipeline
  patterns.add<CombParityOpConversion>(patterns.getContext(), converter);
}
