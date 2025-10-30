//===- CombToLLVM.cpp - Comb to LLVM Conversion Pass ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Comb to LLVM Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/CombToLLVM.h"
#include "circt/Conversion/CombToArith.h"
#include "circt/Conversion/HWToLLVM.h"
#include "circt/Dialect/Comb/CombOps.h"
#include "circt/Dialect/SV/SVOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;

namespace circt {
#define GEN_PASS_DEF_CONVERTTOLLVM
#include "circt/Conversion/Passes.h.inc"
} // namespace circt

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

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct CombToLLVMLoweringPass
    : public circt::impl::ConvertToLLVMBase<CombToLLVMLoweringPass> {
  void runOnOperation() override;

private:
  void convertFuncOp(func::FuncOp funcOp);
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

void CombToLLVMLoweringPass::runOnOperation() {
  // Iterate over all func.func operations in the module and convert them
  for (auto funcOp :
       llvm::make_early_inc_range(getOperation().getOps<func::FuncOp>())) {
    convertFuncOp(funcOp);
  }
}

void CombToLLVMLoweringPass::convertFuncOp(func::FuncOp funcOp) {
  MLIRContext *context = &getContext();
  RewritePatternSet patterns(context);
  auto converter = mlir::LLVMTypeConverter(context);

  // Add HW to LLVM type conversions
  populateHWToLLVMTypeConversions(converter);

  LLVMConversionTarget target(*context);
  target.addIllegalDialect<comb::CombDialect>();
  target.addIllegalDialect<arith::ArithDialect>();

  // Mark HW and SV dialects as legal - we only convert operations inside
  // func.func
  target.addLegalDialect<hw::HWDialect>();
  target.addLegalDialect<sv::SVDialect>();

  // Setup the conversion patterns in the correct order:
  // 1. SCF to ControlFlow (for structured control flow)
  populateSCFToControlFlowConversionPatterns(patterns);

  // 2. Func to LLVM (for function operations)
  populateFuncToLLVMConversionPatterns(converter, patterns);

  // 3. ControlFlow to LLVM (for control flow operations)
  cf::populateControlFlowToLLVMConversionPatterns(converter, patterns);

  // 4. Comb to Arith (for most combinational operations)
  populateCombToArithConversionPatterns(converter, patterns);

  // 5. Arith to LLVM (for arithmetic operations)
  arith::populateArithToLLVMConversionPatterns(converter, patterns);

  // 6. Index to LLVM (for index operations)
  index::populateIndexToLLVMConversionPatterns(converter, patterns);

  // 7. Any function op interface type conversion
  populateAnyFunctionOpInterfaceTypeConversionPattern(patterns, converter);

  // 8. Comb to LLVM (for operations without Comb-to-Arith patterns, like
  // parity)
  populateCombToLLVMConversionPatterns(converter, patterns);

  // Apply the partial conversion only to this func.func operation
  if (failed(applyPartialConversion(funcOp, target, std::move(patterns))))
    signalPassFailure();
}

/// Create a Comb to LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertCombToLLVMPass() {
  return std::make_unique<CombToLLVMLoweringPass>();
}
