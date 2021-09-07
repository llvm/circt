//===- ConvertToLLVM.cpp -  LLHD/HW/Comb to LLVM Conversion Pass ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main Lowering to LLVM Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/ConvertToLLVM/ConvertToLLVM.h"
#include "../PassDetail.h"
#include "CombToLLVM.h"
#include "HWToLLVM.h"
#include "LLHDToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/StandardToLLVM/ConvertStandardToLLVM.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

using namespace mlir;
using namespace circt;

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

namespace {
struct ConvertToLLVMLoweringPass
    : public ConvertToLLVMBase<ConvertToLLVMLoweringPass> {
  void runOnOperation() override;
};
} // namespace

void ConvertToLLVMLoweringPass::runOnOperation() {
  // Keep a counter to infer a signal's index in his entity's signal table.
  size_t sigCounter = 0;

  // Keep a counter to infer a reg's index in his entity.
  size_t regCounter = 0;

  RewritePatternSet patterns(&getContext());
  auto converter = mlir::LLVMTypeConverter(&getContext());
  LLVMConversionTarget target(getContext());

  // Register type conversions.
  populateHWToLLVMTypeConversions(converter);
  populateLLHDToLLVMTypeConversions(converter);

  // Register operation rewrite patterns and illegal ops for a partial pre-pass.
  setupPartialLLHDPrePass(converter, patterns, target);

  target.addLegalOp<UnrealizedConversionCastOp>();

  // Apply the partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();

  patterns.clear();

  // Register operation rewrite patterns for full conversion.
  populateStdToLLVMConversionPatterns(converter, patterns);
  populateHWToLLVMConversionPatterns(converter, patterns);
  populateCombToLLVMConversionPatterns(converter, patterns);
  populateLLHDToLLVMConversionPatterns(converter, patterns, sigCounter,
                                       regCounter);

  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addLegalOp<ModuleOp>();
  target.addIllegalOp<UnrealizedConversionCastOp>();

  // Apply the full conversion.
  if (failed(applyFullConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create an LLVM conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertToLLVMPass() {
  return std::make_unique<ConvertToLLVMLoweringPass>();
}
