//===- InteropLoweringPatterns.cpp - Interop Lowering Patterns ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements rewrite patterns for container-side lowering and interop
// mechanism bridging to be used by downstream dialects and tools.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Interop/InteropLoweringPatterns.h"
#include "circt/Dialect/Interop/InteropOpInterfaces.h"
#include "circt/Dialect/Interop/InteropOps.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace circt::interop;

//===----------------------------------------------------------------------===//
// Container Interop Lowering Patterns
//===----------------------------------------------------------------------===//

namespace {
/// Query the ProceduralContainerInteropOpInterface of the closest surrounding
/// operation for the allocation lowering.
struct ProceduralAllocOpConversion : OpConversionPattern<ProceduralAllocOp> {
  using OpConversionPattern<ProceduralAllocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProceduralAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto interopParent =
            op->getParentOfType<ProceduralContainerInteropOpInterface>())
      return interopParent.lowerAllocation(rewriter, op, adaptor);

    return failure();
  }
};

/// Query the ProceduralContainerInteropOpInterface of the closest surrounding
/// operation for the initialization lowering.
struct ProceduralInitOpConversion : OpConversionPattern<ProceduralInitOp> {
  using OpConversionPattern<ProceduralInitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProceduralInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto interopParent =
            op->getParentOfType<ProceduralContainerInteropOpInterface>())
      return interopParent.lowerInitialization(rewriter, op, adaptor);

    return failure();
  }
};

/// Query the ProceduralContainerInteropOpInterface of the closest surrounding
/// operation for the update lowering.
struct ProceduralUpdateOpConversion : OpConversionPattern<ProceduralUpdateOp> {
  using OpConversionPattern<ProceduralUpdateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProceduralUpdateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto interopParent =
            op->getParentOfType<ProceduralContainerInteropOpInterface>())
      return interopParent.lowerUpdate(rewriter, op, adaptor);

    return failure();
  }
};

/// Query the ProceduralContainerInteropOpInterface of the closest surrounding
/// operation for the deallocation lowering.
struct ProceduralDeallocOpConversion
    : OpConversionPattern<ProceduralDeallocOp> {
  using OpConversionPattern<ProceduralDeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProceduralDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (auto interopParent =
            op->getParentOfType<ProceduralContainerInteropOpInterface>())
      return interopParent.lowerDeallocation(rewriter, op, adaptor);

    return failure();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Population function implementations
//===----------------------------------------------------------------------===//

void interop::populateContainerInteropPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<ProceduralAllocOpConversion, ProceduralInitOpConversion,
               ProceduralUpdateOpConversion, ProceduralDeallocOpConversion>(
      ctx);
}
