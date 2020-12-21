//===- RTLToLLHD.cpp - RTL to LLHD Conversion Pass ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main RTL to LLHD Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/RTLToLLHD/RTLToLLHD.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/RTL/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace llhd {
#define GEN_PASS_CLASSES
#include "circt/Conversion/RTLToLLHD/Passes.h.inc"
} // namespace llhd
} // namespace circt

using namespace circt;
using namespace llhd;
using namespace rtl;

//===----------------------------------------------------------------------===//
// RTL to LLHD Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct RTLToLLHDPass : public ConvertRTLToLLHDBase<RTLToLLHDPass> {
  void runOnOperation() override;
};

/// A helper type converter class that automatically populates the relevant
/// materializations and type conversions for converting RTL to LLHD.
struct RTLToLLHDTypeConverter : public TypeConverter {
  RTLToLLHDTypeConverter();
};
} // namespace

/// Create a RTL to LLHD conversion pass.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
llhd::createConvertRTLToLLHDPass() {
  return std::make_unique<RTLToLLHDPass>();
}

/// Register the RTL to LLHD conversion pass.
namespace {
#define GEN_PASS_REGISTRATION
#include "circt/Conversion/RTLToLLHD/Passes.h.inc"
} // namespace

void circt::llhd::registerRTLToLLHDPasses() { registerPasses(); }

/// Forward declare conversion patterns.
struct ConvertRTLModule;
struct ConvertOutput;

/// This is the main entrypoint for the RTL to LLHD conversion pass.
void RTLToLLHDPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(context);
  target.addLegalDialect<LLHDDialect>();
  target.addIllegalOp<RTLModuleOp>();

  RTLToLLHDTypeConverter typeConverter;
  OwningRewritePatternList patterns;
  patterns.insert<ConvertRTLModule>(typeConverter, &context);
  patterns.insert<ConvertOutput>(typeConverter, &context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// TypeConverter conversions and materializations
//===----------------------------------------------------------------------===//
RTLToLLHDTypeConverter::RTLToLLHDTypeConverter() {
  // Convert IntegerType by just wrapping it in SigType.
  addConversion([](IntegerType type) { return SigType::get(type); });
}

//===----------------------------------------------------------------------===//
// Convert structure operations
//===----------------------------------------------------------------------===//

/// This works on each RTL module, creates corresponding entities, moves the
/// bodies of the modules into the entities, and converts the bodies.
struct ConvertRTLModule : public OpConversionPattern<RTLModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RTLModuleOp module, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Collect the RTL module's port types in a signature conversion.
    FunctionType moduleType = module.getType();
    unsigned numInputs = moduleType.getNumInputs();
    TypeConverter::SignatureConversion argConversion(module.getNumArguments() +
                                                     module.getNumResults());

    if (failed(typeConverter->convertSignatureArgs(moduleType.getInputs(),
                                                   argConversion)) ||
        failed(typeConverter->convertSignatureArgs(moduleType.getResults(),
                                                   argConversion, numInputs)))
      return module.emitError("type converter failed to convert signature");

    // Create the entity.
    auto entity = rewriter.create<EntityOp>(module.getLoc(), numInputs);

    // Inline the RTL module body into the entity body.
    Region &entityBodyRegion = entity.getBodyRegion();
    rewriter.inlineRegionBefore(module.getBodyRegion(), entityBodyRegion,
                                entityBodyRegion.end());

    // Attempt to convert the entry block's args using the previous conversion.
    if (failed(rewriter.convertRegionTypes(&entityBodyRegion, *typeConverter,
                                           &argConversion)))
      return module.emitError("could not convert region types");

    // Add the LLHD terminator op after the RTL module's output ops.
    rewriter.setInsertionPointToEnd(entity.getBodyBlock());
    rewriter.create<llhd::TerminatorOp>(entity.getLoc());

    // Set the entity type and name attributes.
    auto entityType =
        rewriter.getFunctionType(argConversion.getConvertedTypes(), {});
    rewriter.updateRootInPlace(entity, [&] {
      entity.setAttr(entity.getTypeAttrName(), TypeAttr::get(entityType));
      entity.setName(module.getName());
    });

    // Erase the RTL module.
    rewriter.eraseOp(module);

    return success();
  }
};

/// This works on each output op, creating ops to drive the appropriate results.
struct ConvertOutput : public OpConversionPattern<OutputOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp output, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // Construct the `1d` time value for the drive.
    auto timeType = TimeType::get(rewriter.getContext());
    auto deltaAttr = TimeAttr::get(timeType, {0, 1, 0}, "ns");
    auto delta = rewriter.create<ConstOp>(output.getLoc(), timeType, deltaAttr);

    // Get the number of inputs in the entity to offset into the block args.
    auto entity = output.getParentOfType<EntityOp>();
    size_t numInputs = entity.ins();

    // Drive the results from the mapped operands.
    for (size_t i = 0, e = operands.size(); i != e; ++i) {
      // Get the source and destination signals.
      auto src = operands[i];
      auto dest = entity.getArgument(numInputs + i);
      assert(src && dest && "output operand must map to result block arg");

      // If the source has a signal type, probe it.
      if (auto sigTy = src.getType().dyn_cast<SigType>())
        src = rewriter.create<PrbOp>(output.getLoc(), sigTy.getUnderlyingType(),
                                     src);

      // Drive the destination block argument value.
      rewriter.create<DrvOp>(output.getLoc(), dest, src, delta, Value());
    }

    // Erase the original output terminator.
    rewriter.eraseOp(output);

    return success();
  }
};
