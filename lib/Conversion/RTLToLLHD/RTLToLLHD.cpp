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
#include "../PassDetail.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/RTL/RTLDialect.h"
#include "circt/Dialect/RTL/RTLOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

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
circt::createConvertRTLToLLHDPass() {
  return std::make_unique<RTLToLLHDPass>();
}

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
  populateFunctionLikeTypeConversionPattern<RTLModuleOp>(patterns, &context,
                                                         typeConverter);
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
    // Collect the RTL module's port types.
    FunctionType moduleType = module.getType();
    unsigned numInputs = moduleType.getNumInputs();
    TypeRange moduleInputs = moduleType.getInputs();
    TypeRange moduleOutputs = moduleType.getResults();

    // LLHD entities port types are all expressed as block arguments to the op,
    // so collect all of the types in the expected order (inputs then outputs).
    SmallVector<Type, 4> entityTypes(moduleInputs);
    entityTypes.append(moduleOutputs.begin(), moduleOutputs.end());

    // Ensure the input and output types have all been converted already. This
    // is handled separately by the upstream FunctionLikeTypeConversionPattern.
    if (!llvm::all_of(entityTypes,
                      [](Type type) { return type.isa<SigType>(); }))
      return rewriter.notifyMatchFailure(module, "Not all ports had SigType");

    // Create the entity.
    auto entity = rewriter.create<EntityOp>(module.getLoc(), numInputs);

    // Inline the RTL module body into the entity body.
    Region &entityBodyRegion = entity.getBodyRegion();
    rewriter.inlineRegionBefore(module.getBodyRegion(), entityBodyRegion,
                                entityBodyRegion.end());

    // Add the LLHD terminator op after the RTL module's output ops.
    rewriter.setInsertionPointToEnd(entity.getBodyBlock());
    rewriter.create<llhd::TerminatorOp>(entity.getLoc());

    // Set the entity type and name attributes. Add block arguments for each
    // output, since LLHD entity outputs are still block arguments to the op.
    auto entityType = rewriter.getFunctionType(entityTypes, {});
    rewriter.updateRootInPlace(entity, [&] {
      entity->setAttr(entity.getTypeAttrName(), TypeAttr::get(entityType));
      entity.setName(module.getName());
      entityBodyRegion.addArguments(moduleOutputs);
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
    // Get the number of inputs in the entity to offset into the block args.
    auto entity = output->getParentOfType<EntityOp>();
    size_t numInputs = entity.ins();

    // Drive the results from the mapped operands.
    Value delta;
    for (size_t i = 0, e = operands.size(); i != e; ++i) {
      // Get the source and destination signals.
      auto src = operands[i];
      auto dest = entity.getArgument(numInputs + i);
      if (!src || !dest)
        return rewriter.notifyMatchFailure(
            output, "output operand must map to result block arg");

      // If the source has a signal type, connect it.
      if (auto sigTy = src.getType().dyn_cast<SigType>()) {
        rewriter.create<llhd::ConnectOp>(output.getLoc(), dest, src);
        continue;
      }

      // Otherwise, drive the destination block argument value.
      if (!delta) {
        auto timeType = TimeType::get(rewriter.getContext());
        auto deltaAttr = TimeAttr::get(timeType, {0, 1, 0}, "ns");
        delta = rewriter.create<ConstOp>(output.getLoc(), timeType, deltaAttr);
      }
      rewriter.create<DrvOp>(output.getLoc(), dest, src, delta, Value());
    }

    // Erase the original output terminator.
    rewriter.eraseOp(output);

    return success();
  }
};
