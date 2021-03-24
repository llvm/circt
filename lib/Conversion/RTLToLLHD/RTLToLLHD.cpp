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
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertRTLToLLHDPass() {
  return std::make_unique<RTLToLLHDPass>();
}

/// Forward declare conversion patterns.
struct ConvertRTLModule;
struct ConvertOutput;
struct ConvertInstance;

/// This is the main entrypoint for the RTL to LLHD conversion pass.
void RTLToLLHDPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(context);
  target.addLegalDialect<LLHDDialect>();
  target.addIllegalOp<RTLModuleOp>();

  RTLToLLHDTypeConverter typeConverter;
  OwningRewritePatternList patterns;
  mlir::populateFunctionLikeTypeConversionPattern<RTLModuleOp>(
      patterns, &context, typeConverter);
  patterns.insert<ConvertRTLModule>(typeConverter, &context);
  patterns.insert<ConvertOutput>(typeConverter, &context);
  patterns.insert<ConvertInstance>(typeConverter, &context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// TypeConverter conversions and materializations
//===----------------------------------------------------------------------===//

RTLToLLHDTypeConverter::RTLToLLHDTypeConverter() {
  // Convert IntegerType by just wrapping it in SigType.
  addConversion([](IntegerType type) { return SigType::get(type); });

  // Mark SigType legal by converting it to itself.
  addConversion([](SigType type) { return type; });
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

    // Create the entity. Note that LLHD does not support parameterized
    // entities, so this conversion does not support parameterized modules.
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

/// This works on each instance op, converting them to the LLHD dialect. If the
/// RTL instance ops were defined in terms of the CallableOpInterface, we could
/// generalize this in terms of the upstream pattern to rewrite call ops' types.
struct ConvertInstance : public OpConversionPattern<InstanceOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp instance, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    // RTL instances model output ports as SSA results produced by the op. LLHD
    // instances model output ports as arguments to the op, so we need to find
    // or create SSA values. For each output port in the RTL instance, try to
    // find a signal that can be used directly, or else create a new signal.
    SmallVector<Value, 4> results;
    for (auto result : instance.getResults()) {
      auto resultType = result.getType();
      if (!resultType.isa<IntegerType>())
        return rewriter.notifyMatchFailure(instance, [&](Diagnostic &diag) {
          diag << "result type " << resultType << " is not supported";
        });

      Location loc = result.getLoc();

      Value init;
      for (auto &use : result.getUses()) {
        Value sig;

        if (isa<OutputOp>(use.getOwner())) {
          // If the result is an entity output, use that signal.
          auto entity = instance->getParentOfType<EntityOp>();
          sig = entity.getArgument(entity.ins() + use.getOperandNumber());
        } else {
          // Otherwise, materialize a signal.
          if (!init)
            init = rewriter.create<ConstOp>(
                loc, resultType, rewriter.getIntegerAttr(resultType, 0));

          SmallString<8> sigName(instance.instanceName());
          sigName += "_result_";
          sigName += std::to_string(result.getResultNumber());
          sig = rewriter.createOrFold<SigOp>(loc, SigType::get(resultType),
                                             sigName, init);
        }

        // Replace all uses of this result with the signal.
        rewriter.updateRootInPlace(use.getOwner(), [&]() { use.set(sig); });

        results.push_back(sig);
      }
    }

    // Create the LLHD instance from the operands and results.
    rewriter.create<InstOp>(instance.getLoc(), instance.instanceName(),
                            instance.moduleName(), operands, results);

    rewriter.eraseOp(instance);

    return success();
  }
};
