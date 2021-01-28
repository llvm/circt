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
#include "circt/Dialect/RTL/RTLDialect.h"
#include "circt/Dialect/RTL/RTLOps.h"
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
  RTLToLLHDTypeConverter(MLIRContext *context);

private:
  MLIRContext *context;
  size_t tmpCounter;
  StringAttr getTmpName();
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
struct ConvertInstance;

/// This is the main entrypoint for the RTL to LLHD conversion pass.
void RTLToLLHDPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  ConversionTarget target(context);
  target.addLegalDialect<LLHDDialect>();
  target.addIllegalOp<RTLModuleOp>();

  RTLToLLHDTypeConverter typeConverter(&context);
  OwningRewritePatternList patterns;
  populateFunctionLikeTypeConversionPattern<RTLModuleOp>(patterns, &context,
                                                         typeConverter);
  patterns.insert<ConvertRTLModule>(typeConverter, &context);
  patterns.insert<ConvertOutput>(typeConverter, &context);
  patterns.insert<ConvertInstance>(typeConverter, &context);

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

//===----------------------------------------------------------------------===//
// TypeConverter conversions and materializations
//===----------------------------------------------------------------------===//

RTLToLLHDTypeConverter::RTLToLLHDTypeConverter(MLIRContext *context) {
  // Set instance variables.
  this->context = context;
  this->tmpCounter = 0;

  // Convert IntegerType by just wrapping it in SigType.
  addConversion([](IntegerType type) { return SigType::get(type); });

  // Materialize SigType from IntegerType by wrapping with a SigOp.
  addTargetMaterialization([this](OpBuilder &builder, SigType type,
                                  ValueRange inputs,
                                  Location loc) -> Optional<Value> {
    if (inputs.size() != 1 || !inputs[0].getType().isa<IntegerType>())
      return llvm::None;

    return builder.createOrFold<SigOp>(loc, type, getTmpName(), inputs[0]);
  });
}

StringAttr RTLToLLHDTypeConverter::getTmpName() {
  SmallString<4> tmpName("tmp");
  tmpName += std::to_string(++tmpCounter);
  return StringAttr::get(tmpName, context);
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
    // Construct the `1d` time value for the drive.
    auto timeType = TimeType::get(rewriter.getContext());
    auto deltaAttr = TimeAttr::get(timeType, {0, 1, 0}, "ns");
    auto delta = rewriter.create<ConstOp>(output.getLoc(), timeType, deltaAttr);

    // Get the number of inputs in the entity to offset into the block args.
    auto entity = output->getParentOfType<EntityOp>();
    size_t numInputs = entity.ins();

    // Drive the results from the mapped operands.
    for (size_t i = 0, e = operands.size(); i != e; ++i) {
      // Get the source and destination signals.
      auto src = operands[i];
      auto dest = entity.getArgument(numInputs + i);
      assert(src && dest && "output operand must map to result block arg");

      // If the source has a signal type, probe it.
      if (auto sigTy = src.getType().dyn_cast<SigType>())
        src = rewriter.createOrFold<PrbOp>(output.getLoc(),
                                           sigTy.getUnderlyingType(), src);

      // Drive the destination block argument value.
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
    // instances model output ports as arguments to the op, so we need to create
    // SSA values. For each output port in the RTL instance create a new signal.
    SmallVector<Value, 4> results;
    for (auto result : instance.getResults()) {
      auto resultType = result.getType();
      if (!resultType.isa<IntegerType>())
        return rewriter.notifyMatchFailure(instance, [&](Diagnostic &diag) {
          diag << "result type " << resultType << " is not supported";
        });

      Location loc = result.getLoc();

      // Create a constant for the signal's initial value.
      auto init = rewriter.create<ConstOp>(
          loc, resultType, rewriter.getIntegerAttr(resultType, 0));

      // Create the signal itself.
      SmallString<8> sigName(instance.instanceName());
      sigName += "_result_";
      sigName += std::to_string(result.getResultNumber());
      auto sig = rewriter.createOrFold<SigOp>(loc, SigType::get(resultType),
                                              sigName, init);

      // Probe the signal, and replace the original result's uses with the
      // probed value.
      auto prb = rewriter.createOrFold<PrbOp>(loc, resultType, sig);

      for (auto &use : result.getUses())
        rewriter.updateRootInPlace(use.getOwner(), [&]() { use.set(prb); });

      results.push_back(sig);
    }

    // Create the LLHD instance from the RTL instance. An RTL instance inputs
    // and outputs are SSA inputs and outputs, but an LLHD instance inputs and
    // outputs are all SSA inputs. We initially leave the outputs empty, and
    // fill them in after they have been converted. Note that LLHD does not
    // support parameterized entities, so this conversion does not support
    // parameterized instances.
    rewriter.create<InstOp>(instance.getLoc(), instance.instanceName(),
                            instance.moduleName(), operands, results);

    rewriter.eraseOp(instance);

    return success();
  }
};
