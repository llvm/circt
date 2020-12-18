//===- RTLToLLHD.cpp - RTL to LLHD Conversion Pass ------------------------===//
//
// This is the main RTL to LLHD Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/RTLToLLHD/RTLToLLHD.h"
#include "circt/Dialect/LLHD/IR/LLHDDialect.h"
#include "circt/Dialect/LLHD/IR/LLHDOps.h"
#include "circt/Dialect/RTL/Dialect.h"
#include "circt/Dialect/RTL/Ops.h"
#include "mlir/IR/StandardTypes.h"
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
// Convert modules
//===----------------------------------------------------------------------===//

/// This works on each RTL module, creates corresponding entities, moves the
/// bodies of the modules into the entities, and converts the bodies.
struct ConvertRTLModule : public OpConversionPattern<RTLModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(RTLModuleOp module, ArrayRef<Value> operands,
                                ConversionPatternRewriter &rewriter) const {
    // Collect the RTL module's port types in a signature conversion.
    FunctionType moduleType = module.getType();
    unsigned numInputs = moduleType.getNumInputs();
    TypeConverter::SignatureConversion argConversion(module.getNumArguments() +
                                                     module.getNumResults());

    if (failed(typeConverter->convertSignatureArgs(moduleType.getInputs(),
                                                   argConversion)) ||
        failed(typeConverter->convertSignatureArgs(moduleType.getResults(),
                                                   argConversion, numInputs)))
      return failure();

    // Create the entity.
    auto entity = rewriter.create<EntityOp>(module.getLoc(), numInputs);

    // Inline the RTL module body into the entity body.
    Region &entityBodyRegion = entity.getBodyRegion();
    rewriter.inlineRegionBefore(module.getBodyRegion(), entityBodyRegion,
                                entityBodyRegion.end());

    // Attempt to convert the entry block's args using the previous conversion.
    if (failed(rewriter.convertRegionTypes(&entityBodyRegion, *typeConverter,
                                           &argConversion)))
      return failure();

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
