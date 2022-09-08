//===- HWToSystemC.cpp - HW To SystemC Conversion Pass --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main HW to SystemC Conversion Pass Implementation.
//
//===----------------------------------------------------------------------===//

#include "circt/Conversion/HWToSystemC.h"
#include "../PassDetail.h"
#include "circt/Dialect/Comb/CombDialect.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace circt;
using namespace hw;
using namespace systemc;

//===----------------------------------------------------------------------===//
// Operation Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

/// This works on each HW module, creates corresponding SystemC modules, moves
/// the body of the module into the new SystemC module by splitting up the body
/// into field declarations, initializations done in a newly added systemc.ctor,
/// and internal methods to be registered in the constructor.
struct ConvertHWModule : public OpConversionPattern<HWModuleOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(HWModuleOp module, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Parameterized modules are supported yet.
    if (!module.getParameters().empty())
      return emitError(module->getLoc(), "module parameters not supported yet");

    if (llvm::any_of(module.getAllPorts(),
                     [](auto port) { return port.isInOut(); }))
      return emitError(module->getLoc(), "inout arguments not supported yet");

    // Create the SystemC module.
    SmallVector<PortInfo> ports = module.getAllPorts();
    for (size_t i = 0; i < ports.size(); ++i)
      ports[i].type = typeConverter->convertType(ports[i].type);

    auto scModule = rewriter.create<SCModuleOp>(module.getLoc(),
                                                module.getNameAttr(), ports);
    scModule.setVisibility(module.getVisibility());

    SmallVector<Attribute> portAttrs;
    if (auto argAttrs = module.getAllArgAttrs())
      portAttrs.append(argAttrs.begin(), argAttrs.end());
    else
      portAttrs.append(module.getNumInputs(), Attribute());
    if (auto resultAttrs = module.getAllResultAttrs())
      portAttrs.append(resultAttrs.begin(), resultAttrs.end());
    else
      portAttrs.append(module.getNumOutputs(), Attribute());

    scModule.setAllArgAttrs(portAttrs);

    // Create a systemc.func operation inside the module after the ctor.
    // TODO: implement logic to extract a better name and properly unique it.
    rewriter.setInsertionPointToStart(scModule.getBodyBlock());
    auto scFunc = rewriter.create<SCFuncOp>(
        module.getLoc(), rewriter.getStringAttr("innerLogic"));

    // Inline the HW module body into the systemc.func body.
    // TODO: do some dominance analysis to detect use-before-def and cycles in
    // the use chain, which are allowed in graph regions but not in SSACFG
    // regions, and when possible fix them.
    scFunc.getBodyBlock()->erase();
    Region &scFuncBody = scFunc.getBody();
    rewriter.inlineRegionBefore(module.getBody(), scFuncBody, scFuncBody.end());

    // Register the systemc.func inside the systemc.ctor
    rewriter.setInsertionPointToStart(
        scModule.getOrCreateCtor().getBodyBlock());
    rewriter.create<MethodOp>(scModule.getLoc(), scFunc.getHandle());

    // Move the block arguments of the systemc.func (that we got from the
    // hw.module) to the systemc.module
    rewriter.setInsertionPointToStart(scFunc.getBodyBlock());
    for (size_t i = 0, e = scFunc.getRegion().getNumArguments(); i < e; ++i) {
      auto inputRead =
          rewriter
              .create<SignalReadOp>(scFunc.getLoc(), scModule.getArgument(i))
              .getResult();
      auto converted = typeConverter->materializeSourceConversion(
          rewriter, scModule.getLoc(), module.getAllPorts()[i].type, inputRead);
      scFuncBody.getArgument(0).replaceAllUsesWith(converted);
      scFuncBody.eraseArgument(0);
    }

    // Erase the HW module.
    rewriter.eraseOp(module);

    return success();
  }
};

/// Convert output operations to alias operations connecting the result SSA
/// values to the output block arguments.
struct ConvertOutput : public OpConversionPattern<OutputOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OutputOp outputOp, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    if (auto scModule = outputOp->getParentOfType<SCModuleOp>()) {
      for (auto args :
           llvm::zip(scModule.getOutputPorts(), adaptor.getOperands())) {
        rewriter.create<SignalWriteOp>(outputOp->getLoc(), std::get<0>(args),
                                       std::get<1>(args));
      }

      // Erase the HW OutputOp.
      rewriter.eraseOp(outputOp);
      return success();
    }

    return failure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Conversion Infrastructure
//===----------------------------------------------------------------------===//

static void populateLegality(ConversionTarget &target) {
  target.addIllegalDialect<HWDialect>();
  target.addLegalDialect<mlir::BuiltinDialect>();
  target.addLegalDialect<systemc::SystemCDialect>();
  target.addLegalDialect<comb::CombDialect>();
  target.addLegalDialect<emitc::EmitCDialect>();
  target.addLegalOp<hw::ConstantOp>();
  target.addLegalOp<UnrealizedConversionCastOp>();
}

static void populateOpConversion(RewritePatternSet &patterns,
                                 TypeConverter &typeConverter) {
  patterns.add<ConvertHWModule, ConvertOutput>(typeConverter,
                                               patterns.getContext());
}

static void populateTypeConversion(TypeConverter &converter) {
  converter.addConversion([](IntegerType type) -> Type {
    auto bw = type.getIntOrFloatBitWidth();
    if (bw == 1)
      return type;
    if (bw <= 64 && type.isSigned())
      return systemc::IntType::get(type.getContext(), bw);
    if (bw <= 64)
      return UIntType::get(type.getContext(), bw);
    if (bw <= 512 && type.isSigned())
      return BigIntType::get(type.getContext(), bw);
    if (bw <= 512)
      return BigUIntType::get(type.getContext(), bw);

    return BitVectorType::get(type.getContext(), bw);
  });
  converter.addConversion([](UIntType type) { return type; });
  converter.addSourceMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        auto op = builder.create<ConvertOp>(loc, type, values[0]);
        return op.getResult();
      });
  converter.addTargetMaterialization(
      [](OpBuilder &builder, Type type, ValueRange values, Location loc) {
        assert(values.size() == 1);
        auto op = builder.create<ConvertOp>(loc, type, values[0]);
        return op.getResult();
      });
}

//===----------------------------------------------------------------------===//
// HW to SystemC Conversion Pass
//===----------------------------------------------------------------------===//

namespace {
struct HWToSystemCPass : public ConvertHWToSystemCBase<HWToSystemCPass> {
  void runOnOperation() override;
};
} // namespace

/// Create a HW to SystemC dialects conversion pass.
std::unique_ptr<OperationPass<ModuleOp>> circt::createConvertHWToSystemCPass() {
  return std::make_unique<HWToSystemCPass>();
}

/// This is the main entrypoint for the HW to SystemC conversion pass.
void HWToSystemCPass::runOnOperation() {
  MLIRContext &context = getContext();
  ModuleOp module = getOperation();

  // Create the include operation here to have exactly one 'systemc' include at
  // the top instead of one per module.
  OpBuilder builder(module.getRegion());
  builder.create<emitc::IncludeOp>(module->getLoc(), "systemc.h", true);

  ConversionTarget target(context);
  TypeConverter typeConverter;
  RewritePatternSet patterns(&context);
  populateLegality(target);
  populateTypeConversion(typeConverter);
  populateOpConversion(patterns, typeConverter);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
