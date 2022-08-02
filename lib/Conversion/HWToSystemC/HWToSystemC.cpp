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
    if (module.getParameters().size() > 0)
      return emitError(module->getLoc(), "module parameters not supported yet");

    // Collect the HW module's port types.
    FunctionType moduleType = module.getFunctionType();
    TypeRange moduleOutputs = moduleType.getResults();

    // SystemC module port types are all expressed as block arguments to the op,
    // so collect all of the types and add them as arguments to the SystemC
    // module.
    SmallVector<Type, 4> portTypes((TypeRange)moduleType.getInputs());
    portTypes.append(moduleOutputs.begin(), moduleOutputs.end());

    // Collect all the port directions.
    SmallVector<systemc::PortDirection> directions;
    for (auto port : module.getAllPorts()) {
      if (port.isInput())
        directions.push_back(systemc::PortDirection::Input);
      else if (port.isOutput())
        directions.push_back(systemc::PortDirection::Output);
      else
        return emitError(module->getLoc(), "inout arguments not supported yet");
    }
    PortDirectionsAttr portDirections =
        PortDirectionsAttr::get(rewriter.getContext(), directions);

    // Collect all the port names (inputs and outputs).
    SmallVector<Attribute> portNames;
    ArrayRef<Attribute> args = module.getArgNames().getValue();
    ArrayRef<Attribute> results = module.getResultNames().getValue();
    portNames.append(args.begin(), args.end());
    portNames.append(results.begin(), results.end());

    // Create the SystemC module.
    auto scModule = rewriter.create<SCModuleOp>(
        module.getLoc(), portDirections,
        ArrayAttr::get(rewriter.getContext(), portNames));

    // Create a systemc.ctor operation inside the module.
    Block *moduleBlock = new Block;
    scModule.getBodyRegion().push_back(moduleBlock);
    rewriter.setInsertionPointToStart(moduleBlock);
    auto ctor = rewriter.create<CtorOp>(module.getLoc());

    // Create a systemc.func operation inside the module after the ctor.
    // TODO: implement logic to extract a better name and properly unique it.
    auto scFunc = rewriter.create<SCFuncOp>(
        module.getLoc(), rewriter.getStringAttr("innerLogic"));

    // Inline the HW module body into the systemc.func body.
    // TODO: do some dominance analysis to detect use-before-def and cycles in
    // the use chain, which are allowed in graph regions but not in SSACFG
    // regions, and when possible fix them.
    Region &scFuncBody = scFunc.getBody();
    rewriter.inlineRegionBefore(module.getBody(), scFuncBody, scFuncBody.end());

    // Register the systemc.func inside the systemc.ctor
    Block *ctorBlock = new Block;
    ctor.getRegion().push_back(ctorBlock);
    rewriter.setInsertionPointToStart(ctorBlock);
    rewriter.create<MethodOp>(ctor.getLoc(), scFunc.getHandle());

    // Set the SCModule type and name attributes. Add block arguments for each
    // output.
    auto scModuleType = rewriter.getFunctionType(portTypes, {});
    rewriter.updateRootInPlace(scModule, [&] {
      scModule->setAttr(scModule.getTypeAttrName(),
                        TypeAttr::get(scModuleType));
      scModule.setName(module.getName());
      scModule.getBodyRegion().addArguments(
          portTypes,
          SmallVector<Location, 4>(portTypes.size(), rewriter.getUnknownLoc()));

      // Move the block arguments of the systemc.func (that we got from the
      // hw.module) to the systemc.module
      Region &funcBody = scFunc.getBody();
      for (size_t i = 0, e = scFunc.getRegion().getNumArguments(); i < e; ++i) {
        funcBody.getArgument(0).replaceAllUsesWith(scModule.getArgument(i));
        funcBody.eraseArgument(0);
      }
    });

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

    SmallVector<Value> moduleOutputs;
    if (auto scModule = outputOp->getParentOfType<SCModuleOp>()) {
      scModule.getOutputs(moduleOutputs);
      for (auto args : llvm::zip(moduleOutputs, outputOp.getOperands())) {
        rewriter.create<AliasOp>(outputOp->getLoc(), std::get<0>(args),
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
  target.addLegalOp<hw::ConstantOp>();
}

static void populateOpConversion(RewritePatternSet &patterns) {
  patterns.add<ConvertHWModule, ConvertOutput>(patterns.getContext());
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

  ConversionTarget target(context);
  TypeConverter typeConverter;
  RewritePatternSet patterns(&context);
  populateLegality(target);
  populateOpConversion(patterns);

  if (failed(applyFullConversion(module, target, std::move(patterns))))
    signalPassFailure();
}
