//===- SystemCContainerInteropLowering.cpp - Lower container-side interop -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the main SystemC container-side interop lowering pass implementation.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/Interop/InteropOps.h"
#include "circt/Dialect/SystemC/SystemCOpInterfaces.h"
#include "circt/Dialect/SystemC/SystemCOps.h"
#include "circt/Dialect/SystemC/SystemCPasses.h"
#include "circt/Support/Namespace.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace circt;
using namespace circt::systemc;
using namespace circt::interop;

namespace {
struct SCModuleNamespace : Namespace {
  SCModuleNamespace() = default;
  SCModuleNamespace(SCModuleOp module) { add(module); }

  void add(SCModuleOp module) {
    for (auto portName : module.getPortNames())
      nextIndex.insert({portName.cast<StringAttr>().getValue(), 0});

    module->walk([&](SystemCNameDeclOpInterface op) {
      nextIndex.insert({op.getName(), 0});
    });
  }
};

template <typename OpTy>
class UniquingOpConversionPattern : public OpConversionPattern<OpTy> {
public:
  UniquingOpConversionPattern(SCModuleNamespace &uniquer, MLIRContext *context,
                              PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(context, benefit), nameUniquer(uniquer) {}
  UniquingOpConversionPattern(SCModuleNamespace &uniquer,
                              TypeConverter &typeConverter,
                              MLIRContext *context, PatternBenefit benefit = 1)
      : OpConversionPattern<OpTy>(typeConverter, context, benefit),
        nameUniquer(uniquer) {}

protected:
  SCModuleNamespace &nameUniquer;
};

class ProceduralAllocOpConversion
    : public UniquingOpConversionPattern<ProceduralAllocOp> {
public:
  using UniquingOpConversionPattern<
      ProceduralAllocOp>::UniquingOpConversionPattern;

  LogicalResult
  matchAndRewrite(ProceduralAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    Operation *interopParent = op->getParentWithTrait<HasInterop>();
    if (!interopParent)
      return failure();
    SCModuleOp module = dyn_cast<SCModuleOp>(interopParent);
    if (!module)
      return failure();

    auto stateBuilder = OpBuilder::atBlockBegin(module.getBodyBlock());
    SmallVector<Value> variables;
    for (Value state : op.getStates()) {
      Value variable = stateBuilder.create<VariableOp>(
          op.getLoc(), state.getType(),
          StringAttr::get(rewriter.getContext(),
                          nameUniquer.newName("interopState")),
          Value());
      variables.push_back(variable);
    }

    rewriter.replaceOp(op, variables);
    return success();
  }
};

class ProceduralInitOpConversion
    : public OpConversionPattern<ProceduralInitOp> {
public:
  using OpConversionPattern<ProceduralInitOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProceduralInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *interopParent = op->getParentWithTrait<HasInterop>();
    if (!interopParent)
      return failure();
    SCModuleOp module = dyn_cast<SCModuleOp>(interopParent);
    if (!module)
      return failure();

    rewriter.mergeBlocks(op.getBody(),
                         &module.getOrCreateCtor().getBody().front(), {});

    auto initBuilder =
        OpBuilder::atBlockEnd(&module.getOrCreateCtor().getBody().front());

    auto returnOp = cast<interop::ReturnOp>(
        &module.getOrCreateCtor().getBody().front().back());

    for (auto [state, val] :
         llvm::zip(adaptor.getStates(), returnOp.getReturnValues())) {
      initBuilder.create<AssignOp>(module.getLoc(), state, val);
    }

    rewriter.eraseOp(returnOp);
    rewriter.eraseOp(op);
    return success();
  }
};

class ProceduralUpdateOpConversion
    : public OpConversionPattern<ProceduralUpdateOp> {
public:
  using OpConversionPattern<ProceduralUpdateOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProceduralUpdateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *interopParent = op->getParentWithTrait<HasInterop>();
    if (!interopParent)
      return failure();
    SCModuleOp module = dyn_cast<SCModuleOp>(interopParent);
    if (!module)
      return failure();

    auto returnOp = cast<interop::ReturnOp>(op.getBody()->getTerminator());

    SmallVector<Value> replacements(adaptor.getStates());
    for (auto input : adaptor.getInputs())
      replacements.push_back(input);
    rewriter.mergeBlockBefore(op.getBody(), op, replacements);

    rewriter.replaceOp(op, returnOp.getReturnValues());
    rewriter.eraseOp(returnOp);
    return success();
  }
};

class ProceduralDeallocOpConversion
    : public OpConversionPattern<ProceduralDeallocOp> {
public:
  using OpConversionPattern<ProceduralDeallocOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ProceduralDeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *interopParent = op->getParentWithTrait<HasInterop>();
    if (!interopParent)
      return failure();
    SCModuleOp module = dyn_cast<SCModuleOp>(interopParent);
    if (!module)
      return failure();

    rewriter.mergeBlocks(op.getBody(),
                         &module.getOrCreateDestructor().getBody().front(),
                         adaptor.getStates());

    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace

//===----------------------------------------------------------------------===//
// Pass initialization
//===----------------------------------------------------------------------===//

namespace {
struct SystemCContainerInteropLoweringPass
    : SystemCContainerInteropLoweringBase<SystemCContainerInteropLoweringPass> {
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override;
};
} // namespace

void SystemCContainerInteropLoweringPass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<SystemCDialect, interop::InteropDialect, emitc::EmitCDialect,
                  func::FuncDialect>();
}

void SystemCContainerInteropLoweringPass::runOnOperation() {
  RewritePatternSet patterns(&getContext());

  ConversionTarget target(getContext());
  target.addLegalDialect<emitc::EmitCDialect>();
  target.addLegalDialect<SystemCDialect>();
  target.addLegalOp<func::CallIndirectOp>();
  target.addDynamicallyLegalOp<ProceduralAllocOp, ProceduralInitOp,
                               ProceduralUpdateOp, ProceduralDeallocOp>(
      [](Operation *op) {
        if (auto *parent = op->getParentWithTrait<HasInterop>())
          return !isa<SCModuleOp>(parent);
        return false;
      });

  SCModuleNamespace nameUniquer;
  getOperation()->walk([&](SCModuleOp module) -> WalkResult {
    nameUniquer.add(module);
    return WalkResult::skip();
  });

  // Setup the conversion.
  // interop::populateContainerInteropPatterns(patterns, &getContext());

  patterns.add<ProceduralAllocOpConversion>(nameUniquer, &getContext());
  patterns.add<ProceduralInitOpConversion, ProceduralUpdateOpConversion,
               ProceduralDeallocOpConversion>(&getContext());

  // Apply the partial conversion.
  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}

/// Create the SystemC Lower Interop pass.
std::unique_ptr<Pass>
circt::systemc::createSystemCContainerInteropLoweringPass() {
  return std::make_unique<SystemCContainerInteropLoweringPass>();
}
