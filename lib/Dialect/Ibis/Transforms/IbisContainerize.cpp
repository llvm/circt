//===- IbisContainerize.cpp - Implementation of containerizing ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"

#include "circt/Dialect/Ibis/IbisDialect.h"
#include "circt/Dialect/Ibis/IbisOps.h"
#include "circt/Dialect/Ibis/IbisPasses.h"
#include "circt/Dialect/Ibis/IbisTypes.h"

#include "circt/Support/SymCache.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace ibis;

// Iterates the symbolcache until a unique name is found.
static StringAttr getUniqueName(mlir::MLIRContext *ctx, StringAttr baseName,
                                SymbolCache &symCache) {
  StringAttr uniqueName = baseName;
  int uniqueCntr = 0;
  while (symCache.getDefinition(uniqueName))
    uniqueName =
        StringAttr::get(ctx, baseName.strref() + "_" + Twine(uniqueCntr++));
  return uniqueName;
}
namespace {

struct OutlineContainerPattern : public OpConversionPattern<ContainerOp> {
  OutlineContainerPattern(MLIRContext *context, SymbolCache *symCache)
      : OpConversionPattern<ContainerOp>(context), symCache(symCache) {}

  using OpAdaptor = typename OpConversionPattern<ContainerOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ContainerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Outline the container into the module scope, by prefixing it with the
    // parent class name.
    auto parentClass =
        dyn_cast_or_null<ClassOp>(op.getOperation()->getParentOp());
    if (!parentClass)
      return failure();

    rewriter.setInsertionPoint(parentClass);
    auto newContainerName =
        rewriter.getStringAttr(parentClass.getName() + "_" + op.getName());
    // unique it...
    newContainerName =
        getUniqueName(rewriter.getContext(), newContainerName, *symCache);
    auto newContainer =
        rewriter.create<ContainerOp>(op.getLoc(), newContainerName);
    symCache->addDefinition(newContainerName, newContainer);

    rewriter.mergeBlocks(op.getBodyBlock(), newContainer.getBodyBlock(), {});
    rewriter.eraseOp(op);

    // Rename the ibis.this operation to refer to the proper op.
    auto thisOp =
        cast<ThisOp>(cast<ScopeOpInterface>(*newContainer.getOperation())
                         .getThis()
                         .getDefiningOp());
    rewriter.setInsertionPoint(thisOp);
    rewriter.replaceOpWithNewOp<ThisOp>(thisOp, newContainer.getNameAttr());
    return success();
  }

  SymbolCache *symCache;
};

struct ClassToContainerPattern : public OpConversionPattern<ClassOp> {
  using OpConversionPattern<ClassOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClassOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the class by a container of the same name.
    auto newContainer =
        rewriter.create<ContainerOp>(op.getLoc(), op.getNameAttr());
    rewriter.mergeBlocks(op.getBodyBlock(), newContainer.getBodyBlock(), {});
    rewriter.eraseOp(op);
    return success();
  }
};

/// Run all the physical lowerings.
struct ContainerizePass : public IbisContainerizeBase<ContainerizePass> {
  void runOnOperation() override;

private:
  // Outlines containers nested within classes into the module scope.
  LogicalResult outlineContainers();

  // Converts classes to containers.
  LogicalResult containerizeClasses();
};
} // anonymous namespace

LogicalResult ContainerizePass::outlineContainers() {
  auto *context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<IbisDialect>();
  target.addDynamicallyLegalOp<ContainerOp>(
      [&](auto *op) { return isa<mlir::ModuleOp>(op->getParentOp()); });
  RewritePatternSet patterns(context);
  SymbolCache symCache;
  symCache.addDefinitions(getOperation());
  patterns.insert<OutlineContainerPattern>(context, &symCache);
  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

LogicalResult ContainerizePass::containerizeClasses() {
  auto *context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<IbisDialect>();
  target.addIllegalOp<ClassOp>();
  RewritePatternSet patterns(context);
  patterns.insert<ClassToContainerPattern>(context);
  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

void ContainerizePass::runOnOperation() {
  if (failed(outlineContainers()) || failed(containerizeClasses()))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::ibis::createContainerizePass() {
  return std::make_unique<ContainerizePass>();
}
