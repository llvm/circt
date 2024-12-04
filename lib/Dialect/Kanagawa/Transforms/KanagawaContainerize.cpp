//===- KanagawaContainerize.cpp - Implementation of containerizing --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "mlir/Pass/Pass.h"

#include "circt/Dialect/Kanagawa/KanagawaDialect.h"
#include "circt/Dialect/Kanagawa/KanagawaOps.h"
#include "circt/Dialect/Kanagawa/KanagawaPasses.h"
#include "circt/Dialect/Kanagawa/KanagawaTypes.h"

#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"
#include "mlir/Transforms/DialectConversion.h"

namespace circt {
namespace kanagawa {
#define GEN_PASS_DEF_KANAGAWACONTAINERIZE
#include "circt/Dialect/Kanagawa/KanagawaPasses.h.inc"
} // namespace kanagawa
} // namespace circt

using namespace circt;
using namespace kanagawa;

namespace {

struct OutlineContainerPattern : public OpConversionPattern<ContainerOp> {
  OutlineContainerPattern(MLIRContext *context, Namespace &ns)
      : OpConversionPattern<ContainerOp>(context), ns(ns) {}

  using OpAdaptor = typename OpConversionPattern<ContainerOp>::OpAdaptor;

  LogicalResult
  matchAndRewrite(ContainerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Outline the container into the module scope, by prefixing it with the
    // parent class name.
    auto parentClass =
        dyn_cast_or_null<ClassOp>(op.getOperation()->getParentOp());
    assert(parentClass && "This pattern should never be called on a container"
                          "that is not nested within a class.");
    auto design = parentClass.getParentOp<DesignOp>();
    assert(design && "Parent class should be nested within a design.");

    rewriter.setInsertionPoint(parentClass);
    StringAttr newContainerName = rewriter.getStringAttr(
        ns.newName(parentClass.getInnerNameAttr().strref() + "_" +
                   op.getInnerNameAttr().strref()));
    auto newContainer = rewriter.create<ContainerOp>(
        op.getLoc(), newContainerName, /*isTopLevel=*/false);

    rewriter.mergeBlocks(op.getBodyBlock(), newContainer.getBodyBlock(), {});

    // Rename the kanagawa.this operation to refer to the proper op.
    auto thisOp =
        cast<ThisOp>(cast<ScopeOpInterface>(*newContainer.getOperation())
                         .getThis()
                         .getDefiningOp());
    rewriter.setInsertionPoint(thisOp);
    rewriter.replaceOpWithNewOp<ThisOp>(thisOp, design.getSymNameAttr(),
                                        newContainer.getInnerSymAttr());

    // Create a container instance op in the parent class.
    rewriter.setInsertionPoint(op);
    rewriter.create<ContainerInstanceOp>(
        parentClass.getLoc(), hw::InnerSymAttr::get(newContainerName),
        newContainer.getInnerRef());
    rewriter.eraseOp(op);
    return success();
  }

  Namespace &ns;
};

struct ClassToContainerPattern : public OpConversionPattern<ClassOp> {
  using OpConversionPattern<ClassOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ClassOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the class by a container of the same name.
    auto newContainer =
        rewriter.create<ContainerOp>(op.getLoc(), op.getInnerSymAttr(),
                                     /*topLevel*/ false, op.getNameAttr());
    rewriter.mergeBlocks(op.getBodyBlock(), newContainer.getBodyBlock(), {});
    rewriter.eraseOp(op);
    return success();
  }
};

struct InstanceToContainerInstancePattern
    : public OpConversionPattern<InstanceOp> {
  using OpConversionPattern<InstanceOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(InstanceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Replace the instance by a container instance of the same name.
    rewriter.replaceOpWithNewOp<ContainerInstanceOp>(op, op.getInnerSym(),
                                                     op.getTargetNameAttr());
    return success();
  }
};

/// Run all the physical lowerings.
struct ContainerizePass
    : public circt::kanagawa::impl::KanagawaContainerizeBase<ContainerizePass> {
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
  target.addLegalDialect<KanagawaDialect>();
  target.addDynamicallyLegalOp<ContainerOp>(
      [&](auto *op) { return !isa<kanagawa::ClassOp>(op->getParentOp()); });
  RewritePatternSet patterns(context);

  // Setup a namespace to ensure that the new container names are unique.
  // Grab existing names from the InnerSymbolTable of the top-level design op.
  SymbolCache symCache;
  (void)hw::InnerSymbolTable::walkSymbols(
      getOperation(), [&](StringAttr name, const hw::InnerSymTarget &target) {
        symCache.addDefinition(name, target.getOp());
        return success();
      });

  Namespace ns;
  symCache.addDefinitions(getOperation());
  ns.add(symCache);
  patterns.insert<OutlineContainerPattern>(context, ns);
  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

LogicalResult ContainerizePass::containerizeClasses() {
  auto *context = &getContext();
  ConversionTarget target(*context);
  target.addLegalDialect<KanagawaDialect>();
  target.addIllegalOp<ClassOp, InstanceOp>();
  RewritePatternSet patterns(context);
  patterns.insert<ClassToContainerPattern, InstanceToContainerInstancePattern>(
      context);
  return applyPartialConversion(getOperation(), target, std::move(patterns));
}

void ContainerizePass::runOnOperation() {
  if (failed(outlineContainers()) || failed(containerizeClasses()))
    signalPassFailure();
}

std::unique_ptr<Pass> circt::kanagawa::createContainerizePass() {
  return std::make_unique<ContainerizePass>();
}
