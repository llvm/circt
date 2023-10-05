//===- LowerPathToHereOps.cpp - Lower hw.path.to_here ops -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Lowers `hw.path.to_here` operations to `hw.hierpath` ops.
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Support/InstanceGraph.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace circt;
using namespace hw;

namespace {

struct PathToHereOpConversionPattern
    : public OpConversionPattern<hw::PathToHereOp> {
  PathToHereOpConversionPattern(MLIRContext *ctx,
                                igraph::InstanceGraph &instanceGraph)
      : OpConversionPattern(ctx), instanceGraph(instanceGraph) {}

  LogicalResult
  matchAndRewrite(hw::PathToHereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentOp = dyn_cast<igraph::ModuleOpInterface>(op->getParentOp());
    if (!parentOp)
      return op.emitError("hw.path.to_here must be in an op implementing "
                          "igraph::ModuleOpInterface.");
    igraph::InstanceGraphNode *parentNode = instanceGraph.lookup(parentOp);
    llvm::SmallVector<Attribute> path;
    while (parentNode) {
      // Verify that exactly one instance of the parent exists.
      size_t nUses = parentNode->getNumUses();
      if (nUses > 1) {
        auto err = op.emitError(
            "cannot lower path.to_here ops in module hierarchies with "
            "multiple instantiations.");
        for (igraph::InstanceRecord *use : parentNode->uses())
          err.attachNote(use->getInstance().getLoc())
              << "instantiated here: " << use->getInstance();
        return err;
      }
      if (nUses == 0) {
        if (path.empty())
          return op.emitError(
              "cannot lower path.to_here ops in modules with no "
              "instantiations.");
        // End of hierarchy.
        break;
      }

      igraph::InstanceRecord *instanceNode = *parentNode->uses().begin();
      igraph::InstanceGraphNode *instantiatedIn = instanceNode->getParent();
      if (!instantiatedIn) {
        // We've recursed up to the top of the instance graph.
        break;
      }

      igraph::InstanceOpInterface instance = instanceNode->getInstance();
      path.insert(
          path.begin(),
          hw::InnerRefAttr::get(instantiatedIn->getModule().getModuleNameAttr(),
                                instance.getInstanceNameAttr()));

      // Recurse up to parent of parent, if any.
      parentNode = instantiatedIn;
    }

    // Create a new path op.
    rewriter.replaceOpWithNewOp<hw::HierPathOp>(op, op.getName(),
                                                rewriter.getArrayAttr(path));
    return success();
  }

  igraph::InstanceGraph &instanceGraph;
};

struct LowerPathToHereOpsPass
    : public LowerPathToHereOpsBase<LowerPathToHereOpsPass> {
  void runOnOperation() override;
};
} // namespace

void LowerPathToHereOpsPass::runOnOperation() {
  Operation *parent = getOperation();
  igraph::InstanceGraph &instanceGraph = getAnalysis<igraph::InstanceGraph>();

  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addIllegalOp<hw::PathToHereOp>();

  RewritePatternSet patterns(&getContext());
  patterns.add<PathToHereOpConversionPattern>(&getContext(), instanceGraph);

  if (failed(applyPartialConversion(parent, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::hw::createLowerPathToHereOpsPass() {
  return std::make_unique<LowerPathToHereOpsPass>();
}
