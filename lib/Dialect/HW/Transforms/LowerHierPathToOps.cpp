//===- LowerHierPathToOps.cpp - Lower hw.hierpath.to ops --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// Lowers `hw.hierpath.to` operations to `hw.hierpath` ops.
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

struct HierPathToOpConversionPattern
    : public OpConversionPattern<hw::HierPathToOp> {
  HierPathToOpConversionPattern(MLIRContext *ctx,
                                igraph::InstanceGraph &instanceGraph)
      : OpConversionPattern(ctx), instanceGraph(instanceGraph) {}

  LogicalResult
  matchAndRewrite(hw::HierPathToOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto parentOp = dyn_cast<igraph::ModuleOpInterface>(op->getParentOp());
    if (!parentOp)
      return op.emitError("hw.hierpath.to must be in an op implementing "
                          "igraph::ModuleOpInterface.");

    // Process the inner level.
    igraph::InstanceGraphNode *parentNode = instanceGraph.lookup(parentOp);
    llvm::SmallVector<Attribute> path{
        hw::InnerRefAttr::get(parentNode->getModule().getModuleNameAttr(),
                              op.getTargetAttr().getAttr())};

    while (parentNode) {
      // Verify that exactly one instance of the parent exists.
      size_t nUses = parentNode->getNumUses();
      if (nUses > 1) {
        auto err = op.emitError(
            "cannot lower hierpath.to ops in module hierarchies with "
            "multiple instantiations.");
        for (igraph::InstanceRecord *use : parentNode->uses())
          err.attachNote(use->getInstance().getLoc())
              << "instantiated here: " << use->getInstance();
        return err;
      }
      if (nUses == 0) {
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
}; // namespace

struct LowerHierPathToOpsPass
    : public LowerHierPathToOpsBase<LowerHierPathToOpsPass> {
  void runOnOperation() override;
};
} // namespace

void LowerHierPathToOpsPass::runOnOperation() {
  Operation *parent = getOperation();
  igraph::InstanceGraph &instanceGraph = getAnalysis<igraph::InstanceGraph>();

  ConversionTarget target(getContext());
  target.addLegalDialect<hw::HWDialect>();
  target.addIllegalOp<hw::HierPathToOp>();

  RewritePatternSet patterns(&getContext());
  patterns.add<HierPathToOpConversionPattern>(&getContext(), instanceGraph);

  if (failed(applyPartialConversion(parent, target, std::move(patterns))))
    signalPassFailure();
}

std::unique_ptr<mlir::Pass> circt::hw::createLowerHierPathToOpsPass() {
  return std::make_unique<LowerHierPathToOpsPass>();
}
