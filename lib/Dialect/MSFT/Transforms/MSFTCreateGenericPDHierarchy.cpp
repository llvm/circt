//===- MSFTCreateGenericPDHierarchy.cpp - PD hierarchy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetails.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWTypes.h"
#include "circt/Dialect/MSFT/MSFTDialect.h"
#include "circt/Dialect/MSFT/MSFTOpInterfaces.h"
#include "circt/Dialect/MSFT/MSFTOps.h"
#include "circt/Dialect/MSFT/MSFTPasses.h"
#include "circt/Support/InstanceGraph.h"
#include "circt/Support/Namespace.h"
#include "circt/Support/SymCache.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/ADT/PostOrderIterator.h"

using namespace circt;
using namespace msft;
using namespace mlir;

namespace {
struct CreateGenericPDHierarchyPass
    : public CreateGenericPDHierarchyBase<CreateGenericPDHierarchyPass> {
  void runOnOperation() override;

  FailureOr<msft::InstanceHierarchyOp>
  lower(OpBuilder &b, SymbolTable symbolTable, hw::HWModuleLike mod);

  Namespace ns;
  DenseMap<Operation *, msft::InstanceHierarchyOp> hierarchyOpForModule;
};
} // anonymous namespace

FailureOr<msft::InstanceHierarchyOp>
CreateGenericPDHierarchyPass::lower(OpBuilder &b, SymbolTable symbolTable,
                                    hw::HWModuleLike mod) {
  OpBuilder::InsertionGuard g(b);
  msft::InstanceHierarchyOp hierarchyOpForMod;
  auto getOrCreateHierOp = [&]() -> msft::InstanceHierarchyOp {
    if (hierarchyOpForMod)
      return hierarchyOpForMod;

    OpBuilder::InsertionGuard guard(b);
    b.setInsertionPoint(mod);
    hierarchyOpForMod = b.create<msft::InstanceHierarchyOp>(
        mod.getLoc(),
        b.getStringAttr(ns.newName(mod.getName() + "_generic_pd_config")),
        FlatSymbolRefAttr::get(mod.getNameAttr()),
        /*inst_name*/ StringAttr{});
    hierarchyOpForMod.getBodyRegion().push_back(new Block);
    return hierarchyOpForMod;
  };

  auto addToConfig = [&](llvm::function_ref<void(OpBuilder & b)> f) {
    b.setInsertionPointToEnd(getOrCreateHierOp().getBody());
    f(b);
  };

  // Look for module-local annotations
  for (auto &op : llvm::make_early_inc_range(*mod.getBodyBlock())) {
    llvm::TypeSwitch<Operation *, void>(&op)
        .Case<msft::PDMulticycleOp>([&](auto mcOp) {
          addToConfig([&](OpBuilder &b) {
            mcOp->moveBefore(b.getInsertionBlock(), b.getInsertionPoint());

            // Also create hierpaths... needed for TCL emission but obviously
            // redundant and should be rethought.
            b.setInsertionPoint(hierarchyOpForMod);
            auto srcHierPath = b.create<hw::HierPathOp>(
                mcOp.getLoc(),
                b.getStringAttr(mod.getName() + "." + mcOp.getSource()),
                b.getArrayAttr(
                    llvm::SmallVector<Attribute>{hw::InnerRefAttr::get(
                        mod.getNameAttr(), mcOp.getSourceAttr().getAttr())}));
            auto dstHierPath = b.create<hw::HierPathOp>(
                mcOp.getLoc(),
                b.getStringAttr(mod.getName() + "." + mcOp.getDest()),
                b.getArrayAttr(
                    llvm::SmallVector<Attribute>{hw::InnerRefAttr::get(
                        mod.getNameAttr(), mcOp.getDestAttr().getAttr())}));
            mcOp.setSource(srcHierPath.getName());
            mcOp.setDest(dstHierPath.getName());
          });
        })
        .Default([&](auto) {});
  }

  // Look for instantiations of modules with local annotations
  for (auto instanceLike : mod.getBodyBlock()->getOps<hw::HWInstanceLike>()) {
    hw::HWModuleLike refMod =
        cast<hw::HWModuleLike>(instanceLike.getReferencedModule(symbolTable));
    if (auto hierarchyOp = hierarchyOpForModule.lookup(refMod)) {
      addToConfig([&](OpBuilder &b) {
        hw::HierPathOp instanceHierPathOp;
        {
          OpBuilder::InsertionGuard g(b);
          b.setInsertionPoint(mod);
          instanceHierPathOp = b.create<hw::HierPathOp>(
              instanceLike.getLoc(),
              b.getStringAttr(mod.getName() + "." +
                              instanceLike.getInstanceName()),
              b.getArrayAttr(llvm::SmallVector<Attribute>{hw::InnerRefAttr::get(
                  mod.getNameAttr(), instanceLike.getInstanceNameAttr())}));
        }

        b.create<msft::PDInstanceHierarchyCallOp>(
            instanceLike.getLoc(),
            FlatSymbolRefAttr::get(hierarchyOp.getSymNameAttr()),
            FlatSymbolRefAttr::get(instanceHierPathOp.getNameAttr()));
      });
    }
  }

  return hierarchyOpForMod;
}

void CreateGenericPDHierarchyPass::runOnOperation() {
  auto top = getOperation();
  auto *ctxt = &getContext();
  SymbolTable symbolTable(top);
  OpBuilder b(ctxt);

  // Populate the top level symbol cache.
  SymbolCache topSyms;
  topSyms.addDefinitions(top);
  ns.add(topSyms);

  // Generate an instance graph
  auto &instanceGraph = getAnalysis<igraph::InstanceGraph>();

  // 1. Iterate the instance graph in post order
  // 2. determine if there's any module-local annotations OR
  //    if there's instantiations of any modules with local annotations
  //    (recursively)
  // 3. If so, create a hierarchy op for the module and populate it with
  //    the annotations
  DenseSet<Operation *> visited;
  for (auto *startNode : instanceGraph) {
    if (visited.contains(startNode->getModule()))
      continue;

    for (igraph::InstanceGraphNode *node : llvm::post_order(startNode)) {
      if (!visited.insert(node->getModule().getOperation()).second)
        continue;

      auto res = lower(b, symbolTable, node->getModule<hw::HWModuleLike>());
      if (failed(res)) {
        signalPassFailure();
        return;
      }
      hierarchyOpForModule[node->getModule().getOperation()] = *res;
    }
  }

  hierarchyOpForModule.clear();
}

namespace circt {
namespace msft {
std::unique_ptr<Pass> createCreateGenericPDHierarchyPass() {
  return std::make_unique<CreateGenericPDHierarchyPass>();
}
} // namespace msft
} // namespace circt
