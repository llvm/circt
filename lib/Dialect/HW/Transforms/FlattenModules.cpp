//===- FlattenModules.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Support/BackedgeBuilder.h"
#include "mlir/IR/AttrTypeSubElements.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/Inliner.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/PostOrderIterator.h"

#define DEBUG_TYPE "hw-flatten-modules"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_FLATTENMODULES
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace circt;
using namespace hw;
using namespace igraph;
using mlir::InlinerConfig;
using mlir::InlinerInterface;

using HierPathTable = DenseMap<hw::InnerRefAttr, SmallVector<hw::HierPathOp>>;

namespace {

// Cache the inner symbol attribute name to avoid repeated lookups
static const StringRef innerSymAttrName =
    InnerSymbolTable::getInnerSymbolAttrName();
struct FlattenModulesPass
    : public circt::hw::impl::FlattenModulesBase<FlattenModulesPass> {
  using FlattenModulesBase::FlattenModulesBase;

  void runOnOperation() override;

private:
  /// Determine if a module should be inlined based on various heuristics.
  bool shouldInline(HWModuleOp module, igraph::InstanceGraphNode *instanceNode,
                    size_t bodySize);
};

/// A simple implementation of the `InlinerInterface` that marks all inlining as
/// legal since we know that we only ever attempt to inline `HWModuleOp` bodies
/// at `InstanceOp` sites.
struct PrefixingInliner : public InlinerInterface {
  StringRef prefix;
  InnerSymbolNamespace *ns;
  HWModuleOp parentModule;
  HWModuleOp sourceModule;
  DenseMap<StringAttr, StringAttr> *symMapping;
  mlir::AttrTypeReplacer *replacer;
  HierPathTable *pathsTable;
  hw::InnerRefAttr instanceRef;

  PrefixingInliner(MLIRContext *context, StringRef prefix,
                   InnerSymbolNamespace *ns, HWModuleOp parentModule,
                   HWModuleOp sourceModule,
                   DenseMap<StringAttr, StringAttr> *symMapping,
                   mlir::AttrTypeReplacer *replacer, HierPathTable *pathsTable,
                   hw::InnerRefAttr instanceRef)
      : InlinerInterface(context), prefix(prefix), ns(ns),
        parentModule(parentModule), sourceModule(sourceModule),
        symMapping(symMapping), replacer(replacer), pathsTable(pathsTable),
        instanceRef(instanceRef) {}

  bool isLegalToInline(Region *dest, Region *src, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }
  bool isLegalToInline(Operation *op, Region *dest, bool wouldBeCloned,
                       IRMapping &valueMapping) const override {
    return true;
  }
  void handleTerminator(Operation *op,
                        mlir::ValueRange valuesToRepl) const override {
    assert(isa<hw::OutputOp>(op));
    for (auto [from, to] : llvm::zip(valuesToRepl, op->getOperands()))
      from.replaceAllUsesWith(to);
  }

  void processInlinedBlocks(
      iterator_range<Region::iterator> inlinedBlocks) override {
    for (Block &block : inlinedBlocks)
      block.walk([&](Operation *op) {
        updateNames(op);
        updateInnerSymbols(op);
      });

    // Update hierarchical paths that reference the inlined instance
    updateHierPaths();
  }

  void updateHierPaths() const {
    // If the instance has an inner symbol, update any hierarchical paths
    // that reference it
    if (!instanceRef)
      return;

    auto it = pathsTable->find(instanceRef);
    if (it == pathsTable->end())
      return;

    // For each hierarchical path that references this instance
    for (hw::HierPathOp path : it->second) {
      SmallVector<Attribute, 4> newPath;
      for (auto elem : path.getNamepath()) {
        // Skip the instance reference being inlined
        if (elem != instanceRef)
          newPath.push_back(replacer->replace(elem));
      }
      path.setNamepathAttr(ArrayAttr::get(path.getContext(), newPath));
    }
  }

  StringAttr updateName(StringAttr attr) const {
    if (attr.getValue().empty())
      return attr;
    return StringAttr::get(attr.getContext(), prefix + "/" + attr.getValue());
  }

  void updateNames(Operation *op) const {
    if (auto name = op->getAttrOfType<StringAttr>("name"))
      op->setAttr("name", updateName(name));
    if (auto name = op->getAttrOfType<StringAttr>("instanceName"))
      op->setAttr("instanceName", updateName(name));
    if (auto namesAttr = op->getAttrOfType<ArrayAttr>("names")) {
      SmallVector<Attribute> names(namesAttr.getValue().begin(),
                                   namesAttr.getValue().end());
      for (auto &name : names)
        if (auto nameStr = dyn_cast<StringAttr>(name))
          name = updateName(nameStr);
      op->setAttr("names", ArrayAttr::get(namesAttr.getContext(), names));
    }
  }

  void updateInnerSymbols(Operation *op) const {
    // Rename inner symbols to avoid conflicts
    if (auto innerSymAttr =
            op->getAttrOfType<hw::InnerSymAttr>(innerSymAttrName)) {
      StringAttr symName = innerSymAttr.getSymName();
      auto it = symMapping->find(symName);
      if (it != symMapping->end())
        op->setAttr(innerSymAttrName, hw::InnerSymAttr::get(it->second));
    }

    // Apply attribute replacements for InnerRefAttr
    replacer->replaceElementsIn(op);
  }

  bool allowSingleBlockOptimization(
      iterator_range<Region::iterator> inlinedBlocks) const final {
    return true;
  }
};
} // namespace

bool FlattenModulesPass::shouldInline(HWModuleOp module,
                                      igraph::InstanceGraphNode *instanceNode,
                                      size_t bodySize) {
  // If inlineAll is enabled, inline everything (default behavior)
  if (this->inlineAll)
    return true;

  // Check whether the module should be inlined based on heuristics.
  bool isEmpty = bodySize == 1;
  bool hasNoOutputs = module.getNumOutputPorts() == 0;
  bool hasOneUse = instanceNode->getNumUses() == 1;
  bool hasState = !module.getOps<seq::FirRegOp>().empty();

  // Don't inline modules with state unless explicitly allowed
  if (hasState && !this->inlineWithState)
    return false;

  // Inline if any of the enabled conditions are met:
  bool shouldInlineModule = false;

  if (this->inlineEmpty && isEmpty)
    shouldInlineModule = true;

  if (this->inlineNoOutputs && hasNoOutputs)
    shouldInlineModule = true;

  if (this->inlineSingleUse && hasOneUse)
    shouldInlineModule = true;

  if (this->inlineSmall && bodySize < this->smallThreshold)
    shouldInlineModule = true;

  return shouldInlineModule;
}

void FlattenModulesPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<hw::InstanceGraph>();
  DenseSet<Operation *> handled;

  InlinerConfig config;

  // Build a mapping of hierarchical path ops.
  DenseSet<StringAttr> leafModules;
  HierPathTable pathsTable;
  llvm::for_each(getOperation().getOps<hw::HierPathOp>(),
                 [&](hw::HierPathOp path) {
                   // Record leaf modules to be banned from inlining.
                   if (path.isModule())
                     leafModules.insert(path.leafMod());

                   // For each instance in the path, record the path
                   for (auto name : path.getNamepath()) {
                     if (auto ref = dyn_cast<hw::InnerRefAttr>(name))
                       pathsTable[ref].push_back(path);
                   }
                 });

  // Cache InnerSymbolNamespace objects per parent module to avoid
  // recreating them for each instance in the same parent.
  DenseMap<HWModuleOp, std::unique_ptr<InnerSymbolNamespace>> nsCache;

  // Iterate over all instances in the instance graph. This ensures we visit
  // every module, even private top modules (private and never instantiated).
  for (auto *startNode : instanceGraph) {
    if (handled.count(startNode->getModule().getOperation()))
      continue;

    // Visit the instance subhierarchy starting at the current module, in a
    // depth-first manner. This allows us to inline child modules into parents
    // before we attempt to inline parents into their parents.
    for (InstanceGraphNode *node : llvm::post_order(startNode)) {
      if (!handled.insert(node->getModule().getOperation()).second)
        continue;

      unsigned numUsesLeft = node->getNumUses();
      if (numUsesLeft == 0)
        continue;

      // Only inline private `HWModuleOp`s (no extern or generated modules).
      auto module =
          dyn_cast_or_null<HWModuleOp>(node->getModule().getOperation());
      if (!module || !module.isPrivate())
        continue;

      // Do not inline a module if it is targeted by a module NLA.
      if (leafModules.count(module.getNameAttr()))
        continue;

      // Check if module should be inlined based on heuristics
      auto *body = module.getBodyBlock();
      size_t bodySize = std::distance(body->begin(), body->end());
      if (!shouldInline(module, node, bodySize))
        continue;

      // Build symbol mapping for the module before inlining any instances
      DenseMap<StringAttr, StringAttr> inlineModuleInnerSyms;
      mlir::AttrTypeReplacer innerRefReplacer;

      // Scan the module body to collect all inner symbols that need renaming
      for (Operation &oldOp : *body) {
        oldOp.walk([&](Operation *op) {
          if (auto innerSymAttr =
                  op->getAttrOfType<hw::InnerSymAttr>(innerSymAttrName))
            inlineModuleInnerSyms.insert(
                {innerSymAttr.getSymName(), StringAttr()});
        });
      }

      for (auto *instRecord : node->uses()) {
        // Only inline at plain old HW `InstanceOp`s.
        auto inst = dyn_cast_or_null<InstanceOp>(
            instRecord->getInstance().getOperation());
        if (!inst)
          continue;

        bool isLastModuleUse = --numUsesLeft == 0;

        // Get the parent module
        HWModuleOp parentModule = inst->getParentOfType<HWModuleOp>();

        // Get or create the InnerSymbolNamespace for the parent module
        auto &nsPtr = nsCache[parentModule];
        if (!nsPtr)
          nsPtr = std::make_unique<InnerSymbolNamespace>(parentModule);

        // Create fresh symbol names for this instance
        DenseMap<StringAttr, StringAttr> oldToNewInnerSyms;
        for (auto [oldSym, _] : inlineModuleInnerSyms)
          oldToNewInnerSyms.insert(
              {oldSym, StringAttr::get(&getContext(),
                                       nsPtr->newName(oldSym.getValue()))});

        // Setup the replacer for InnerRefAttr
        mlir::AttrTypeReplacer instanceReplacer;
        instanceReplacer.addReplacement(
            [&](InnerRefAttr attr) -> std::pair<Attribute, WalkResult> {
              if (attr.getModule() != module.getModuleNameAttr())
                return {attr, WalkResult::skip()};

              auto it = oldToNewInnerSyms.find(attr.getName());
              if (it == oldToNewInnerSyms.end())
                return {attr, WalkResult::skip()};

              auto newAttr = InnerRefAttr::get(parentModule.getModuleNameAttr(),
                                               it->second);
              return {newAttr, WalkResult::skip()};
            });

        // Get the instance's inner reference if it has one
        hw::InnerRefAttr instanceRef;
        if (auto sym = inst.getInnerSymAttr())
          instanceRef = inst.getInnerRef();

        PrefixingInliner inliner(&getContext(), inst.getInstanceName(),
                                 nsPtr.get(), parentModule, module,
                                 &oldToNewInnerSyms, &instanceReplacer,
                                 &pathsTable, instanceRef);

        if (failed(mlir::inlineRegion(inliner, config.getCloneCallback(),
                                      &module.getBody(), inst,
                                      inst.getOperands(), inst.getResults(),
                                      std::nullopt, !isLastModuleUse))) {
          inst.emitError("failed to inline '")
              << module.getModuleName() << "' into instance '"
              << inst.getInstanceName() << "'";
          return signalPassFailure();
        }

        inst.erase();
        if (isLastModuleUse)
          module->erase();
      }
    }
  }
}
