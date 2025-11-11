//===- HWInliner.cpp - Inline HW modules ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the HWInliner pass.
//
// Algorithm:
// 1. Build hierarchical path table: Collect all hw.hierpath operations and
//    map them to affected modules to properly update paths during inlining.
//
// 2. Post-order traversal: Visit modules in the instance graph from leaves
//    to roots. This ensures that when module A is inlined into module B,
//    any modules instantiated by A have already been inlined into A first,
//    maximizing inlining opportunities in a single pass.
//
// 3. For each private module:
//    a. Check if it meets any inlining criteria (empty, no outputs,
//       single-use, small, etc.)
//    b. If yes, inline all instances of this module:
//       - Clone the module body into each instantiation site
//       - Remap values: connect instance inputs to cloned operations
//       - Rename symbols to avoid conflicts using InnerSymbolNamespace
//       - Update hierarchical paths by removing the inlined instance
//       - Replace instance outputs with the cloned operation results
//       - Remove the instance operation from the instance graph
//    c. If the module has no remaining uses, delete it
//
// 4. Repeat for all modules in post-order until all eligible modules are
//    inlined.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include <mlir/IR/MLIRContext.h>

#define DEBUG_TYPE "hw-inliner"

using namespace mlir;
using namespace circt;
using namespace hw;

namespace circt {
namespace hw {
#define GEN_PASS_DEF_HWINLINER
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using HierPathTable = DenseMap<hw::InnerRefAttr, SmallVector<hw::HierPathOp>>;

//===----------------------------------------------------------------------===//
// Inlining helper
//===----------------------------------------------------------------------===//

namespace {
class Inliner {
public:
  Inliner(hw::InstanceGraph &graph, HierPathTable &paths, MLIRContext *context)
      : graph(graph), paths(paths), context(context) {}

  LogicalResult copy(InstanceOp into, HWModuleOp from);

private:
  hw::InstanceGraph &graph;
  HierPathTable paths;

  // Cache InnerSymbolNamespace objects per parent module to avoid
  // recreating them for each instance in the same parent.
  DenseMap<HWModuleOp, std::unique_ptr<InnerSymbolNamespace>> nsCache;

  MLIRContext *context;
};
} // namespace

LogicalResult Inliner::copy(InstanceOp into, HWModuleOp from) {
  HWModuleOp parentModule(into->getParentOfType<HWModuleOp>());
  igraph::InstanceGraphNode *intoNode(graph[parentModule]);

  // Get or create the InnerSymbolNamespace for the parent module.
  auto &nsPtr = nsCache[parentModule];
  if (!nsPtr)
    nsPtr = std::make_unique<InnerSymbolNamespace>(parentModule);
  LLVM_DEBUG(llvm::dbgs() << "  - Into: " << into << "\n");

  Block *body = from.getBodyBlock();
  OutputOp term = cast<OutputOp>(body->getTerminator());

  // Map the arguments to the instance inputs.
  IRMapping mapping;
  for (auto [value, arg] : llvm::zip(into.getInputs(), body->getArguments()))
    mapping.map(arg, value);

  // Create a mapping for de-duplicated inner symbol names.
  DenseMap<StringAttr, StringAttr> symMapping;
  AttrTypeReplacer replacer;
  for (Operation &oldOp : *body) {
    oldOp.walk([&](Operation *op) {
      if (auto symAttr = op->getAttrOfType<hw::InnerSymAttr>(
              InnerSymbolTable::getInnerSymbolAttrName())) {
        auto symName = symAttr.getSymName();
        auto newSym =
            StringAttr::get(context, nsPtr->newName(symName.getValue()));
        symMapping.insert({symName, newSym});
      }
    });
  }
  replacer.addReplacement(
      [&](InnerRefAttr attr) -> std::pair<Attribute, WalkResult> {
        if (attr.getModule() != from.getModuleNameAttr())
          return {attr, WalkResult::skip()};

        auto it = symMapping.find(attr.getName());
        assert(it != symMapping.end() && "mapping not found");
        auto newAttr =
            InnerRefAttr::get(parentModule.getModuleNameAttr(), it->second);
        return {newAttr, WalkResult::skip()};
      });

  // Clone the body. This will leave arguments unchanged in the new function
  // if definitions are out of order (seq ops).  Will be adjusted later.
  OpBuilder builder(into);
  for (auto it = body->begin(); std::next(it) != body->end(); ++it) {
    Operation *oldOp = &*it;
    Operation *newOp = oldOp->clone(mapping);

    newOp->walk([&](Operation *op) {
      // Rename all user-visible names by prepending the instance name.
      if (!into.getInstanceName().empty()) {
        std::array<StringRef, 3> attrNames = {"name", "sv.namehint",
                                              "instanceName"};
        for (auto attrName : attrNames) {
          auto nameAttr = op->getAttrOfType<StringAttr>(attrName);
          if (nameAttr && !nameAttr.getValue().empty()) {
            std::string name;
            llvm::raw_string_ostream os(name);
            os << into.getInstanceName() << "." << nameAttr.getValue();
            op->setAttr(attrName, builder.getStringAttr(name));
          }
        }
      }
      // Using the mapping, rewrite symbols.
      if (auto symAttr = op->getAttrOfType<hw::InnerSymAttr>(
              InnerSymbolTable::getInnerSymbolAttrName())) {
        op->setAttr(InnerSymbolTable::getInnerSymbolAttrName(),
                    hw::InnerSymAttr::get(symMapping[symAttr.getSymName()]));
      }
      // Apply the replacements.
      replacer.replaceElementsIn(op);
    });

    builder.insert(newOp);
    if (auto newInstLike = dyn_cast<HWInstanceLike>(newOp)) {
      for (auto ref : newInstLike.getReferencedModuleNamesAttr()) {
        Operation *targetOp = graph.lookup(cast<StringAttr>(ref))->getModule();
        if (auto target = dyn_cast_or_null<HWModuleLike>(targetOp)) {
          intoNode->addInstance(newInstLike, graph.lookup(target));
        }
      }
    }
  }

  // Scan the parent for ops with operands pointing to the source module.
  // Rewrite them using the mapping, which should be fully populated now.
  parentModule.walk([&](Operation *op) {
    for (OpOperand &use : llvm::make_early_inc_range(op->getOpOperands())) {
      Value oldValue = use.get();
      Value newValue = mapping.lookupOrDefault(oldValue);

      if (oldValue != newValue)
        use.set(newValue);
    }
  });

  // Fix up the instance by replacing it with the arguments of the output.
  for (auto [result, value] : llvm::zip(into.getResults(), term.getOutputs())) {
    std::function<Value(Value)> chase = [&](Value value) -> Value {
      auto mapped = mapping.lookupOrNull(value);
      if (!mapped) {
        mlir::emitError(value.getLoc(), "cannot map result");
        return {};
      }
      if (mapped.getDefiningOp() != into)
        return mapped;
      auto resultNo = cast<OpResult>(mapped).getResultNumber();
      return chase(term.getOutputs()[resultNo]);
    };
    result.replaceAllUsesWith(chase(value));
  }

  // Adjust the hier path ops the instance is participating in.
  // Drop the instance references and rewrite the next symbol.
  if (auto sym = into.getInnerSymAttr()) {
    auto ref = into.getInnerRef();
    if (auto it = paths.find(ref); it != paths.end()) {
      for (hw::HierPathOp path : it->second) {
        SmallVector<Attribute, 4> newPath;
        for (auto elem : path.getNamepath()) {
          if (elem != ref)
            newPath.push_back(replacer.replace(elem));
        }
        path.setNamepathAttr(ArrayAttr::get(context, newPath));
      }
    }
  }

  into.erase();

  return success();
}

//===----------------------------------------------------------------------===//
// Pass Infrastructure
//===----------------------------------------------------------------------===//

namespace {
struct HWInlinerPass : public circt::hw::impl::HWInlinerBase<HWInlinerPass> {
  using HWInlinerBase::HWInlinerBase;

  void runOnOperation() override;

private:
  /// Determine if a module should be inlined based on various heuristics.
  bool shouldInline(HWModuleOp module, igraph::InstanceGraphNode *instanceNode,
                    size_t bodySize);
};
} // namespace

bool HWInlinerPass::shouldInline(HWModuleOp module,
                                 igraph::InstanceGraphNode *instanceNode,
                                 size_t bodySize) {
  // Check whether the module should be inlined based on heuristics.
  bool isEmpty = bodySize == 1;
  bool hasNoOutputs = module.getNumOutputPorts() == 0;
  bool hasOneUse = instanceNode->getNumUses() == 1;
  bool hasState = !module.getOps<seq::FirRegOp>().empty();
  bool isSmall = bodySize < this->smallThreshold;

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

  if (this->inlineSmall && isSmall)
    shouldInlineModule = true;

  return shouldInlineModule;
}

void HWInlinerPass::runOnOperation() {
  auto &graph = getAnalysis<hw::InstanceGraph>();

  // Build a mapping of hier path ops.
  DenseSet<StringAttr> leafModules;
  HierPathTable pathsTable;
  llvm::for_each(getOperation().getOps<hw::HierPathOp>(),
                 [&](hw::HierPathOp path) {
                   // Record leaf modules to be banned from inlining.
                   if (path.isModule())
                     leafModules.insert(path.leafMod());

                   // For each instance, record
                   for (auto name : path.getNamepath()) {
                     if (auto ref = dyn_cast<hw::InnerRefAttr>(name))
                       pathsTable[ref].push_back(path);
                   }
                 });

  for (auto *instanceNode : llvm::to_vector(llvm::post_order(&graph))) {
    // Find private modules.
    Operation *moduleLike = instanceNode->getModule();
    if (!moduleLike)
      continue;
    auto module = dyn_cast<HWModuleOp>(moduleLike);
    if (!module || module.isPublic())
      continue;
    // Do not delete a module if it is targeted by a module NLA.
    if (leafModules.count(module.getNameAttr()))
      continue;

    auto *body = module.getBodyBlock();
    size_t bodySize = std::distance(body->begin(), body->end());

    // Check whether the module should be inlined.
    if (!shouldInline(module, instanceNode, bodySize))
      continue;

    LLVM_DEBUG(llvm::dbgs() << "Inlining " << module.getModuleName() << "\n");

    Inliner inliner(graph, pathsTable, &getContext());
    // Find all instance sites.
    for (auto *record : llvm::make_early_inc_range(instanceNode->uses())) {
      Operation *instLike = record->getInstance();
      if (!instLike)
        continue;
      auto inst = dyn_cast_or_null<InstanceOp>(instLike);
      if (!inst)
        continue;

      if (failed(inliner.copy(inst, module)))
        return signalPassFailure();
      record->erase();
      numInlined++;
    }

    if (instanceNode->noUses()) {
      graph.erase(instanceNode);
      module.erase();
    }
  }
}
