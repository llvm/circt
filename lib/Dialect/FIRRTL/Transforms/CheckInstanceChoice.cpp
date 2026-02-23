//===- CheckInstanceChoice.cpp - Check instance choice configurations -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/InstanceChoiceInfo.h"
#include "circt/Dialect/FIRRTL/Passes.h"
#include "circt/Support/LLVM.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/PostOrderIterator.h"

namespace circt {
namespace firrtl {
#define GEN_PASS_DEF_CHECKINSTANCECHOICE
#include "circt/Dialect/FIRRTL/Passes.h.inc"
} // namespace firrtl
} // namespace circt

using namespace circt;
using namespace firrtl;
using namespace mlir;

LogicalResult InstanceChoiceInfo::run() {
  for (auto &op : *circuit.getBodyBlock()) {
    auto module = dyn_cast<FModuleLike>(op);
    if (!module || !module.isPublic())
      continue;

    auto *rootNode = instanceGraph.lookup(module);
    if (!rootNode)
      continue;

    alwaysReachable[module].insert(module);
    computeAlwaysReachable(rootNode, module);

    moduleChoices[module][module] = {};

    DenseSet<igraph::InstanceGraphNode *> visited;
    SmallVector<igraph::InstanceGraphNode *> postOrderNodes;
    for (auto *node : llvm::post_order_ext(rootNode, visited))
      postOrderNodes.push_back(node);

    for (auto *node : llvm::reverse(postOrderNodes))
      if (failed(processNode(node, module)))
        return failure();
  }

  return success();
}

void InstanceChoiceInfo::computeAlwaysReachable(igraph::InstanceGraphNode *node,
                                                FModuleLike publicModule) {
  SmallVector<igraph::InstanceGraphNode *> worklist;
  worklist.push_back(node);
  auto &reachableSet = alwaysReachable[publicModule];

  while (!worklist.empty()) {
    auto *currentNode = worklist.pop_back_val();
    auto module =
        dyn_cast<FModuleLike>(currentNode->getModule().getOperation());
    if (!module)
      continue;

    for (auto *record : *currentNode) {
      auto *targetNode = record->getTarget();
      if (!targetNode)
        continue;

      if (!isa<InstanceOp>(record->getInstance().getOperation()))
        continue;

      auto targetModule =
          dyn_cast<FModuleLike>(targetNode->getModule().getOperation());
      if (targetModule && reachableSet.insert(targetModule).second)
        worklist.push_back(targetNode);
    }
  }
}

LogicalResult InstanceChoiceInfo::processNode(igraph::InstanceGraphNode *node,
                                              FModuleLike publicModule) {
  auto module = dyn_cast<FModuleLike>(node->getModule().getOperation());
  if (!module)
    return success();

  if (isAlwaysReachable(publicModule, module)) {
    moduleChoices[publicModule][module] = {};
    return success();
  }

  assert(module != publicModule && "public module should be pre-initialized");

  for (auto *use : node->uses()) {
    auto *instOp = use->getInstance().getOperation();
    auto *parentNode = use->getParent();
    if (!parentNode)
      continue;

    auto parentModule =
        dyn_cast<FModuleLike>(parentNode->getModule().getOperation());
    if (!parentModule)
      continue;

    auto parentIt = moduleChoices[publicModule].find(parentModule);
    // It means the parent module is not reachable (regardless of instance or
    // instance choice) from the public module.
    if (parentIt == moduleChoices[publicModule].end())
      continue;

    const auto &parentChoices = parentIt->second;

    if (auto choice = dyn_cast<InstanceChoiceOp>(instOp)) {
      SymbolRefAttr optionName =
          FlatSymbolRefAttr::get(choice.getOptionNameAttr());

      for (const auto &parentChoice : parentChoices) {
        if (parentChoice.option != optionName) {
          auto diag = choice.emitOpError()
                      << "nested instance choice with option '"
                      << optionName.getLeafReference().getValue()
                      << "' conflicts with option '"
                      << parentChoice.option.getLeafReference().getValue()
                      << "' already on the path from public module '"
                      << publicModule.getModuleName() << "'";
          diag.attachNote(publicModule.getLoc()) << "public module here";
          return failure();
        }
      }

      auto &choices = moduleChoices[publicModule][module];
      choices.insert(parentChoices.begin(), parentChoices.end());

      auto checkAndInsert = [&](SymbolRefAttr caseRef) -> LogicalResult {
        for (const auto &parentChoice : parentChoices) {
          if (parentChoice.option == optionName &&
              parentChoice.caseAttr != caseRef) {
            auto diag = choice.emitOpError()
                        << "nested instance choice with option '"
                        << optionName.getLeafReference().getValue() << "'";
            if (caseRef)
              diag << " and case '" << caseRef << "'";
            else
              diag << " and default case";
            diag << " conflicts with ";
            if (parentChoice.caseAttr)
              diag << "case '" << parentChoice.caseAttr << "'";
            else
              diag << "default case";
            diag << " already on the path from public module '"
                 << publicModule.getModuleName() << "'";
            diag.attachNote(publicModule.getLoc()) << "public module here";
            return failure();
          }
        }
        choices.insert({optionName, caseRef});
        return success();
      };

      if (choice.getDefaultTargetAttr().getAttr() == module.getModuleNameAttr())
        if (failed(checkAndInsert(SymbolRefAttr())))
          return failure();

      for (auto [caseRef, moduleRef] : choice.getTargetChoices())
        if (moduleRef.getAttr() == module.getModuleNameAttr())
          if (failed(checkAndInsert(caseRef)))
            return failure();

    } else if (isa<InstanceOp>(instOp)) {
      moduleChoices[publicModule][module].insert(parentChoices.begin(),
                                                 parentChoices.end());
    }
  }

  return success();
}

void InstanceChoiceInfo::dump(raw_ostream &os) const {
  for (auto [publicModule, destMap] : moduleChoices) {
    os << "Public module: " << publicModule.getModuleName() << "\n";

    for (auto [destModule, choices] : destMap) {
      // Only output modules that have choices
      if (choices.empty()) {
        os << "  -> " << destModule.getModuleName() << ": <always>\n";
        continue;
      }

      os << "  -> " << destModule.getModuleName() << ": ";
      llvm::interleaveComma(choices, os, [&](const ChoiceKey &key) {
        os << key.option.getLeafReference().getValue();
        if (key.caseAttr)
          os << "=" << key.caseAttr.getLeafReference().getValue();
        else
          os << "=<default>";
      });
      os << "\n";
    }
  }
}

namespace {
class CheckInstanceChoicePass
    : public circt::firrtl::impl::CheckInstanceChoiceBase<
          CheckInstanceChoicePass> {
public:
  using CheckInstanceChoiceBase::CheckInstanceChoiceBase;

  void runOnOperation() override {
    auto circuit = getOperation();
    auto &instanceGraph = getAnalysis<InstanceGraph>();
    InstanceChoiceInfo info(circuit, instanceGraph);
    if (failed(info.run()))
      return signalPassFailure();

    if (dumpInfo)
      info.dump(llvm::errs());

    markAllAnalysesPreserved();
  }
};
} // namespace
