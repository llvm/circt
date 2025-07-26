//===- HWInlineInputOnlyModules.cpp - Inline input-only HW modules --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transform inlines modules which do not have output ports.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Emit/EmitOps.h"
#include "circt/Dialect/HW/HWAttributes.h"
#include "circt/Dialect/HW/HWInstanceGraph.h"
#include "circt/Dialect/HW/HWOps.h"
#include "circt/Dialect/HW/HWPasses.h"
#include "circt/Dialect/HW/HWSymCache.h"
#include "circt/Dialect/HW/InnerSymbolNamespace.h"
#include "circt/Dialect/SV/SVOps.h"
#include "circt/Dialect/SV/SVPasses.h"
#include "circt/Dialect/Seq/SeqDialect.h"
#include "circt/Dialect/Seq/SeqOps.h"
#include "circt/Dialect/Verif/VerifOps.h"
#include "circt/Support/Namespace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SetVector.h"

namespace circt {
namespace hw {
#define GEN_PASS_DEF_INLINEINPUTONLYMODULES
#include "circt/Dialect/HW/Passes.h.inc"
} // namespace hw
} // namespace circt

using namespace mlir;
using namespace circt;
using namespace sv;
using namespace hw;

using BindTable = DenseMap<StringAttr, SmallDenseMap<StringAttr, sv::BindOp>>;

// Check if the module has already been bound.
static bool isBound(hw::HWModuleLike op, hw::InstanceGraph &instanceGraph) {
  auto *node = instanceGraph.lookup(op);
  return llvm::any_of(node->uses(), [](igraph::InstanceRecord *a) {
    auto inst = a->getInstance<hw::HWInstanceLike>();
    if (!inst)
      return false;

    return inst.getDoNotPrint();
  });
}

// Add any existing bindings to the bind table.
static void addExistingBinds(Block *topLevelModule, BindTable &bindTable) {
  for (auto bind : topLevelModule->getOps<BindOp>()) {
    hw::InnerRefAttr boundRef = bind.getInstance();
    bindTable[boundRef.getModule()][boundRef.getName()] = bind;
  }
}

// Check if op has any operand using a value that isn't yet defined.
static bool hasOOOArgs(hw::HWModuleOp newMod, Operation *op) {
  for (auto arg : op->getOperands()) {
    auto *argOp = arg.getDefiningOp(); // may be null
    if (!argOp)
      continue;
    if (argOp->getParentOfType<hw::HWModuleOp>() != newMod)
      return true;
  }
  return false;
}

// Update any operand which was emitted before its defining op was.
static void updateOOOArgs(SmallVectorImpl<Operation *> &lateBoundOps,
                          IRMapping &cutMap) {
  for (auto *op : lateBoundOps)
    for (unsigned argidx = 0, e = op->getNumOperands(); argidx < e; ++argidx) {
      Value arg = op->getOperand(argidx);
      if (cutMap.contains(arg))
        op->setOperand(argidx, cutMap.lookup(arg));
    }
}

// Inline any modules that only have inputs for test code.
static void
inlineInputOnly(hw::HWModuleOp oldMod, hw::InstanceGraph &instanceGraph,
                BindTable &bindTable,
                llvm::DenseSet<hw::InnerRefAttr> &innerRefUsedByNonBindOp) {
  // Check if the module only has inputs.
  if (oldMod.getNumOutputPorts() != 0)
    return;


  // Check if it's ok to inline. We cannot inline the module if there exists a
  // declaration with an inner symbol referred by non-bind ops (e.g. hierpath).
  auto oldModName = oldMod.getModuleNameAttr();
  for (auto port : oldMod.getPortList()) {
    auto sym = port.getSym();
    if (sym) {
      for (auto property : sym) {
        auto innerRef = hw::InnerRefAttr::get(oldModName, property.getName());
        if (innerRefUsedByNonBindOp.count(innerRef)) {
          oldMod.emitWarning() << "module " << oldMod.getModuleName()
                               << " is an input only module but cannot "
                                  "be inlined because a signal "
                               << port.name << " is referred by name";
          return;
        }
      }
    }
  }

  for (auto op : oldMod.getBodyBlock()->getOps<hw::InnerSymbolOpInterface>()) {
    if (auto innerSym = op.getInnerSymAttr()) {
      for (auto property : innerSym) {
        auto innerRef = hw::InnerRefAttr::get(oldModName, property.getName());
        if (innerRefUsedByNonBindOp.count(innerRef)) {
          op.emitWarning() << "module " << oldMod.getModuleName()
                           << " is an input only module but cannot be inlined "
                              "because signals are referred by name";
          return;
        }
      }
    }
  }

  SmallPtrSet<Operation *, 32> opsToErase;

  // Get the instance graph node for the old module.
  igraph::InstanceGraphNode *node = instanceGraph.lookup(oldMod);

  // Iterate through each instance of the module.
  OpBuilder b(oldMod);
  bool allInlined = true;
  for (igraph::InstanceRecord *use : llvm::make_early_inc_range(node->uses())) {
    // If there is no instance, move on.
    auto instLike = use->getInstance<hw::HWInstanceLike>();
    if (!instLike) {
      allInlined = false;
      continue;
    }

    // If the instance had a symbol, we can't inline it without more work.
    hw::InstanceOp inst = cast<hw::InstanceOp>(instLike.getOperation());
    if (inst.getInnerSym().has_value()) {
      allInlined = false;
      auto diag =
          oldMod.emitWarning()
          << "module " << oldMod.getModuleName()
          << " cannot be inlined because there is an instance with a symbol";
      diag.attachNote(inst.getLoc());
      continue;
    }

    // Build a mapping from module block arguments to instance inputs.
    IRMapping mapping;
    assert(inst.getInputs().size() == oldMod.getNumInputPorts());
    auto inputPorts = oldMod.getBodyBlock()->getArguments();
    for (size_t i = 0, e = inputPorts.size(); i < e; ++i)
      mapping.map(inputPorts[i], inst.getOperand(i));

    // Inline the body at the instantiation site.
    hw::HWModuleOp instParent =
        cast<hw::HWModuleOp>(use->getParent()->getModule());
    igraph::InstanceGraphNode *instParentNode =
        instanceGraph.lookup(instParent);
    SmallVector<Operation *, 16> lateBoundOps;
    b.setInsertionPoint(inst);
    // Namespace that tracks inner symbols in the parent module.
    hw::InnerSymbolNamespace nameSpace(instParent);
    // A map from old inner symbols to new ones.
    DenseMap<mlir::StringAttr, mlir::StringAttr> symMapping;

    for (auto &op : *oldMod.getBodyBlock()) {
      // If the op was erased by instance extraction, don't copy it over.
      if (opsToErase.contains(&op))
        continue;

      // If the op has an inner sym, first create a new inner sym for it.
      if (auto innerSymOp = dyn_cast<hw::InnerSymbolOpInterface>(op)) {
        if (auto innerSym = innerSymOp.getInnerSymAttr()) {
          for (auto property : innerSym) {
            auto oldName = property.getName();
            auto newName =
                b.getStringAttr(nameSpace.newName(oldName.getValue()));
            auto result = symMapping.insert({oldName, newName});
            (void)result;
            assert(result.second && "inner symbols must be unique");
          }
        }
      }

      // For instances in the bind table, update the bind with the new parent.
      if (auto innerInst = dyn_cast<hw::InstanceOp>(op)) {
        if (auto innerInstSym = innerInst.getInnerSymAttr()) {
          auto it =
              bindTable[oldMod.getNameAttr()].find(innerInstSym.getSymName());
          if (it != bindTable[oldMod.getNameAttr()].end()) {
            sv::BindOp bind = it->second;
            auto oldInnerRef = bind.getInstanceAttr();
            auto it = symMapping.find(oldInnerRef.getName());
            assert(it != symMapping.end() &&
                   "inner sym mapping must be already populated");
            auto newName = it->second;
            auto newInnerRef =
                hw::InnerRefAttr::get(instParent.getModuleNameAttr(), newName);
            OpBuilder::InsertionGuard g(b);
            // Clone bind operations.
            b.setInsertionPoint(bind);
            sv::BindOp clonedBind = cast<sv::BindOp>(b.clone(*bind, mapping));
            clonedBind.setInstanceAttr(newInnerRef);
            bindTable[instParent.getModuleNameAttr()][newName] =
                cast<sv::BindOp>(clonedBind);
          }
        }
      }

      // For all ops besides the output, clone into the parent body.
      if (!isa<hw::OutputOp>(op)) {
        Operation *clonedOp = b.clone(op, mapping);
        // If some of the operands haven't been cloned over yet, due to cycles,
        // remember to revisit this op.
        if (hasOOOArgs(instParent, clonedOp))
          lateBoundOps.push_back(clonedOp);

        // If the cloned op is an instance, record it within the new parent in
        // the instance graph.
        if (auto innerInst = dyn_cast<hw::InstanceOp>(clonedOp)) {
          igraph::InstanceGraphNode *innerInstModule =
              instanceGraph.lookup(innerInst.getModuleNameAttr().getAttr());
          instParentNode->addInstance(innerInst, innerInstModule);
        }

        // If the cloned op has an inner sym, then attach an updated inner sym.
        if (auto innerSymOp = dyn_cast<hw::InnerSymbolOpInterface>(clonedOp)) {
          if (auto oldInnerSym = innerSymOp.getInnerSymAttr()) {
            SmallVector<hw::InnerSymPropertiesAttr> properties;
            for (auto property : oldInnerSym) {
              auto newSymName = symMapping[property.getName()];
              properties.push_back(hw::InnerSymPropertiesAttr::get(
                  op.getContext(), newSymName, property.getFieldID(),
                  property.getSymVisibility()));
            }
            auto innerSym = hw::InnerSymAttr::get(op.getContext(), properties);
            innerSymOp.setInnerSymbolAttr(innerSym);
          }
        }
      }
    }

    // Map over any ops that didn't have their operands mapped when cloned.
    updateOOOArgs(lateBoundOps, mapping);

    // Erase the old instantiation site.
    assert(inst.use_empty() && "inlined instance should have no uses");
    use->erase();
    opsToErase.insert(inst);
  }

  // If all instances were inlined, remove the module.
  if (allInlined) {
    // Erase old bind statements.
    for (auto [_, bind] : bindTable[oldMod.getNameAttr()])
      bind.erase();
    bindTable[oldMod.getNameAttr()].clear();
    instanceGraph.erase(node);
    opsToErase.insert(oldMod);
  }

  while (!opsToErase.empty()) {
    Operation *op = *opsToErase.begin();
    op->walk([&](Operation *erasedOp) { opsToErase.erase(erasedOp); });
    op->dropAllUses();
    op->erase();
  }
}

namespace {
struct InlineInputOnlyModulesPass
    : public circt::hw::impl::InlineInputOnlyModulesBase<
          InlineInputOnlyModulesPass> {
  void runOnOperation() override;
};
} // namespace

void InlineInputOnlyModulesPass::runOnOperation() {
  auto &instanceGraph = getAnalysis<circt::hw::InstanceGraph>();
  auto top = getOperation();
  BindTable bindTable;
  addExistingBinds(top.getBody(), bindTable);

  // It takes extra effort to inline modules which contains inner symbols
  // referred through hierpaths or unknown operations since we have to update
  // inner refs users globally. However we do want to inline modules which
  // contain bound instances so create a set of inner refs used by non bind op
  // in order to allow bind ops.
  DenseSet<hw::InnerRefAttr> innerRefUsedByNonBindOp;
  top.walk([&](Operation *op) {
    if (!isa<sv::BindOp>(op))
      for (auto attr : op->getAttrs())
        attr.getValue().walk([&](hw::InnerRefAttr attr) {
          innerRefUsedByNonBindOp.insert(attr);
        });
  });

  for (auto module :
       llvm::make_early_inc_range(top.getBody()->getOps<HWModuleOp>())) {
    if (isBound(module, instanceGraph))
      continue;

    inlineInputOnly(module, instanceGraph, bindTable, innerRefUsedByNonBindOp);
  }

  markAnalysesPreserved<circt::hw::InstanceGraph>();
}
