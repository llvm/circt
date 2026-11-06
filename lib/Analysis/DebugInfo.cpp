//===- DebugInfo.cpp - Debug info analysis --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Analysis/DebugInfo.h"
#include "circt/Dialect/Debug/DebugOps.h"
#include "circt/Dialect/HW/HWOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "di"

using namespace mlir;
using namespace circt;

/// The direction of a port as debug info records it.
static debug::PortDirection getPortDirection(const hw::PortInfo &port) {
  if (isa<hw::InOutType>(port.type))
    return debug::PortDirection::InOut;
  return port.isOutput() ? debug::PortDirection::Out : debug::PortDirection::In;
}

namespace circt {
namespace detail {

/// What `visitModule` learns about one module before it builds the nodes.
struct ModuleScan {
  ModuleScan(hw::HWModuleOp moduleOp, DIModule &module)
      : moduleOp(moduleOp), module(module), views{&module} {}

  hw::HWModuleOp moduleOp;
  DIModule &module;
  /// The module and the modules dedup merged into it, all on `moduleOp`.
  SmallVector<DIModule *> views;
  /// (scope, null for the module; view name) to the ops in it.
  DenseMap<std::pair<Value, StringAttr>, SmallVector<Operation *>> scopeOps;
  /// The instances and wires, which every view of the module has.
  SmallVector<Operation *> sharedOps;
  /// The views that have an entry for the module in `scopeOps`.
  SmallVector<StringAttr> rootViews;
  /// (parent scope, instance name) to the inlined scope of that instance.
  DenseMap<std::pair<Value, StringAttr>, Value> siblings;
  bool hasVariables = false;
  /// Each inlined scope of the view to its node.
  DenseMap<Value, DIModule *> scopeNodes;
  SmallVector<DIInstance *> instances;
};

/// Helper to populate a `DebugInfo` with nodes.
struct DebugInfoBuilder {
  DebugInfoBuilder(DebugInfo &di) : di(di) {}
  DebugInfo &di;

  void visitRoot(Operation *op);
  void visitModule(hw::HWModuleOp moduleOp, DIModule &module);

  void addView(ModuleScan &scan, StringAttr name);
  StringAttr getOrigin(ModuleScan &scan, Operation *op);
  SmallVector<std::pair<StringAttr, debug::InstanceAttr>>
  getScopeEntries(ModuleScan &scan, debug::ScopeOp scopeOp);
  void groupOps(ModuleScan &scan);
  ArrayRef<Operation *> getOps(ModuleScan &scan, Value scope, StringAttr name);
  /// Collect the ops of `scope` into `node`.
  void collectOps(ModuleScan &scan, DIModule &node, Value scope);
  void addPortVariables(ModuleScan &scan, DIModule &view);
  void describeInstances(ModuleScan &scan, DIModule &view);

  /// Instances whose module, as their description names it, may be one that
  /// dedup merged into their target. Resolved once all modules are known.
  SmallVector<std::pair<DIInstance *, StringAttr>> instanceOrigins;

  DIModule *createModule() {
    return new (di.moduleAllocator.Allocate()) DIModule;
  }

  DIInstance *createInstance() {
    return new (di.instanceAllocator.Allocate()) DIInstance;
  }

  DIVariable *createVariable() {
    return new (di.variableAllocator.Allocate()) DIVariable;
  }

  DIModule &getOrCreateModule(StringAttr moduleName) {
    auto &slot = di.moduleNodes[moduleName];
    if (!slot) {
      slot = createModule();
      slot->name = moduleName;
    }
    return *slot;
  }
};

void DebugInfoBuilder::visitRoot(Operation *op) {
  op->walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (auto moduleOp = dyn_cast<hw::HWModuleOp>(op)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Collect DI for module " << moduleOp.getNameAttr() << "\n");
      auto &module = getOrCreateModule(moduleOp.getNameAttr());
      module.op = op;
      module.isArtificial = op->hasAttr(debug::kDbgArtificialAttr);
      module.isLowered = op->hasAttr(debug::kDbgLoweredAttr);
      auto info = op->getAttrOfType<DictionaryAttr>(debug::kDbgModuleInfoAttr);
      if (info) {
        module.typeName = info.getAs<StringAttr>("name");
        module.ext = info.getAs<DictionaryAttr>("ext");
      }
      visitModule(moduleOp, module);
      return WalkResult::skip();
    }

    if (auto moduleOp = dyn_cast<hw::HWModuleExternOp>(op)) {
      LLVM_DEBUG(llvm::dbgs() << "Collect DI for extern module "
                              << moduleOp.getNameAttr() << "\n");
      auto &module = getOrCreateModule(moduleOp.getNameAttr());
      module.op = op;
      module.isExtern = true;
      module.isLowered = op->hasAttr(debug::kDbgLoweredAttr);

      // Add variables for each of the ports.
      for (auto &port : moduleOp.getPortList()) {
        auto *var = createVariable();
        var->name = port.name;
        var->loc = port.loc;
        var->dir = getPortDirection(port);
        module.variables.push_back(var);
      }

      return WalkResult::skip();
    }

    return WalkResult::advance();
  });

  // Take the module an instance description names if it shares the op of the
  // target, i.e. dedup merged it into the target. Otherwise the name is stale.
  for (auto [instance, origin] : instanceOrigins) {
    auto *module = di.moduleNodes.lookup(origin);
    if (!module || !module->op)
      continue;

    if (module->op == instance->module->op)
      instance->module = module;
  }
}

/// The inlined scope that `op` is in, if any.
static debug::ScopeOp getInlinedScope(Operation *op) {
  Value scope;
  if (auto varOp = dyn_cast<debug::VariableOp>(op))
    scope = varOp.getScope();
  else if (auto scopeOp = dyn_cast<debug::ScopeOp>(op))
    scope = scopeOp.getScope();
  if (!scope)
    return {};

  return scope.getDefiningOp<debug::ScopeOp>();
}

void DebugInfoBuilder::addView(ModuleScan &scan, StringAttr name) {
  if (!name || name == scan.module.name)
    return;

  auto &view = getOrCreateModule(name);
  if (view.op)
    return;

  view.op = scan.moduleOp;
  view.isArtificial = scan.module.isArtificial;
  scan.views.push_back(&view);
}

StringAttr DebugInfoBuilder::getOrigin(ModuleScan &scan, Operation *op) {
  if (auto origin = op->getAttrOfType<StringAttr>(debug::kDbgMergedFromAttr))
    return origin;

  if (auto scopeOp = getInlinedScope(op))
    return scopeOp.getModuleNameAttr();
  return scan.module.name;
}

/// The views that see `scopeOp`, each with its description of the inlined
/// instance. After dedup the scope has a `dbg.instance` list, one entry per
/// merged module; an entry without an `origin` is of the module the scope
/// is in. Otherwise the scope describes the instance itself.
SmallVector<std::pair<StringAttr, debug::InstanceAttr>>
DebugInfoBuilder::getScopeEntries(ModuleScan &scan, debug::ScopeOp scopeOp) {
  auto origin = getOrigin(scan, scopeOp);
  auto entries = scopeOp->getAttrOfType<ArrayAttr>(debug::kDbgInstanceAttr);
  if (!entries)
    return {{origin, {}}};

  SmallVector<std::pair<StringAttr, debug::InstanceAttr>> result;
  for (auto entry : entries.getAsRange<debug::InstanceAttr>())
    result.push_back({entry.getOrigin() ? entry.getOrigin() : origin, entry});
  return result;
}

void DebugInfoBuilder::groupOps(ModuleScan &scan) {
  scan.moduleOp.walk([&](Operation *op) {
    if (isa<debug::VariableOp>(op))
      scan.hasVariables = true;
    else if (!isa<debug::ScopeOp, hw::InstanceOp, hw::WireOp>(op))
      return;

    Value scope;
    if (auto scopeOp = getInlinedScope(op))
      scope = scopeOp;
    if (!scope) {
      auto mergedFrom =
          op->getAttrOfType<StringAttr>(debug::kDbgMergedFromAttr);
      addView(scan, mergedFrom);
    }

    auto addOp = [&](StringAttr origin) {
      auto [it, inserted] = scan.scopeOps.try_emplace({scope, origin});
      if (inserted && !scope) {
        it->second = scan.sharedOps;
        scan.rootViews.push_back(it->first.second);
      }
      it->second.push_back(op);
    };

    // A scope is in each view it has an entry for.
    if (auto scopeOp = dyn_cast<debug::ScopeOp>(op)) {
      scan.siblings.try_emplace({scope, scopeOp.getInstanceNameAttr()},
                                scopeOp);
      for (auto [origin, entry] : getScopeEntries(scan, scopeOp)) {
        if (!scope)
          addView(scan, origin);
        addOp(origin);
      }
      return;
    }

    // Every view has the instances and wires: a view seen later starts
    // with the ones met so far, and the views seen so far get this one.
    if (isa<hw::InstanceOp, hw::WireOp>(op)) {
      scan.sharedOps.push_back(op);
      for (auto name : scan.rootViews)
        scan.scopeOps[{scope, name}].push_back(op);
      return;
    }

    addOp(getOrigin(scan, op));
  });
}

ArrayRef<Operation *> DebugInfoBuilder::getOps(ModuleScan &scan, Value scope,
                                               StringAttr name) {
  if (auto it = scan.scopeOps.find({scope, name}); it != scan.scopeOps.end())
    return it->second;
  return scope ? ArrayRef<Operation *>()
               : ArrayRef<Operation *>(scan.sharedOps);
}

// NOLINTNEXTLINE(misc-no-recursion)
void DebugInfoBuilder::collectOps(ModuleScan &scan, DIModule &node,
                                  Value scope) {
  for (auto *op : getOps(scan, scope, node.name)) {
    if (auto varOp = dyn_cast<debug::VariableOp>(op)) {
      auto *var = createVariable();
      var->name = varOp.getNameAttr();
      var->loc = varOp.getLoc();
      var->value = varOp.getValue();
      var->kind = varOp.getKind();
      var->dir = varOp.getDir();
      node.variables.push_back(var);
      continue;
    }

    if (auto scopeOp = dyn_cast<debug::ScopeOp>(op)) {
      auto *child = createModule();
      child->isInline = true;
      child->name = scopeOp.getModuleNameAttr();
      child->op = scopeOp;
      auto *instance = createInstance();
      instance->name = scopeOp.getInstanceNameAttr();
      instance->op = scopeOp;
      instance->module = child;
      instance->typeName = scopeOp.getTypeNameAttr();
      instance->ext = scopeOp.getExtAttr();
      for (auto [origin, entry] : getScopeEntries(scan, scopeOp)) {
        if (!entry || origin != node.name)
          continue;
        if (auto module = entry.getModule())
          child->name = module;
        instance->name = entry.getName();
        instance->typeName = entry.getTypeName();
        instance->ext = entry.getExt();
        break;
      }
      node.instances.push_back(instance);

      scan.scopeNodes[scopeOp] = child;
      collectOps(scan, *child, scopeOp);
      continue;
    }

    // Fallback if the module has no DI for its instances.
    if (auto instOp = dyn_cast<hw::InstanceOp>(op)) {
      StringAttr childName = instOp.getModuleNameAttr().getAttr();
      auto &childModule = getOrCreateModule(childName);
      auto *instance = createInstance();
      instance->name = instOp.getInstanceNameAttr();
      instance->op = instOp;
      instance->module = &childModule;
      node.instances.push_back(instance);
      scan.instances.push_back(instance);

      // TODO: What do we do with the port assignments? These should be
      // tracked somewhere.
      continue;
    }

    // Fallback if the module has no DI for its variables.
    if (auto wireOp = dyn_cast<hw::WireOp>(op); wireOp && !scan.hasVariables) {
      auto *var = createVariable();
      var->name = wireOp.getNameAttr();
      var->loc = wireOp.getLoc();
      var->value = wireOp;
      node.variables.push_back(var);
    }
  }
}

void DebugInfoBuilder::addPortVariables(ModuleScan &scan, DIModule &view) {
  auto inputValues = scan.moduleOp.getBody().getArguments();
  auto outputValues =
      scan.moduleOp.getBodyBlock()->getTerminator()->getOperands();
  for (auto &port : scan.moduleOp.getPortList()) {
    auto value =
        port.isOutput() ? outputValues[port.argNum] : inputValues[port.argNum];
    auto *var = createVariable();
    var->name = port.name;
    var->loc = port.loc;
    var->value = value;
    var->dir = getPortDirection(port);
    view.variables.push_back(var);
  }
}

void DebugInfoBuilder::describeInstances(ModuleScan &scan, DIModule &view) {
  // The origin of an entry in this view: none for the module itself, or the
  // module dedup merged into it. An entry with a path names the module it
  // was in, which is the module of the inlined scope.
  auto viewOrigin = view.name == scan.module.name ? StringAttr{} : view.name;
  SmallPtrSet<DIInstance *, 8> moved;
  for (auto *instance : scan.instances) {
    auto entries =
        instance->op->getAttrOfType<ArrayAttr>(debug::kDbgInstanceAttr);
    if (!entries)
      continue;

    // Find the inlined scope the instance sits in by the source names of
    // the inlined instances on the way.
    auto getNode = [&](debug::InstanceAttr entry) -> DIModule * {
      auto path = entry.getPath();
      if (!path)
        return &view;
      Value scope;
      for (auto name : path.getAsRange<StringAttr>())
        if (!(scope = scan.siblings.lookup({scope, name})))
          return nullptr;
      return scan.scopeNodes.lookup(scope);
    };
    DIModule *node = nullptr;
    debug::InstanceAttr entry;
    for (auto candidate : entries.getAsRange<debug::InstanceAttr>()) {
      node = getNode(candidate);
      if (node &&
          candidate.getOrigin() == (node == &view ? viewOrigin : node->name)) {
        entry = candidate;
        break;
      }
    }
    if (!entry)
      continue;

    if (node != &view) {
      moved.insert(instance);
      node->instances.push_back(instance);
    }

    instance->name = entry.getName();
    if (auto module = entry.getModule())
      instanceOrigins.push_back({instance, module});
    instance->typeName = entry.getTypeName();
    instance->ext = entry.getExt();
  }

  llvm::erase_if(view.instances, [&](DIInstance *instance) {
    return moved.contains(instance);
  });
}

void DebugInfoBuilder::visitModule(hw::HWModuleOp moduleOp, DIModule &module) {
  // Try to gather debug info from debug ops in the module. If we find any,
  // return. Otherwise collect ports, instances, and variables as a
  // fallback.
  ModuleScan scan(moduleOp, module);

  // Dedup may have merged other modules into this one.
  // Each of them gets its own `DIModule` on the same op, a view.
  // A root debug op belongs to the module named in its `dbg.merged_from`,
  // else to this one; an op in an inlined scope to the module of that scope.
  // Instances and wires are hardware that all views share.
  auto origins =
      moduleOp->getAttrOfType<DictionaryAttr>(debug::kDbgModuleLocsAttr);
  if (origins)
    for (auto origin : origins)
      addView(scan, origin.getName());

  // Group the ops by the scope they are in, null for the module,
  // and by the view they belong to.
  // After dedup an inlined instance has one scope for all views.
  // Instances and wires are in the module and go into the group of each view.
  groupOps(scan);

  for (auto *view : scan.views) {
    // If the module has no DI for variables,
    // add variables for each of the ports as a fallback.
    if (!scan.hasVariables)
      addPortVariables(scan, *view);

    scan.scopeNodes.clear();
    scan.instances.clear();
    collectOps(scan, *view, {});

    // An instance takes the source name, module and class from its
    // description for this view, and sits in the inlined module its path
    // names.
    describeInstances(scan, *view);
  }
}

} // namespace detail
} // namespace circt

DebugInfo::DebugInfo(Operation *op) : operation(op) {
  detail::DebugInfoBuilder(*this).visitRoot(op);
}
