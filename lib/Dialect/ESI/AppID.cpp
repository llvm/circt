//===- AppID.cpp - AppID related code -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/ESI/AppID.h"
#include "circt/Dialect/ESI/ESIOps.h"

#include "circt/Support/InstanceGraph.h"

using namespace circt;
using namespace esi;

AppIDAttr circt::esi::getAppID(Operation *op) {
  if (auto appidOp = dyn_cast<esi::HasAppID>(op))
    return appidOp.getAppID();
  if (auto appid = op->getAttrOfType<AppIDAttr>(AppIDAttr::AppIDAttrName))
    return appid;
  return AppIDAttr();
}

/// Helper class constructed on a per-HWModuleLike basis. Contains a map for
/// fast lookups to the operation involved in an appid component.
class AppIDIndex::ModuleAppIDs {
public:
  /// Add an appid component to the index. 'Inherited' is true if we're
  /// bubbling up from an instance and is used to inform the conflicting entry
  /// error message.
  LogicalResult add(AppIDAttr id, Operation *op, bool inherited) {
    if (childAppIDPaths.find(id) != childAppIDPaths.end()) {
      return op->emitOpError("Found multiple identical AppIDs in same module")
                 .attachNote(childAppIDPaths[id]->getLoc())
             << "first AppID located here."
             << (inherited ? " Must insert appid to differentiate one instance "
                             "branch from the other."
                           : "");
    }
    childAppIDPaths[id] = op;
    childAppIDPathsOrdered.emplace_back(id, op);
    return success();
  }

  FailureOr<Operation *> lookup(AppIDAttr id, Location loc) const {
    auto f = childAppIDPaths.find(id);
    if (f == childAppIDPaths.end())
      return emitError(loc, "could not find appid '") << id << "'";
    return f->second;
  }

  // Returns a range iterator to the AppID components exposed by this module.
  auto getAppIDs() const {
    return llvm::make_first_range(childAppIDPathsOrdered);
  }

  // Get a read-only reference to the index.
  ArrayRef<std::pair<AppIDAttr, Operation *>> getChildren() const {
    return childAppIDPathsOrdered;
  }

private:
  // Operations involved in appids.
  DenseMap<AppIDAttr, Operation *> childAppIDPaths;
  // For every entry in childAppIDPaths, we need it in the original order. Keep
  // that order here.
  SmallVector<std::pair<AppIDAttr, Operation *>, 8> childAppIDPathsOrdered;
};

AppIDIndex::AppIDIndex(Operation *mlirTop) : valid(true), mlirTop(mlirTop) {
  Block &topBlock = mlirTop->getRegion(0).front();
  symCache.addDefinitions(mlirTop);
  symCache.freeze();

  // Build the per-module cache.
  for (auto mod : topBlock.getOps<hw::HWModuleLike>())
    if (failed(buildIndexFor(mod))) {
      valid = false;
      break;
    }
}
AppIDIndex::~AppIDIndex() {
  for (auto [appId, childAppIDs] : containerAppIDs)
    delete childAppIDs;
}

ArrayAttr AppIDIndex::getChildAppIDsOf(hw::HWModuleLike fromMod) const {
  auto f = containerAppIDs.find(fromMod);
  if (f == containerAppIDs.end())
    return ArrayAttr::get(fromMod.getContext(), {});

  const ModuleAppIDs *fromModIdx = f->getSecond();
  SmallVector<Attribute, 8> attrs(llvm::map_range(
      fromModIdx->getAppIDs(), [](AppIDAttr a) -> Attribute { return a; }));
  return ArrayAttr::get(fromMod.getContext(), attrs);
}

// NOLINTNEXTLINE(misc-no-recursion)
LogicalResult AppIDIndex::walk(
    hw::HWModuleLike top, hw::HWModuleLike current,
    SmallVectorImpl<AppIDAttr> &pathStack,
    SmallVectorImpl<Operation *> &opStack,
    function_ref<void(AppIDPathAttr, ArrayRef<Operation *>)> fn) const {
  ModuleAppIDs *modIDs = containerAppIDs.lookup(current);
  if (!modIDs)
    return success();

  for (auto [appid, op] : modIDs->getChildren()) {
    // If we encounter an instance op which isn't ID'd, iterate down the
    // instance hierarchy until we find it.
    AppIDAttr opAppID = getAppID(op);
    while (!opAppID) {
      // We make a bunch of assumptions based on correct construction of the
      // index here. Assert on a bunch of things which would ordinarily be
      // failures, but we can assume never to happen based on the index
      // construction.

      auto inst = dyn_cast<hw::HWInstanceLike>(op);
      assert(inst && "Search bottomed out. Invalid appid index.");

      auto moduleNames = inst.getReferencedModuleNamesAttr();
      if (moduleNames.size() != 1)
        return inst.emitError("expected an instance with a single reference");

      auto tgtMod =
          dyn_cast<hw::HWModuleLike>(symCache.getDefinition(moduleNames[0]));
      assert(tgtMod && "invalid module reference");

      ModuleAppIDs *ffModIds = containerAppIDs.at(tgtMod);
      assert(ffModIds && "could not find module in index.");

      auto opF = ffModIds->lookup(appid, op->getLoc());
      assert(succeeded(opF) &&
             "could not find appid in module index. Invalid index.");

      // Set the iteration variables for the next iteration.
      op = *opF;
      opAppID = getAppID(op);
    }

    // Push the appid and op onto the shared stacks.
    opStack.push_back(op);
    pathStack.push_back(appid);

    // Call the callback.
    AppIDPathAttr path = AppIDPathAttr::get(
        current.getContext(), FlatSymbolRefAttr::get(top.getNameAttr()),
        pathStack);
    fn(path, opStack);

    // We must recurse on an instance.
    if (auto inst = dyn_cast<hw::HWInstanceLike>(op)) {
      auto moduleNames = inst.getReferencedModuleNamesAttr();
      if (moduleNames.size() != 1)
        return inst.emitError("expected an instance with a single reference");

      auto tgtMod =
          dyn_cast<hw::HWModuleLike>(symCache.getDefinition(moduleNames[0]));
      assert(tgtMod && "invalid module reference");

      if (failed(walk(top, tgtMod, pathStack, opStack, fn)))
        return failure();
    }

    // Since the stacks are shared (for efficiency reasons), pop them.
    pathStack.pop_back();
    opStack.pop_back();
  }
  return success();
}

LogicalResult AppIDIndex::walk(
    hw::HWModuleLike top,
    function_ref<void(AppIDPathAttr, ArrayRef<Operation *>)> fn) const {
  SmallVector<AppIDAttr, 8> path;
  SmallVector<Operation *, 8> opStack;
  return walk(top, top, path, opStack, fn);
}
LogicalResult AppIDIndex::walk(
    StringRef top,
    function_ref<void(AppIDPathAttr, ArrayRef<Operation *>)> fn) const {
  Operation *op = symCache.getDefinition(
      FlatSymbolRefAttr::get(mlirTop->getContext(), top));
  if (auto topMod = dyn_cast_or_null<hw::HWModuleLike>(op))
    return walk(topMod, fn);
  return mlirTop->emitOpError("Could not find module '") << top << "'";
}

FailureOr<ArrayAttr> AppIDIndex::getAppIDPathAttr(hw::HWModuleLike fromMod,
                                                  AppIDAttr appid,
                                                  Location loc) const {
  SmallVector<Attribute, 8> path;
  do {
    auto f = containerAppIDs.find(fromMod);
    if (f == containerAppIDs.end())
      return emitError(loc, "Could not find appid index for module '")
             << fromMod.getName() << "'";

    const ModuleAppIDs *modIDs = f->getSecond();
    FailureOr<hw::InnerSymbolOpInterface> op = modIDs->lookup(appid, loc);
    if (failed(op))
      return failure();
    op->dump();
    path.push_back(op->getInnerRef());

    if (getAppID(*op))
      break;

    if (auto inst = dyn_cast<hw::HWInstanceLike>(op->getOperation())) {
      auto moduleNames = inst.getReferencedModuleNamesAttr();
      if (moduleNames.size() != 1)
        return inst.emitError("expected an instance with a single reference");
      fromMod = cast<hw::HWModuleLike>(symCache.getDefinition(moduleNames[0]));
    } else {
      assert(false && "Search bottomed out");
    }
  } while (true);
  return ArrayAttr::get(fromMod.getContext(), path);
}

/// Do a DFS of the instance hierarchy, 'bubbling up' appids.
FailureOr<const AppIDIndex::ModuleAppIDs *>
AppIDIndex::buildIndexFor(hw::HWModuleLike mod) {
  // Memoize.
  ModuleAppIDs *&appIDs = containerAppIDs[mod];
  if (appIDs != nullptr)
    return appIDs;
  appIDs = new ModuleAppIDs();

  auto done = mod.walk([&](Operation *op) {
    // If an op has an appid attribute, add it to the index and terminate the
    // DFS (since AppIDs only get 'bubbled up' until they encounter an ID'd
    // instantiation).
    if (AppIDAttr appid = getAppID(op)) {
      if (failed(appIDs->add(appid, op, false)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    // If we encounter an instance op which isn't ID'd...
    if (auto inst = dyn_cast<hw::HWInstanceLike>(op)) {
      auto moduleNames = inst.getReferencedModuleNamesAttr();
      if (moduleNames.size() != 1) {
        inst.emitError("expected an instance with a single reference");
        return WalkResult::interrupt();
      }
      auto tgtMod =
          dyn_cast<hw::HWModuleLike>(symCache.getDefinition(moduleNames[0]));
      // Do the assert here to get a more precise message.
      assert(tgtMod && "invalid module reference");

      // Recurse.
      FailureOr<const ModuleAppIDs *> childIds = buildIndexFor(tgtMod);
      if (failed(childIds))
        return WalkResult::interrupt();

      // Then add the 'bubbled up' appids to the cache.
      for (AppIDAttr appid : (*childIds)->getAppIDs())
        if (failed(appIDs->add(appid, op, true)))
          return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (done.wasInterrupted())
    return failure();
  return appIDs;
}
