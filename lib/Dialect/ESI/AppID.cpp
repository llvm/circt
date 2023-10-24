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
  LogicalResult add(AppIDAttr id, hw::InnerSymbolOpInterface op,
                    bool inherited) {
    if (!op.getInnerName())
      return op->emitOpError(
          "to carry an appid, this op must have an inner symbol");

    if (childAppIDPaths.find(id) != childAppIDPaths.end()) {
      return op->emitOpError("Found multiple identical AppIDs in same module")
                 .attachNote(childAppIDPaths[id]->getLoc())
             << "first AppID located here."
             << (inherited ? " Must insert appid to differentiate one instance "
                             "branch from the other."
                           : "");
    }
    childAppIDPaths[id] = op;
    return success();
  }

  FailureOr<hw::InnerSymbolOpInterface> lookup(AppIDAttr id,
                                               Location loc) const {
    auto f = childAppIDPaths.find(id);
    if (f == childAppIDPaths.end())
      return emitError(loc, "could not find appid '") << id << "'";
    return f->second;
  }

  // Returns a range iterator to the AppID components exposed by this module.
  auto getAppIDs() const { return llvm::make_first_range(childAppIDPaths); }

  // Get a read-only reference to the index.
  const auto &getIndex() const { return childAppIDPaths; }

private:
  // Operations involved in appids.
  DenseMap<AppIDAttr, hw::InnerSymbolOpInterface> childAppIDPaths;
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
  SmallVector<AppIDAttr, 8> appids;
  for (AppIDAttr childID : fromModIdx->getAppIDs())
    appids.push_back(childID);
  llvm::sort(appids, [](const AppIDAttr a, const AppIDAttr b) {
    if (a.getName() == b.getName())
      return a.getIndex() < b.getIndex();
    return a.getName().compare(b.getName()) > 0;
  });

  SmallVector<Attribute, 8> attrs(
      llvm::map_range(appids, [](AppIDAttr a) -> Attribute { return a; }));
  return ArrayAttr::get(fromMod.getContext(), attrs);
}

/// Walk the AppID hierarchy rooted at the specified module.
LogicalResult
AppIDIndex::walk(hw::HWModuleLike mod, SmallVectorImpl<AppIDAttr> &pathStack,
                 function_ref<void(AppIDPathAttr, Operation *)> fn) const {
  ModuleAppIDs *modIDs = containerAppIDs.lookup(mod);
  if (!modIDs) {
    mod.emitWarning("Module has no AppIDs");
    return success();
  }
  for (auto [appid, innerSym] : modIDs->getIndex()) {
    AppIDPathAttr path = AppIDPathAttr::get(
        mod.getContext(), FlatSymbolRefAttr::get(mod.getNameAttr()), pathStack);
    fn(path, innerSym);

    if (auto inst = dyn_cast<hw::HWInstanceLike>(innerSym.getOperation())) {
      auto tgtMod = dyn_cast<hw::HWModuleLike>(
          symCache.getDefinition(inst.getReferencedModuleNameAttr()));
      // Do the assert here to get a more precise message.
      assert(tgtMod && "invalid module reference");

      AppIDAttr appid =
          innerSym->getAttrOfType<AppIDAttr>(AppIDAttr::AppIDAttrName);
      if (appid)
        pathStack.push_back(appid);
      LogicalResult rc = walk(tgtMod, pathStack, fn);
      if (appid)
        pathStack.pop_back();
      if (failed(rc))
        return failure();
    }
  }
  return success();
}

LogicalResult
AppIDIndex::walk(hw::HWModuleLike top,
                 function_ref<void(AppIDPathAttr, Operation *)> fn) const {
  SmallVector<AppIDAttr, 8> path;
  return walk(top, path, fn);
}
LogicalResult
AppIDIndex::walk(StringRef top,
                 function_ref<void(AppIDPathAttr, Operation *)> fn) const {
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
    path.push_back(op->getInnerRef());

    if (getAppID(*op))
      break;

    if (auto inst = dyn_cast<hw::HWInstanceLike>(op->getOperation()))
      fromMod = cast<hw::HWModuleLike>(
          symCache.getDefinition(inst.getReferencedModuleNameAttr()));
    else
      assert(false && "Search bottomed out");
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

  auto done = mod.walk([&](hw::InnerSymbolOpInterface op) {
    // If an op has an appid attribute, add it to the index and terminate the
    // DFS (since AppIDs only get 'bubbled up' until they encounter an ID'd
    // instantiation).
    if (AppIDAttr appid = getAppID(op)) {
      if (failed(appIDs->add(appid, op, false)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    // If we encounter an instance op which isn't ID'd...
    if (auto inst = dyn_cast<hw::HWInstanceLike>(op.getOperation())) {
      auto tgtMod = dyn_cast<hw::HWModuleLike>(
          symCache.getDefinition(inst.getReferencedModuleNameAttr()));
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
