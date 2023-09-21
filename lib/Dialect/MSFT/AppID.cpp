//===- AppID.cpp - AppID related code -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/MSFT/AppID.h"

#include "circt/Support/InstanceGraph.h"

using namespace circt;
using namespace msft;

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

private:
  // Operations involved in appids.
  DenseMap<AppIDAttr, hw::InnerSymbolOpInterface> childAppIDPaths;
};

AppIDIndex::AppIDIndex(Operation *mlirTop) : valid(true), mlirTop(mlirTop) {
  Block &topBlock = mlirTop->getRegion(0).front();
  symCache.addDefinitions(mlirTop);
  symCache.freeze();

  // Build a cache for the existing dynamic instance hierarchy.
  for (auto instHierOp : topBlock.getOps<InstanceHierarchyOp>()) {
    dynHierRoots[instHierOp.getTopModuleRefAttr()] = instHierOp;
    instHierOp.walk([this](DynamicInstanceOp dyninst) {
      auto &childInsts = dynInstChildLookup[dyninst];
      for (auto child : dyninst.getOps<DynamicInstanceOp>())
        childInsts[child.getInstanceRefAttr()] = child;
    });
  }

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

    if (op->getOperation()->hasAttr(AppIDAttr::AppIDAttrName))
      break;

    if (auto inst = dyn_cast<hw::HWInstanceLike>(op->getOperation()))
      fromMod = cast<hw::HWModuleLike>(
          symCache.getDefinition(inst.getReferencedModuleNameAttr()));
    else
      assert(false && "Search bottomed out");
  } while (true);
  return ArrayAttr::get(fromMod.getContext(), path);
}

FailureOr<DynamicInstanceOp> AppIDIndex::getInstance(AppIDPathAttr path,
                                                     Location loc) const {
  // Resolve the module at the root of the path.
  FlatSymbolRefAttr rootSym = path.getRoot();
  auto rootMod =
      dyn_cast_or_null<hw::HWModuleLike>(symCache.getDefinition(rootSym));
  if (!rootMod)
    return emitError(loc, "could not find module '") << rootSym << "'";

  // Get or create.
  auto &dynHierRoot = dynHierRoots[rootSym];
  if (dynHierRoot == nullptr) {
    dynHierRoot = OpBuilder::atBlockEnd(&mlirTop->getRegion(0).front())
                      .create<InstanceHierarchyOp>(loc, rootSym, StringAttr());
    dynHierRoot->getRegion(0).emplaceBlock();
  }

  // Resolve the rest of the path.
  return getSubInstance(rootMod, dynHierRoot, path.getPath(), loc);
}

DynamicInstanceOp AppIDIndex::getOrCreate(Operation *parent,
                                          hw::InnerRefAttr name,
                                          Location loc) const {
  NamedChildren &children = dynInstChildLookup[parent];
  DynamicInstanceOp &child = children[name];
  if (child)
    return child;
  auto dyninst = OpBuilder::atBlockEnd(&parent->getRegion(0).front())
                     .create<DynamicInstanceOp>(loc, name);
  dyninst->getRegion(0).emplaceBlock();
  return dyninst;
}

FailureOr<DynamicInstanceOp>
AppIDIndex::getSubInstance(hw::HWModuleLike rootMod,
                           InstanceHierarchyOp dynInstRoot,
                           ArrayRef<AppIDAttr> subpath, Location loc) const {

  // Loop-carried iteration variables.
  Operation *dynInst = dynInstRoot;
  hw::HWModuleLike mod = rootMod;
  assert(mod && "The first iteration assumes mod to be non-null.");
  hw::InnerSymbolOpInterface op;

  // Iterate through the components, resolving them one at a time and chaining
  // the results together.
  for (auto component : subpath) {
    if (!mod)
      return (emitError(loc, "could not find appid '") << component << "'")
                 .attachNote(op.getLoc())
             << "bottomed out instance hierarchy here";

    // Resolve the component.
    auto instOpFail = getSubInstance(mod, dynInst, component, loc);
    if (failed(instOpFail))
      return failure();

    std::tie(dynInst, op) = *instOpFail;
    if (auto inst = dyn_cast<hw::HWInstanceLike>(op.getOperation()))
      // If it points to an instance, setup the next iteration.
      mod = dyn_cast<hw::HWModuleLike>(
          symCache.getDefinition(inst.getReferencedModuleNameAttr()));
    else
      // Otherwise we've bottomed out.
      mod = nullptr;
  }
  return cast<DynamicInstanceOp>(dynInst);
}

FailureOr<std::pair<DynamicInstanceOp, hw::InnerSymbolOpInterface>>
AppIDIndex::getSubInstance(hw::HWModuleLike parentTgtMod, Operation *parent,
                           AppIDAttr appid, Location loc) const {
  // The exit and increments are not succinct enough to fit in a typical loop.
  // The reality is that this is better modeled as a recursive function, but
  // since it would be tail-recursive we unroll it here so as to not violate
  // clang-tidy rules.
  while (true) {
    // Find the the index for the target module and lookup the appid operation
    // within.
    auto fChildAppID = containerAppIDs.find(parentTgtMod);
    if (fChildAppID == containerAppIDs.end())
      return emitError(loc, "Could not find child appid '") << appid << "'";
    const ModuleAppIDs *cAppIDs = fChildAppID->getSecond();
    FailureOr<hw::InnerSymbolOpInterface> appidOpOrFail =
        cAppIDs->lookup(appid, loc);
    if (failed(appidOpOrFail))
      return failure();

    hw::InnerSymbolOpInterface appidOp = *appidOpOrFail;
    auto innerRef = hw::InnerRefAttr::get(parentTgtMod.getModuleNameAttr(),
                                          appidOp.getInnerNameAttr());

    // If the op has an appid itself, return the dynamic instance for it.
    if (auto opAppid =
            appidOp->getAttrOfType<AppIDAttr>(AppIDAttr::AppIDAttrName)) {
      assert(opAppid == appid && "Wrong appid");
      return std::make_pair(getOrCreate(parent, innerRef, loc), appidOp);
    }

    if (auto inst = dyn_cast<hw::HWInstanceLike>(appidOp.getOperation())) {
      // If 'op' is an instantiantion, continue the search down.
      parentTgtMod = cast<hw::HWModuleLike>(
          symCache.getDefinition(inst.getReferencedModuleNameAttr()));
      parent = getOrCreate(parent, innerRef, loc);
    } else {
      // Otherwise, fail.
      return emitError(loc, "could not find appid '") << appid << "'";
    }
  }
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
    if (auto appid = op->getAttrOfType<AppIDAttr>(AppIDAttr::AppIDAttrName)) {
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
