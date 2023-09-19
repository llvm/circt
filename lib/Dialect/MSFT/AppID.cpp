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

AppIDIndex::AppIDIndex(Operation *mlirTop) : valid(true), mlirTop(mlirTop) {
  Block &topBlock = mlirTop->getRegion(0).front();
  symCache.addDefinitions(mlirTop);
  symCache.freeze();

  for (auto instHierOp : topBlock.getOps<InstanceHierarchyOp>()) {
    dynHierRoots[instHierOp.getTopModuleRefAttr()] = instHierOp;
    instHierOp.walk([this](DynamicInstanceOp dyninst) {
      auto &childInsts = dynInstChildLookup[dyninst];
      for (auto child : dyninst.getOps<DynamicInstanceOp>())
        childInsts[child.getInstanceRefAttr()] = child;
    });
  }

  for (auto mod : topBlock.getOps<hw::HWModuleLike>())
    if (failed(process(mod))) {
      valid = false;
      break;
    }
}

FailureOr<DynamicInstanceOp> AppIDIndex::getInstance(AppIDPathAttr path,
                                                     Location loc) const {
  FlatSymbolRefAttr rootSym = path.getRoot();
  auto rootMod =
      dyn_cast_or_null<hw::HWModuleLike>(symCache.getDefinition(rootSym));
  if (!rootMod)
    return emitError(loc, "could not find module '") << rootSym << "'";

  auto &dynHierRoot = dynHierRoots[rootSym];
  if (dynHierRoot == nullptr) {
    dynHierRoot = OpBuilder::atBlockEnd(&mlirTop->getRegion(0).front())
                      .create<InstanceHierarchyOp>(loc, rootSym, StringAttr());
    dynHierRoot->getRegion(0).emplaceBlock();
  }
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

FailureOr<std::pair<DynamicInstanceOp, hw::InnerSymbolOpInterface>>
AppIDIndex::getSubInstance(hw::HWModuleLike parentTgtMod, Operation *parent,
                           AppIDAttr appid, Location loc) const {
  do {
    auto fChildAppID = containerAppIDs.find(parentTgtMod);
    if (fChildAppID == containerAppIDs.end())
      return emitError(loc, "Could not find child appid '") << appid << "'";
    const ChildAppIDs *cAppIDs = fChildAppID->getSecond();
    FailureOr<Operation *> appidOpOrFail =
        cAppIDs->lookup(parentTgtMod, appid, loc);
    if (failed(appidOpOrFail))
      return failure();

    auto appidOp = cast<hw::InnerSymbolOpInterface>(*appidOpOrFail);
    auto innerRef = hw::InnerRefAttr::get(parentTgtMod.getModuleNameAttr(),
                                          appidOp.getInnerNameAttr());
    if (auto opAppid =
            appidOp->getAttrOfType<AppIDAttr>(AppIDAttr::AppIDAttrName)) {
      if (opAppid != appid)
        return emitError(loc, "Wrong appid '")
               << opAppid << "'. Expected '" << appid << "'.";

      return std::make_pair(getOrCreate(parent, innerRef, loc), appidOp);
    }

    if (auto inst = dyn_cast<hw::HWInstanceLike>(appidOp.getOperation())) {
      parentTgtMod = cast<hw::HWModuleLike>(
          symCache.getDefinition(inst.getReferencedModuleNameAttr()));
      parent = getOrCreate(parent, innerRef, loc);
    } else {
      return emitError(loc, "could not find appid '") << appid << "'";
    }
  } while (true);
}

FailureOr<DynamicInstanceOp>
AppIDIndex::getSubInstance(hw::HWModuleLike mod, Operation *dynInstParent,
                           ArrayRef<AppIDAttr> subpath, Location loc) const {

  FailureOr<std::pair<DynamicInstanceOp, hw::InnerSymbolOpInterface>>
      instOpFail;
  for (auto component : subpath) {
    if (!mod)
      return emitError(loc, "could not find appid '") << component << "'";

    instOpFail = getSubInstance(mod, dynInstParent, component, loc);
    if (failed(instOpFail))
      return failure();

    auto [dyninst, op] = *instOpFail;
    if (auto inst = dyn_cast<hw::HWInstanceLike>(op.getOperation()))
      mod = dyn_cast<hw::HWModuleLike>(
          symCache.getDefinition(inst.getReferencedModuleNameAttr()));
    else
      mod = nullptr;
    dynInstParent = dyninst;
  }
  return (*instOpFail).first;
}

LogicalResult AppIDIndex::ChildAppIDs::add(AppIDAttr id, Operation *op,
                                           bool inherited) {
  auto innerSym = dyn_cast<hw::InnerSymbolOpInterface>(op);
  if (!innerSym || !innerSym.getInnerName())
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

FailureOr<Operation *> AppIDIndex::ChildAppIDs::lookup(Operation *op,
                                                       AppIDAttr id,
                                                       Location loc) const {
  auto f = childAppIDPaths.find(id);
  if (f == childAppIDPaths.end())
    return emitError(loc, "could not find appid '") << id << "'";
  return f->second;
}

FailureOr<const AppIDIndex::ChildAppIDs *>
AppIDIndex::process(hw::HWModuleLike mod) {
  ChildAppIDs *&appIDs = containerAppIDs[mod];
  if (appIDs != nullptr)
    return appIDs;
  appIDs = new ChildAppIDs();

  auto done = mod.walk([&](Operation *op) {
    if (auto appid = op->getAttrOfType<AppIDAttr>(AppIDAttr::AppIDAttrName)) {
      if (failed(appIDs->add(appid, op, false)))
        return WalkResult::interrupt();
      return WalkResult::advance();
    }

    if (auto inst = dyn_cast<hw::HWInstanceLike>(op)) {
      auto tgtMod = dyn_cast<hw::HWModuleLike>(
          symCache.getDefinition(inst.getReferencedModuleNameAttr()));
      assert(tgtMod && "invalid module reference");
      FailureOr<const ChildAppIDs *> childIds = process(tgtMod);
      if (failed(childIds))
        return WalkResult::interrupt();

      for (AppIDAttr appid : (*childIds)->getAppIDs())
        if (failed(appIDs->add(appid, inst, true)))
          return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (done.wasInterrupted())
    return failure();
  return appIDs;
}
