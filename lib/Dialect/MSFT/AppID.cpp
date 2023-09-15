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

AppIDIndex::AppIDIndex(Operation *mlirTop) : mlirTop(mlirTop) {
  Block &topBlock = mlirTop->getRegion(0).front();
  symCache.addDefinitions(mlirTop);
  symCache.freeze();

  for (auto instHierOp : topBlock.getOps<InstanceHierarchyOp>()) {
    dynHierRoots[instHierOp.getTopModuleRefAttr()] = instHierOp;
    instHierOp.walk([this](DynamicInstanceOp dyninst) {
      auto &childInsts = childIndex[dyninst];
      for (auto child : dyninst.getOps<DynamicInstanceOp>())
        childInsts[child.getInstanceRefAttr()] = child;
    });
  }

  igraph::InstanceGraph graph(mlirTop);
}

DynamicInstanceOp AppIDIndex::getInstance(AppIDPathAttr path) {
  MLIRContext *ctxt = path.getContext();
  FlatSymbolRefAttr rootSym = path.getRoot();
  auto rootMod = dyn_cast<hw::HWModuleLike>(symCache.getDefinition(rootSym));

  auto &dynHierRoot = dynHierRoots[rootSym];
  if (dynHierRoot == nullptr) {
    dynHierRoot = OpBuilder::atBlockEnd(&mlirTop->getRegion(0).front())
                      .create<InstanceHierarchyOp>(UnknownLoc::get(ctxt),
                                                   rootSym, StringAttr());
    dynHierRoot->getRegion(0).emplaceBlock();
  }
  return getSubInstance(rootMod, dynHierRoot, path.getPath());
}

DynamicInstanceOp AppIDIndex::getSubInstance(hw::HWModuleLike submod,
                                             InstanceHierarchyOp inst,
                                             ArrayRef<AppIDAttr> subpath) {
  return nullptr;
}

LogicalResult AppIDIndex::ChildAppIDs::addChildAppID(AppIDAttr id,
                                                     Operation *op) {
  hw::HWModuleLike owner = op->getParentOfType<hw::HWModuleLike>();
  if (!mod)
    mod = owner;
  assert(mod == owner && "Owner doesn't match");
}

LogicalResult
AppIDIndex::ChildAppIDs::process(hw::HWModuleLike modToProcess,
                                 igraph::InstanceGraph &instLookup) {
  if (!mod)
    mod = modToProcess;
  assert(mod == modToProcess && "Process mod doesn't match");
  assert(!processed && "Already processed");

  mod.walk([&](Operation *op) {
    // If an operation has an "appid" dialect attribute
    // "local" appid.
    if (auto appid = op->getAttrOfType<AppIDAttr>(AppIDAttr::AppIDAttrName)) {
      if (localAppIDs.find(appid) != localAppIDs.end()) {
        op->emitOpError("Found multiple identical AppIDs in same module")
                .attachNote(localAppIDs[appid]->getLoc())
            << "first AppID located here";
      } else {
        localAppIDs[appid] = op;
      }
      localAppIDBases.insert(appid.getName());
    }

    // Instance ops should expose their module's AppIDs recursively. Track
    // the number of instances which contain a base name.
    if (auto inst = dyn_cast<InstanceOp>(op)) {
      auto targetMod = dyn_cast<MSFTModuleOp>(
          topLevelSyms.getDefinition(inst.getModuleNameAttr()));
      if (targetMod && targetMod.getChildAppIDBases())
        for (auto base :
             targetMod.getChildAppIDBasesAttr().getAsRange<StringAttr>())
          appBaseCounts[base] += 1;
    }
  });
}
