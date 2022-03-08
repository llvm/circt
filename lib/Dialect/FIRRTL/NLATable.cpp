//===- NLATable.cpp - Non-Local Anchor Table --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/FIRRTL/NLATable.h"
#include "circt/Dialect/FIRRTL/FIRRTLOps.h"
#include "mlir/IR/BuiltinOps.h"

using namespace circt;
using namespace firrtl;

NLATable::NLATable(Operation *operation) {
  if (auto mod = dyn_cast<mlir::ModuleOp>(operation))
    for (auto &op : *mod.getBody())
      if ((operation = dyn_cast<CircuitOp>(&op)))
        break;

  auto circuit = cast<CircuitOp>(operation);
  // We are assuming it's faster to iterate over the top level twice than cache
  // a large number of options.
  for (auto &op : *circuit.getBody()) {
    if (auto module = dyn_cast<FModuleLike>(op))
      symToOp[module.moduleNameAttr()] = module;
    if (auto nla = dyn_cast<NonLocalAnchor>(op)) {
      symToOp[nla.sym_nameAttr()] = nla;
      for (auto ent : nla.namepath()) {
        if (auto mod = ent.dyn_cast<FlatSymbolRefAttr>())
          nodeMap[mod.getAttr()].push_back(nla);
        else if (auto inr = ent.dyn_cast<hw::InnerRefAttr>())
          nodeMap[inr.getModule()].push_back(nla);
      }
    }
  }
}

void NLATable::renameModule(StringAttr oldModName, StringAttr newModName) {
  auto op = symToOp.find(oldModName);
  if (op == symToOp.end())
    return;
  auto iter = nodeMap.find(oldModName);
  if (iter == nodeMap.end())
    return;
  for (auto nla : iter->second)
    nla.updateModule(oldModName, newModName);
  nodeMap[newModName] = iter->second;
  nodeMap.erase(oldModName);
  symToOp[newModName] = op->second;
  symToOp.erase(oldModName);
}

void NLATable::updateModuleInNLA(StringAttr name, StringAttr oldModule,
                                 StringAttr newModule) {
  auto nlaOp = getNLA(name);
  if (!nlaOp)
    return;
  updateModuleInNLA(nlaOp, oldModule, newModule);
}

void NLATable::updateModuleInNLA(NonLocalAnchor nlaOp, StringAttr oldModule,
                                 StringAttr newModule) {
  nlaOp.updateModule(oldModule, newModule);
  auto &nlas = nodeMap[oldModule];
  auto *iter = std::find(nlas.begin(), nlas.end(), nlaOp);
  if (iter != nlas.end()) {
    nlas.erase(iter);
    if (nlas.empty())
      nodeMap.erase(oldModule);
    nodeMap[newModule].push_back(nlaOp);
  }
}

NonLocalAnchor NLATable::getNLA(StringAttr name) {
  auto *n = symToOp.lookup(name);
  return dyn_cast_or_null<NonLocalAnchor>(n);
}

FModuleLike NLATable::getModule(StringAttr name) {
  auto *n = symToOp.lookup(name);
  return dyn_cast_or_null<FModuleLike>(n);
}

// void NLATable::retargetInstance(InstanceOp inst, StringRef newModName) {
//     inst.moduleNameAttr(FlatSymbolRefAttr::get(inst.getContext(),
//     newModName));
// }

ArrayRef<NonLocalAnchor> NLATable::lookup(StringAttr name) {
  auto iter = nodeMap.find(name);
  if (iter == nodeMap.end())
    return {};
  return iter->second;
}

ArrayRef<NonLocalAnchor> NLATable::lookup(Operation *op) {
  auto name = op->getAttrOfType<StringAttr>("sym_name");
  if (!name)
    return {};
  return lookup(name);
}

void NLATable::eraseModule(StringAttr name) {
  symToOp.erase(name);
  nodeMap.erase(name);
}
