//===- Design.cpp - ESI design hierarchy implementation -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT (lib/dialect/ESI/runtime/cpp/).
//
//===----------------------------------------------------------------------===//

#include "esi/Design.h"

#include <map>
#include <stdexcept>

using namespace esi;

namespace esi {

/// Build an index of children by AppID.
static std::map<AppID, Instance *>
buildIndex(const std::vector<std::unique_ptr<Instance>> &insts) {
  std::map<AppID, Instance *> index;
  for (auto &item : insts)
    index[item->getID()] = item.get();
  return index;
}

/// Build an index of ports by AppID.
static std::map<AppID, BundlePort &>
buildIndex(const std::vector<std::unique_ptr<BundlePort>> &ports) {
  std::map<AppID, BundlePort &> index;
  for (auto &item : ports)
    index.emplace(item->getID(), *item);
  return index;
}

HWModule::HWModule(std::optional<ModuleInfo> info,
                   std::vector<std::unique_ptr<Instance>> children,
                   std::vector<services::Service *> services,
                   std::vector<std::unique_ptr<BundlePort>> &ports)
    : info(info), children(std::move(children)),
      childIndex(buildIndex(this->children)), services(services),
      ports(std::move(ports)), portIndex(buildIndex(this->ports)) {}

bool HWModule::poll() {
  bool result = false;
  for (auto &port : ports)
    result |= port->poll();
  for (auto &child : children)
    result |= child->poll();
  return result;
}

const HWModule *HWModule::resolveInst(const AppIDPath &path,
                                      AppIDPath &lastLookup) const {
  const HWModule *hwmodule = this;
  for (auto &id : path) {
    lastLookup.push_back(id);
    auto childIter = hwmodule->childIndex.find(id);
    if (childIter == hwmodule->childIndex.end())
      return nullptr;
    hwmodule = childIter->second;
  }
  return hwmodule;
}

BundlePort *HWModule::resolvePort(const AppIDPath &path,
                                  AppIDPath &lastLookup) const {
  AppID portID = path.back();
  AppIDPath instPath = path;
  instPath.pop_back();
  const HWModule *hwmodule = resolveInst(instPath, lastLookup);
  lastLookup.push_back(portID);
  const auto &ports = hwmodule->getPorts();
  const auto &portIter = ports.find(portID);
  if (portIter == ports.end())
    return nullptr;
  return &portIter->second;
}

} // namespace esi
