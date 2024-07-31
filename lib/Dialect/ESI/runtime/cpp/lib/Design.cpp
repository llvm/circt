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
static std::map<AppID, const BundlePort &>
buildIndex(const std::vector<std::unique_ptr<BundlePort>> &ports) {
  std::map<AppID, const BundlePort &> index;
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

} // namespace esi
