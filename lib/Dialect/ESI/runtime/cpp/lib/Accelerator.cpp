//===- Accelerator.cpp - ESI accelerator system API -----------------------===//
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

#include "esi/Accelerator.h"

#include <map>
#include <stdexcept>

using namespace std;

using namespace esi;
using namespace esi::services;

namespace esi {

/// Build an index of children by AppID.
static map<AppID, Instance *>
buildIndex(const vector<unique_ptr<Instance>> &insts) {
  map<AppID, Instance *> index;
  for (auto &item : insts)
    index[item->getID()] = item.get();
  return index;
}

/// Build an index of ports by AppID.
static map<AppID, const BundlePort &>
buildIndex(const vector<BundlePort> &ports) {
  map<AppID, const BundlePort &> index;
  for (auto &item : ports)
    index.emplace(item.getID(), item);
  return index;
}

HWModule::HWModule(std::optional<ModuleInfo> info,
                   std::vector<std::unique_ptr<Instance>> children,
                   std::vector<services::Service *> services,
                   std::vector<BundlePort> ports)
    : info(info), children(std::move(children)),
      childIndex(buildIndex(this->children)), services(services), ports(ports),
      portIndex(buildIndex(this->ports)) {}

services::Service *AcceleratorConnection::getService(Service::Type svcType,
                                                     AppIDPath id,
                                                     std::string implName,
                                                     ServiceImplDetails details,
                                                     HWClientDetails clients) {
  unique_ptr<Service> &cacheEntry = serviceCache[make_tuple(&svcType, id)];
  if (cacheEntry == nullptr) {
    Service *svc = createService(svcType, id, implName, details, clients);
    if (!svc)
      return nullptr;
    cacheEntry = unique_ptr<Service>(svc);
  }
  return cacheEntry.get();
}

namespace registry {
namespace internal {

static map<string, BackendCreate> backendRegistry;
void registerBackend(string name, BackendCreate create) {
  if (backendRegistry.count(name))
    throw runtime_error("Backend already exists in registry");
  backendRegistry[name] = create;
}
} // namespace internal

unique_ptr<AcceleratorConnection> connect(string backend, string connection) {
  auto f = internal::backendRegistry.find(backend);
  if (f == internal::backendRegistry.end())
    throw runtime_error("Backend not found");
  return f->second(connection);
}

} // namespace registry
} // namespace esi
