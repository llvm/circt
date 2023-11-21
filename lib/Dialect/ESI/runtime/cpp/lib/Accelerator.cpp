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

CustomService::CustomService(AppIDPath idPath,
                             const ServiceImplDetails &details,
                             const HWClientDetails &clients)
    : id(idPath) {
  if (auto f = details.find("service"); f != details.end()) {
    serviceSymbol = any_cast<string>(f->second);
    // Strip off initial '@'.
    serviceSymbol = serviceSymbol.substr(1);
  }
}

namespace esi {
services::Service *Accelerator::getService(Service::Type svcType, AppIDPath id,
                                           ServiceImplDetails details,
                                           HWClientDetails clients) {
  unique_ptr<Service> &cacheEntry = serviceCache[make_tuple(&svcType, id)];
  if (cacheEntry == nullptr)
    cacheEntry =
        unique_ptr<Service>(createService(svcType, id, details, clients));
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

unique_ptr<Accelerator> connect(string backend, string connection) {
  auto f = internal::backendRegistry.find(backend);
  if (f == internal::backendRegistry.end())
    throw runtime_error("Backend not found");
  return f->second(connection);
}

} // namespace registry
} // namespace esi
