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

services::Service *AcceleratorConnection::getService(Service::Type svcType,
                                                     AppIDPath id,
                                                     std::string implName,
                                                     ServiceImplDetails details,
                                                     HWClientDetails clients) {
  unique_ptr<Service> &cacheEntry = serviceCache[make_tuple(&svcType, id)];
  if (cacheEntry == nullptr) {
    Service *svc = createService(svcType, id, implName, details, clients);
    if (!svc)
      svc = ServiceRegistry::createService(this, svcType, id, implName, details,
                                           clients);
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

unique_ptr<AcceleratorConnection> connect(Context &ctxt, string backend,
                                          string connection) {
  auto f = internal::backendRegistry.find(backend);
  if (f == internal::backendRegistry.end())
    throw runtime_error("Backend not found");
  return f->second(ctxt, connection);
}

} // namespace registry
} // namespace esi
