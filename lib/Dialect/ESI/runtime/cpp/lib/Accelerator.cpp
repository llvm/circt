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

using namespace esi;
using namespace esi::services;

CustomService::CustomService(AppIDPath idPath,
                             const ServiceImplDetails &details,
                             const HWClientDetails &clients) {
  _serviceSymbol = std::any_cast<std::string>(details.at("service"));
  // Strip off initial '@'.
  _serviceSymbol = _serviceSymbol.substr(1);
}

namespace esi {
services::Service *Accelerator::getService(Service::Type svcType, AppIDPath id,
                                           ServiceImplDetails details,
                                           HWClientDetails clients) {
  std::unique_ptr<Service> &cacheEntry =
      serviceCache[std::make_tuple(&svcType, id)];
  if (cacheEntry == nullptr)
    cacheEntry =
        std::unique_ptr<Service>(createService(svcType, id, details, clients));
  return cacheEntry.get();
}

namespace registry {
namespace internal {

static std::map<std::string, BackendCreate> backendRegistry;
void registerBackend(std::string name, BackendCreate create) {
  if (backendRegistry.count(name))
    throw std::runtime_error("Backend already exists in registry");
  backendRegistry[name] = create;
}
} // namespace internal

std::unique_ptr<Accelerator> connect(std::string backend,
                                     std::string connection) {
  auto f = internal::backendRegistry.find(backend);
  if (f == internal::backendRegistry.end())
    throw std::runtime_error("Backend not found");
  return f->second(connection);
}

} // namespace registry
} // namespace esi
