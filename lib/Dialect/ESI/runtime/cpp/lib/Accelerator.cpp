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
// should always be modified within CIRCT
// (lib/dialect/ESI/runtime/cpp/lib/backends/Cosim.cpp).
//
//===----------------------------------------------------------------------===//

#include "esi/Accelerator.h"

#include <map>
#include <stdexcept>

namespace esi {
static std::map<std::string, BackendCreate> BackendRegistry;

void registerBackend(std::string name, BackendCreate create) {
  if (BackendRegistry.count(name))
    throw std::runtime_error("Backend already exists in registry");
  BackendRegistry[name] = create;
}

std::unique_ptr<Accelerator> Accelerator::connect(std::string backend,
                                                  std::string connection) {
  auto f = BackendRegistry.find(backend);
  if (f == BackendRegistry.end())
    throw std::runtime_error("Backend not found");
  return f->second(connection);
}

} // namespace esi
