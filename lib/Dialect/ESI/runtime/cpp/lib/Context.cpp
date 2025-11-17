//===- Context.cpp --------------------------------------------------------===//
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

#include "esi/Context.h"
#include "esi/Accelerator.h"
#include <format>
#include <sstream>

using namespace esi;

void Context::registerType(Type *type) {
  if (types.count(type->getID()))
    throw std::runtime_error(
        std::format("Type '{}' already registered in context (type is '{}')",
                    type->getID(), type->toString()));
  types.emplace(type->getID(), std::unique_ptr<Type>(type));
}

std::unique_ptr<AcceleratorConnection>
Context::connect(std::string backend, std::string connection) {
  return registry::connect(*this, backend, connection);
}
