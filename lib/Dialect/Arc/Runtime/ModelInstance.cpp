//===- ModelInstance.cpp - Instance of a model in the ArcRuntime ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements the context for a model instance in the ArcRuntime library.
//
//===----------------------------------------------------------------------===//

#include "circt/Dialect/Arc/Runtime/ModelInstance.h"

#include <cassert>
#include <iostream>
#include <string_view>
#include <vector>

using namespace circt::arc::runtime;

namespace circt::arc::runtime::impl {

// Global counter for instances
static uint64_t instanceIDsGlobal = 0;

ModelInstance::ModelInstance(const ArcRuntimeModelInfo *modelInfo,
                             const char *args, ArcState *state)
    : instanceID(instanceIDsGlobal++), modelInfo(modelInfo), state(state) {
  parseArgs(args);
  if (verbose) {
    std::cout << "[ArcRuntime] "
              << "Created instance"
              << " of model \"" << getModelName() << "\""
              << " with ID " << instanceID << std::endl;
  }
}

ModelInstance::~ModelInstance() {
  if (verbose) {
    std::cout << "[ArcRuntime] "
              << "Deleting instance"
              << " of model \"" << getModelName() << "\""
              << " with ID " << instanceID << " after " << stepCounter
              << " step(s)" << std::endl;
  }
  assert(state->impl == static_cast<void *>(this) && "Inconsistent ArcState");
  state->impl = nullptr;
}

void ModelInstance::parseArgs(const char *args) {
  if (!args)
    return;

  // Split the argument string at semicolon delimiters
  auto argStr = std::string_view(args);
  if (argStr.empty())
    return;
  std::vector<std::string_view> options;
  size_t start = 0;
  size_t end = 0;

  while ((end = argStr.find(";", start)) != std::string_view::npos) {
    options.push_back(argStr.substr(start, end - start));
    start = end + 1;
  }
  if (start <= argStr.size())
    options.push_back(argStr.substr(start));

  // Parse the individual options
  for (auto &option : options) {
    if (option == "debug")
      verbose = true;
  }
}

} // namespace circt::arc::runtime::impl
