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

#include "circt/Dialect/Arc/Runtime/TraceTaps.h"

#include <cassert>
#include <iostream>
#include <string_view>
#include <vector>

using namespace circt::arc::runtime;

namespace circt::arc::runtime::impl {

// Global counter for instances
static uint64_t instanceIDsGlobal = 0;

ModelInstance::ModelInstance(const ArcRuntimeModelInfo *modelInfo,
                             const char *args, ArcState *mutableState)
    : instanceID(instanceIDsGlobal++), modelInfo(modelInfo),
      state(mutableState) {
  bool hasTraceInstrumentation = !!modelInfo->traceInfo;
  traceMode = TraceMode::DUMMY;
  parseArgs(args);

  if (verbose) {
    std::cout << "[ArcRuntime] "
              << "Created instance"
              << " of model \"" << getModelName() << "\""
              << " with ID " << instanceID << std::endl;
    std::cout << "[ArcRuntime] Model \"" << getModelName() << "\"";
    if (hasTraceInstrumentation)
      std::cout << " has trace instrumentation." << std::endl;
    else
      std::cout << " does not have trace instrumentation." << std::endl;
  }

  if (!hasTraceInstrumentation && traceMode != TraceMode::DUMMY)
    std::cerr
        << "[ArcRuntime] WARNING: "
        << "Tracing has been requested but model \"" << getModelName()
        << "\" contains no instrumentation."
        << " No trace will be produced.\n\t\tMake sure to compile the model"
           " with tracing enabled and that it contains observed signals."
        << std::endl;

  if (hasTraceInstrumentation) {
    switch (traceMode) {
    case TraceMode::DUMMY:
      traceEncoder =
          std::make_unique<DummyTraceEncoder>(modelInfo, mutableState);
      break;
    }
  } else {
    traceEncoder = {};
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
  if (traceEncoder)
    traceEncoder->finish(state);
}

void ModelInstance::onEval(ArcState *mutableState) {
  assert(mutableState == state);
  ++stepCounter;
  if (traceEncoder)
    traceEncoder->step(state);
}

void ModelInstance::onInitialized(ArcState *mutableState) {
  assert(mutableState == state);
  if (traceEncoder)
    traceEncoder->run(mutableState);

  if (verbose) {
    std::cout << "[ArcRuntime] "
              << "Instance with ID " << instanceID << " initialized"
              << std::endl;
  }
}

uint64_t *ModelInstance::swapTraceBuffer() {
  if (!traceEncoder)
    impl::fatalError(
        "swapTraceBuffer called on model without trace instrumentation");
  if (verbose)
    std::cout << "[ArcRuntime] Consuming trace buffer of size "
              << state->traceBufferSize << " for instance ID " << instanceID
              << std::endl;
  return traceEncoder->dispatch(state->traceBufferSize);
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
