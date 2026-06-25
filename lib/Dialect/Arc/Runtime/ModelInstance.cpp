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
#include "circt/Dialect/Arc/Runtime/VCDTraceEncoder.h"
#ifdef CIRCT_LIBFST_ENABLED
#include "circt/Dialect/Arc/Runtime/FSTTraceEncoder.h"
#endif

#include <cassert>
#include <cctype>
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
    case TraceMode::VCD:
      traceEncoder = std::make_unique<VCDTraceEncoder>(
          modelInfo, mutableState, getTraceFilePath(".vcd"), verbose);
      break;
    case TraceMode::FST:
#ifdef CIRCT_LIBFST_ENABLED
      traceEncoder = std::make_unique<FSTTraceEncoder>(
          modelInfo, mutableState, getTraceFilePath(".fst"), verbose);
#else
      std::cerr << "[ArcRuntime] ERROR: FST tracing was requested but CIRCT "
                   "was not built with FST support (CIRCT_LIBFST_ENABLED=OFF)."
                << std::endl;
      traceEncoder =
          std::make_unique<DummyTraceEncoder>(modelInfo, mutableState);
#endif
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

std::filesystem::path
ModelInstance::getTraceFilePath(const std::string &suffix) {
  auto it = arguments.find("traceFile");
  if (it != arguments.end() && it->second.has_value())
    return std::filesystem::path(*it->second);

  std::string saneName;
  if (modelInfo->modelName)
    saneName = std::string(modelInfo->modelName);
  for (auto &c : saneName) {
    if (c == ' ' || c == '/' || c == '\\')
      c = '_';
  }
  saneName += '_';
  saneName += std::to_string(instanceID);
  saneName += suffix;
  return std::filesystem::current_path() / std::filesystem::path(saneName);
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

// Parse the argument string into a map of key to optional value.
// Flags (bare keys without '=') map to std::nullopt. Keys occurring later
// override identical earlier keys. Quoted values (key="...") may contain ';'
// and support \" and \\ escape sequences. Malformed tokens are warned and
// skipped.
static std::map<std::string, std::optional<std::string>>
parseArgsToMap(std::string_view argStr) {
  std::map<std::string, std::optional<std::string>> result;

  enum class State { Key, AfterEq, Unquoted, Quoted, Escape, AfterQuote, Skip };
  State state = State::Key;

  std::string key;
  std::string value;
  bool hasValue = false;

  auto warn = [&](const char *msg) {
    std::cerr << "[ArcRuntime] WARNING: Malformed runtime argument: " << msg;
    if (!key.empty())
      std::cerr << " for key \"" << key << "\"";
    std::cerr << ", ignoring\n";
  };

  auto commit = [&] {
    result[std::move(key)] =
        hasValue ? std::optional(std::move(value)) : std::nullopt;
    key.clear();
    value.clear();
    hasValue = false;
    state = State::Key;
  };

  auto skipToNext = [&] {
    key.clear();
    value.clear();
    hasValue = false;
    state = State::Skip;
  };

  for (size_t i = 0; i <= argStr.size(); ++i) {
    const bool atEnd = (i == argStr.size());
    const char c = atEnd ? '\0' : argStr[i];

    switch (state) {
    case State::Key:
      if (atEnd || c == ';') {
        if (!key.empty())
          commit();
      } else if (std::isgraph(static_cast<unsigned char>(c)) && c != '"' &&
                 c != '=') {
        key += c;
      } else if (c == '=' && !key.empty()) {
        hasValue = true;
        state = State::AfterEq;
      } else {
        warn("Invalid key");
        skipToNext();
      }
      break;

    case State::AfterEq:
      if (c == '"' && !atEnd) {
        state = State::Quoted;
      } else if (atEnd || c == ';') {
        commit(); // empty value
      } else {
        value += c;
        state = State::Unquoted;
      }
      break;

    case State::Unquoted:
      if (atEnd || c == ';') {
        commit();
      } else if (c == '"') {
        warn("Unquoted value contains forbidden character '\"'");
        skipToNext();
      } else {
        value += c;
      }
      break;

    case State::Quoted:
      if (atEnd) {
        warn("Unterminated quoted value");
        skipToNext();
      } else if (c == '"') {
        state = State::AfterQuote;
      } else if (c == '\\') {
        state = State::Escape;
      } else {
        value += c;
      }
      break;

    case State::Escape:
      if (atEnd) {
        warn("Truncated escape sequence in quoted value");
        skipToNext();
      } else if (c == '"' || c == '\\') {
        value += c;
        state = State::Quoted;
      } else {
        warn("Invalid escape sequence in quoted value");
        skipToNext();
      }
      break;

    case State::AfterQuote:
      if (atEnd || c == ';') {
        commit();
      } else {
        warn("Unexpected content after closing quote");
        skipToNext();
      }
      break;

    case State::Skip:
      if (!atEnd && c == ';')
        state = State::Key;
      break;
    }
  }

  return result;
}

void ModelInstance::parseArgs(const char *args) {
  if (!args)
    return;
  auto argStr = std::string_view(args);
  arguments = parseArgsToMap(argStr);

  if (arguments.count("debug"))
    verbose = true;
  if (arguments.count("vcd"))
    traceMode = TraceMode::VCD;
  if (arguments.count("fst"))
    traceMode = TraceMode::FST;

  // Dump arguments
  if (verbose) {
    std::cout << "[ArcRuntime] Argument string for instance ID " << instanceID
              << ": " << argStr << std::endl;
    std::cout << "[ArcRuntime] Parsed argument(s):" << std::endl;
    for (const auto &[key, value] : arguments) {
      std::cout << "[ArcRuntime]   " << key;
      if (value.has_value())
        std::cout << " = \"" << *value << "\"";
      std::cout << std::endl;
    }
  }
}

} // namespace circt::arc::runtime::impl
