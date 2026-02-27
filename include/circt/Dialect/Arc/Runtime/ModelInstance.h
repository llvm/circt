//===- ModelInstance.h - Instance of a model in the ArcRuntime ------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This declares the context for a model instance in the default implementation
// of the ArcRuntime library.
//
// This file is implementation specific and not part of the ArcRuntime's API.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_DIALECT_ARC_RUNTIME_MODELINSTANCE_H
#define CIRCT_DIALECT_ARC_RUNTIME_MODELINSTANCE_H

#include "circt/Dialect/Arc/Runtime/ArcRuntime.h"
#include "circt/Dialect/Arc/Runtime/Common.h"
#include "circt/Dialect/Arc/Runtime/TraceEncoder.h"

#include <filesystem>
#include <memory>
#include <string>

namespace circt {
namespace arc {
namespace runtime {
namespace impl {

class ModelInstance {
public:
  ModelInstance() = delete;
  ModelInstance(const ArcRuntimeModelInfo *modelInfo, const char *args,
                ArcState *state);
  ~ModelInstance();

  const char *getModelName() const {
    return !!modelInfo->modelName ? modelInfo->modelName : "<NULL>";
  }

  void onInitialized(ArcState *mutableState);
  void onEval(ArcState *mutableState);
  uint64_t *swapTraceBuffer();

private:
  void parseArgs(const char *args);
  std::filesystem::path getTraceFilePath(const std::string &suffix);

  const uint64_t instanceID;
  const ArcRuntimeModelInfo *const modelInfo;
  const ArcState *const state;
  enum class TraceMode { DUMMY, VCD };
  TraceMode traceMode;
  std::optional<std::string> traceFileArg;
  std::unique_ptr<TraceEncoder> traceEncoder;
  bool verbose = false;
  uint64_t stepCounter = 0;
};

} // namespace impl
} // namespace runtime
} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_RUNTIME_MODELINSTANCE_H
