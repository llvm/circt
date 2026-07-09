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
#include <map>
#include <memory>
#include <optional>
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
  // Runtime argument keys
  /// Enable verbose debug output
  inline static const std::string kArgKeyDebug = "debug";
  /// Select FST trace mode
  inline static const std::string kArgKeyFst = "fst";
  /// Workdir-relative or absolute path to trace file
  inline static const std::string kArgKeyTraceFile = "traceFile";
  /// Select VCD trace mode
  inline static const std::string kArgKeyVcd = "vcd";
  /// Instance's working directory. Absolute or relative to process workdir.
  inline static const std::string kArgKeyWorkDir = "workDir";

  /// Parse and initialize the instance settings from the given argument string.
  void parseArgs(const char *args);
  /// Get the path to the output trace file. Creates a default file name with
  /// the given suffix within the working directory if not explicitly set by the
  /// runtime arguments.
  std::filesystem::path getTraceFilePath(const std::string &suffix);

  const uint64_t instanceID;
  const ArcRuntimeModelInfo *const modelInfo;
  const ArcState *const state;
  std::map<std::string, std::optional<std::string>> arguments;
  /// The path to the instance's working directory. Matches the process
  /// working directory if not provided by a runtime argument.
  std::filesystem::path workDir;
  // FST is always in the enum so headers don't depend on build configuration.
  // If FST is selected at runtime but not compiled in, an error is emitted.
  enum class TraceMode { DUMMY, VCD, FST };
  TraceMode traceMode;
  std::unique_ptr<TraceEncoder> traceEncoder;
  bool verbose = false;
  uint64_t stepCounter = 0;
};

} // namespace impl
} // namespace runtime
} // namespace arc
} // namespace circt

#endif // CIRCT_DIALECT_ARC_RUNTIME_MODELINSTANCE_H
