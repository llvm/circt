//===- BMCTrace.h - Runtime trace storage for circt-bmc ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the runtime trace store used by circt-bmc.
//
//===----------------------------------------------------------------------===//

#ifndef CIRCT_TOOLS_CIRCT_BMC_BMCTRACE_H
#define CIRCT_TOOLS_CIRCT_BMC_BMCTRACE_H

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace circt::bmc {

class BMCTrace {
public:
  /// Opaque per-signal value reference recorded for a specific cycle.
  using Handle = const void *;
  /// Callback used to materialize a recorded handle into a concrete value when
  /// formatting a trace. The provided width is the declared width of the
  /// signal, which may be zero for i0 values.
  using Evaluator =
      llvm::function_ref<std::optional<llvm::APInt>(Handle, unsigned width)>;

  /// Metadata for a tracked signal in the trace.
  struct Signal {
    std::string name;
    unsigned width;
  };

  /// Create an empty trace for the given top-level design/module name.
  explicit BMCTrace(llvm::StringRef topName = "bmc");

  /// Register a signal to be tracked and return its stable index. Signals may
  /// have zero width to represent i0 values.
  size_t addSignal(llvm::StringRef name, unsigned width);
  /// Record the value handle for a tracked signal at the given cycle.
  void record(size_t step, size_t signal, Handle handle);
  /// Register a signal by name if necessary and record its value handle at the
  /// given cycle.
  void record(size_t step, llvm::StringRef name, unsigned width, Handle handle);

  llvm::StringRef getTopName() const { return topName; }
  llvm::ArrayRef<Signal> getSignals() const { return signals; }
  size_t getNumSteps() const { return recorded.size(); }
  /// Return the recorded handle for a signal at a given cycle, if present.
  std::optional<Handle> lookup(size_t step, size_t signal) const;

  /// Render the trace as cycle-by-cycle text using the provided evaluator to
  /// materialize values from recorded handles.
  bool printTextTrace(llvm::raw_ostream &os, Evaluator evaluate) const;

private:
  /// Per-cycle storage for all tracked signals.
  using Step = std::vector<std::optional<Handle>>;

  void ensureStep(size_t step);

  std::string topName;
  std::vector<Signal> signals;
  llvm::StringMap<size_t> signalIndices;
  std::vector<Step> recorded;
};

/// Runtime entry point called by JIT-compiled BMC code.
extern "C" void circt_bmc_record_trace(BMCTrace *trace, uint32_t step,
                                       const char *name, uint32_t width,
                                       BMCTrace::Handle handle);

} // namespace circt::bmc

#endif // CIRCT_TOOLS_CIRCT_BMC_BMCTRACE_H
