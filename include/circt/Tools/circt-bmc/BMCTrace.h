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
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>
#include <string>
#include <vector>

namespace circt::bmc {

class BMCTrace {
public:
  using Handle = const void *;
  using Evaluator =
      llvm::function_ref<std::optional<llvm::APInt>(Handle, unsigned width)>;

  struct Signal {
    std::string name;
    unsigned width;
  };

  explicit BMCTrace(llvm::StringRef topName = "bmc");

  size_t addSignal(llvm::StringRef name, unsigned width);
  void record(size_t step, size_t signal, Handle handle);

  llvm::StringRef getTopName() const { return topName; }
  llvm::ArrayRef<Signal> getSignals() const { return signals; }
  size_t getNumSteps() const { return recorded.size(); }
  std::optional<Handle> lookup(size_t step, size_t signal) const;

  bool printTextTrace(llvm::raw_ostream &os, Evaluator evaluate) const;

private:
  using Step = std::vector<std::optional<Handle>>;

  void ensureStep(size_t step);

  std::string topName;
  std::vector<Signal> signals;
  std::vector<Step> recorded;
};

} // namespace circt::bmc

#endif // CIRCT_TOOLS_CIRCT_BMC_BMCTRACE_H
