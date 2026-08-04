//===- BMCTrace.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "circt/Tools/circt-bmc/BMCTrace.h"

#include "llvm/ADT/SmallString.h"
#include <cassert>

circt::bmc::BMCTrace::BMCTrace(llvm::StringRef topName) : topName(topName) {}

size_t circt::bmc::BMCTrace::addSignal(llvm::StringRef name, unsigned width) {
  assert(!signalIndices.contains(name) && "duplicate trace signal");
  signalIndices.try_emplace(name, signals.size());
  signals.push_back({name.str(), width});
  for (auto &step : recorded)
    step.resize(signals.size());
  return signals.size() - 1;
}

void circt::bmc::BMCTrace::ensureStep(size_t step) {
  if (step >= recorded.size())
    recorded.resize(step + 1, Step(signals.size()));
}

void circt::bmc::BMCTrace::record(size_t step, size_t signal, Handle handle) {
  assert(signal < signals.size() && "signal index out of range");
  ensureStep(step);
  recorded[step][signal] = handle;
}

void circt::bmc::BMCTrace::record(size_t step, llvm::StringRef name,
                                  unsigned width, Handle handle) {
  auto it = signalIndices.find(name);
  size_t signal;
  if (it == signalIndices.end()) {
    signal = addSignal(name, width);
  } else {
    signal = it->second;
    assert(signals[signal].width == width && "trace signal width changed");
  }
  record(step, signal, handle);
}

std::optional<circt::bmc::BMCTrace::Handle>
circt::bmc::BMCTrace::lookup(size_t step, size_t signal) const {
  if (step >= recorded.size() || signal >= signals.size())
    return std::nullopt;
  return recorded[step][signal];
}

bool circt::bmc::BMCTrace::printTextTrace(llvm::raw_ostream &os,
                                          Evaluator evaluate) const {
  os << "counterexample for " << topName << ":\n";
  for (size_t step = 0, e = recorded.size(); step != e; ++step) {
    os << "cycle " << step << ":\n";
    for (size_t signal = 0, numSignals = signals.size(); signal != numSignals;
         ++signal) {
      auto handle = recorded[step][signal];
      if (!handle)
        return false;
      auto value = evaluate(*handle, signals[signal].width);
      if (!value || value->getBitWidth() != signals[signal].width)
        return false;
      llvm::SmallString<40> str;
      value->toString(str, /*Radix=*/16, /*Signed=*/false,
                      /*formatAsCLiteral=*/false, /*UpperCase=*/false);
      os << "  " << signals[signal].name << " = 0x" << str << "\n";
    }
  }
  return true;
}

extern "C" void circt::bmc::circt_bmc_record_trace(BMCTrace *trace,
                                                   uint32_t step,
                                                   const char *name,
                                                   uint32_t width,
                                                   BMCTrace::Handle handle) {
  if (!trace)
    return;
  trace->record(step, name, width, handle);
}
