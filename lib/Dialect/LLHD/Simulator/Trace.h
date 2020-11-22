//===- Trace.h - Simulation trace definition --------------------*- C++ -*-===//
//
// This file defines the Trace class, used to handle the signal trace generation
// for the llhd-sim tool.
//
//===----------------------------------------------------------------------===//

// clang-tidy seems to expect the absolute path in the header guard on some
// systems, so just disable it.
// NOLINTNEXTLINE(llvm-header-guard)
#ifndef CIRCT_DIALECT_LLHD_SIMULATOR_TRACE_H
#define CIRCT_DIALECT_LLHD_SIMULATOR_TRACE_H

#include "State.h"

#include <map>
#include <vector>

namespace llvm {
class raw_ostream;
} // namespace llvm

namespace circt {
namespace llhd {
namespace sim {

enum TraceMode { full, reduced, merged, mergedReduce, namedOnly };

class Trace {
  llvm::raw_ostream &out;
  std::unique_ptr<State> const &state;
  TraceMode mode;
  Time currentTime;
  // Each entry defines if the respective signal is active for tracing.
  std::vector<bool> isTraced;
  // Buffer of changes ready to be flushed.
  std::vector<std::pair<std::string, std::string>> changes;
  // Buffer of changes for the merged formats.
  std::map<std::pair<unsigned, int>, std::string> mergedChanges;
  // Buffer of last dumped change for each signal.
  std::map<std::pair<std::string, int>, std::string> lastValue;

  /// Push one change to the changes vector.
  void pushChange(std::string inst, unsigned sigIndex, int elem);
  /// Push one change for each element of a signal if it is of a structured
  /// type, or the full signal otherwise.
  void pushAllChanges(std::string inst, unsigned sigIndex);

  /// Add a merged change to the change buffer.
  void addChangeMerged(unsigned);

  /// Sorts the changes buffer lexicographically wrt. the hierarchical paths.
  void sortChanges();

  /// Flush the changes buffer to the output stream with full format.
  void flushFull();
  // Flush the changes buffer to the output stream with merged format.
  void flushMerged();

public:
  Trace(std::unique_ptr<State> const &state, llvm::raw_ostream &out,
        TraceMode mode);

  /// Add a value change to the trace changes buffer.
  void addChange(unsigned);

  /// Flush the changes buffer to the output stream. The flush can be forced for
  /// merged changes, flushing even if the next real-time step has not been
  /// reached.
  void flush(bool force = false);
};
} // namespace sim
} // namespace llhd
} // namespace circt

#endif // CIRCT_DIALECT_LLHD_SIMULATOR_TRACE_H
