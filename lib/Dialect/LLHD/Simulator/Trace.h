#ifndef CIRCT_DIALECT_LLHD_SIMULATOR_TRACE_H
#define CIRCT_DIALECT_LLHD_SIMULATOR_TRACE_H

#include "State.h"

#include <map>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace circt {
namespace llhd {
namespace sim {

enum TraceMode { full, reduced, merged, mergedReduce, namedOnly };

/// Class for generating a signal change trace.
/// This offers various trace formats:
/// - full: a human-readable and diff-friendly trace which lists all the signal
/// changes for all time steps of the simulation. The changes inside a time step
/// are ordered by the signal paths.
/// - reduced: same as full, but only the signals of the top-level entity are
/// dumped.
/// - merged: a human-readable format that merges all the delta step and epsilon
/// step changes into their real-time step. This makes visual inspection against
/// other traces format easier, whenever they don't have infinitesimal sub-steps
/// (e.g. VCD).
/// - merged-reduce: same as merged, but only the signals of the top-level
/// entity are dumped.
class Trace {
  llvm::raw_ostream &out;
  std::unique_ptr<State> const &state;
  TraceMode mode;
  Time currentTime;
  std::vector<bool> isTraced;
  std::vector<std::pair<std::string, std::string>> changes;
  std::map<std::pair<unsigned, int>, std::string> mergedChanges;
  std::map<std::tuple<std::string, unsigned, int>, std::string> lastValue;

  void pushChange(std::string inst, unsigned sigIndex, int elem);
  void pushAllChanges(std::string inst, unsigned sigIndex);

  void addChangeMerged(unsigned);

  void sortChanges();

  void flushFull();
  void flushMerged();

public:
  Trace(std::unique_ptr<State> const &state, llvm::raw_ostream &out,
        TraceMode mode);

  /// Add a value change.
  void addChange(unsigned);

  /// Flush the changes to the output stream.
  void flush(bool force = false);
};
} // namespace sim
} // namespace llhd
} // namespace circt

#endif // CIRCT_DIALECT_LLHD_SIMULATOR_TRACE_H
