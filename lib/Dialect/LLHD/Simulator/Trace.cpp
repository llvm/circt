#include "Trace.h"

#include "llvm/Support/raw_ostream.h"

#include <regex>

using namespace circt::llhd::sim;

Trace::Trace(std::unique_ptr<State> const &state, llvm::raw_ostream &out,
             TraceMode mode)
    : out(out), state(state), mode(mode) {
  auto root = state->root;
  for (auto &sig : state->signals) {
    if (mode != full && mode != merged && sig.owner != root) {
      isTraced.push_back(false);
    } else if (mode == namedOnly &&
               std::regex_match(sig.name, std::regex("(sig)?[0-9]*"))) {
      isTraced.push_back(false);
    } else {
      isTraced.push_back(true);
    }
  }
}

void Trace::pushChange(std::string inst, Signal &sig, int elem = -1) {
  std::string valueDump;
  std::string path;
  llvm::raw_string_ostream ss(path);

  ss << state->instances[inst].path << '/' << sig.name;

  if (elem >= 0) {
    // Add element index to the hierarchical path.
    ss << '[' << elem << ']';
    // Get element value dump.
    valueDump = sig.dump(elem);
  } else {
    // Get signal value dump.
    valueDump = sig.dump();
  }

  changes.push_back(std::make_pair(path, valueDump));
}

void Trace::pushAllChanges(std::string inst, Signal &sig) {
  if (sig.elements.size() > 0) {
    // Push changes for all signal elements.
    for (size_t i = 0, e = sig.elements.size(); i < e; ++i) {
      pushChange(inst, sig, i);
    }
  } else {
    // Push one change for the whole signal.
    pushChange(inst, sig);
  }
}

void Trace::addChange(unsigned sigIndex) {
  currentTime = state->time;
  if (isTraced[sigIndex]) {
    if (mode == full)
      addChangeFull(sigIndex);
    else if (mode == reduced)
      addChangeReduced(sigIndex);
    else if (mode == merged || mode == mergedReduce || mode == namedOnly)
      addChangeMerged(sigIndex);
  }
}

void Trace::addChangeFull(unsigned sigIndex) {
  auto &sig = state->signals[sigIndex];

  // Add a change for each connected instance.
  for (auto inst : sig.triggers) {
    pushAllChanges(inst, sig);
  }
}

void Trace::addChangeReduced(unsigned sigIndex) {
  auto &sig = state->signals[sigIndex];
  pushAllChanges(state->root, sig);
}

void Trace::addChangeMerged(unsigned sigIndex) {
  auto &sig = state->signals[sigIndex];
  if (sig.elements.size() > 0) {
    // Add a change for all sub-elements
    for (size_t i = 0, e = sig.elements.size(); i < e; ++i) {
      auto valueDump = sig.dump(i);
      mergedChanges[std::make_pair(sigIndex, i)] = valueDump;
    }
  } else {
    // Add one change for the whole signal.
    auto valueDump = sig.dump();
    mergedChanges[std::make_pair(sigIndex, -1)] = valueDump;
  }
}

void Trace::flush(bool force) {
  if (changes.size() > 0 || mergedChanges.size() > 0) {
    if (mode == full || mode == reduced)
      flushFull();
    else if (mode == merged || mode == mergedReduce || mode == namedOnly)
      if (state->time.time > currentTime.time || force)
        flushMerged();
  }
}

void Trace::flushFull() {
  std::sort(changes.begin(), changes.end(),
            [](std::pair<std::string, std::string> &lhs,
               std::pair<std::string, std::string> &rhs) -> bool {
              return lhs.first < rhs.first;
            });
  auto timeDump = currentTime.dump();
  for (auto change : changes) {
    out << timeDump << "  " << change.first << "  " << change.second << "\n";
  }
  changes.clear();
}

void Trace::flushMerged() {
  // Move the merged changes to the changes vector for dumping.
  for (auto elem : mergedChanges) {
    auto sigIndex = elem.first.first;
    auto sigElem = elem.first.second;
    auto &sig = state->signals[sigIndex];
    auto change = elem.second;
    // Filter out changes that do not actually introduce changes
    if (lastValue[elem.first] != change) {
      if (mode == merged) {
        // Add the changes for all connected instances.
        for (auto inst : sig.triggers) {
          pushChange(inst, sig, sigElem);
        }
      } else {
        // Add change only for owner instance.
        pushChange(sig.owner, sig, sigElem);
      }
    }
    lastValue[elem.first] = change;
  }

  std::sort(changes.begin(), changes.end(),
            [](std::pair<std::string, std::string> &lhs,
               std::pair<std::string, std::string> &rhs) -> bool {
              return lhs.first < rhs.first;
            });

  // Flush the changes to output stream.
  out << currentTime.time << "ps\n";
  for (auto change : changes) {
    out << "  " << change.first << "  " << change.second << "\n";
  }
  mergedChanges.clear();
  changes.clear();
}