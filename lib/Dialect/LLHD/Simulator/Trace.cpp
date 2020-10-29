#include "Trace.h"

#include "llvm/Support/raw_ostream.h"

#include <regex>

using namespace circt::llhd::sim;

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
    for (size_t i = 0, e = sig.elements.size(); i < e; ++i) {
      pushChange(inst, sig, i);
    }
  } else {
    pushChange(inst, sig);
  }
}

void Trace::addChange(unsigned sigIndex) {
  currentTime = state->time;
  if (mode == full)
    addChangeFull(sigIndex);
  else if (mode == reduced)
    addChangeReduced(sigIndex);
  else if (mode == merged || mode == mergedReduce || mode == namedOnly)
    addChangeMerged(sigIndex);
}

void Trace::addChangeFull(unsigned sigIndex) {
  auto &sig = state->signals[sigIndex];
  for (auto inst : sig.triggers) {
    pushAllChanges(inst, sig);
  }
}

void Trace::addChangeReduced(unsigned sigIndex) {
  auto &sig = state->signals[sigIndex];
  auto root = state->root;
  if (sig.owner == root) {
    pushAllChanges(root, sig);
  }
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
    mergedChanges[std::make_pair(sigIndex, 0)] = valueDump;
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
    auto &sig = state->signals[sigIndex];
    auto change = elem.second;
    // Filter out changes that do not actually introduce changes
    if (lastValue[elem.first] != change) {
      // Add the changes for all signals.
      if (mode == merged) {
        for (auto inst : sig.triggers) {
          if (sig.elements.size() > 0) {
            std::string path;
            llvm::raw_string_ostream ss(path);
            ss << state->instances[inst].path << '/' << sig.name << '['
               << elem.first.second << ']';
            changes.push_back(std::make_pair(path, change));
          } else {
            auto path = state->instances[inst].path + '/' + sig.name;
            changes.push_back(std::make_pair(path, change));
          }
        }
      } else if (mode == mergedReduce || mode == namedOnly) {
        // Add the changes for the top-level signals only.
        auto sigIndex = elem.first.first;
        auto &sig = state->signals[sigIndex];
        auto root = state->root;
        if (sig.owner == root &&
            (mode == mergedReduce ||
             (mode == namedOnly && sig.owner == root &&
              !std::regex_match(sig.name, std::regex("(sig)?[0-9]*"))))) {
          if (sig.elements.size() > 0) {
            std::string path;
            llvm::raw_string_ostream ss(path);
            ss << state->instances[root].path << '/' << sig.name << '['
               << elem.first.second << ']';
            changes.push_back(std::make_pair(path, change));
          } else {
            auto path = state->instances[root].path + '/' + sig.name;
            changes.push_back(std::make_pair(path, change));
          }
        }
      }
      lastValue[elem.first] = change;
    }
  }

  std::sort(changes.begin(), changes.end(),
            [](std::pair<std::string, std::string> &lhs,
               std::pair<std::string, std::string> &rhs) -> bool {
              return lhs.first < rhs.first;
            });
  out << currentTime.time << "ps\n";
  for (auto change : changes) {
    out << "  " << change.first << "  " << change.second << "\n";
  }
  mergedChanges.clear();
  changes.clear();
}