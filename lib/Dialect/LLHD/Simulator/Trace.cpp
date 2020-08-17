#include "Trace.h"

#include "llvm/Support/raw_ostream.h"

using namespace mlir::llhd::sim;

void Trace::addChange(unsigned sigIndex) {
  currentTime = state->time;
  if (mode == full)
    addChangeFull(sigIndex);
  else if (mode == reduced)
    addChangeReduced(sigIndex);
  else if (mode == merged || mode == mergedReduce)
    addChangeMerged(sigIndex);
}

void Trace::addChangeFull(unsigned sigIndex) {
  auto sig = state->signals[sigIndex];
  // Add a change for all signal elements.
  if (sig.elements.size() > 0) {
    for (size_t i = 0, e = sig.elements.size(); i < e; ++i) {
      auto valueDump = sig.dump(i);
      for (auto inst : sig.triggers) {
        std::string path;
        llvm::raw_string_ostream ss(path);
        ss << state->instances[inst].path << '/' << sig.name << '[' << i << ']';
        changes.push_back(std::make_pair(path, valueDump));
      }
    }
  } else {
    // Add a change for the whole signal.
    auto valueDump = sig.dump();
    for (auto inst : sig.triggers) {
      auto path = state->instances[inst].path + '/' + sig.name;
      changes.push_back(std::make_pair(path, valueDump));
    }
  }
}

void Trace::addChangeReduced(unsigned sigIndex) {
  auto sig = state->signals[sigIndex];
  auto root = state->root;
  if (sig.owner == root) {
    // Add a change for all signal sub-elements.
    if (sig.elements.size() > 0) {
      for (size_t i = 0, e = sig.elements.size(); i < e; ++i) {
        auto valueDump = sig.dump(i);
        std::string path;
        llvm::raw_string_ostream ss(path);
        ss << state->instances[root].path << '/' << sig.name << '[' << i << ']';
        changes.push_back(std::make_pair(path, valueDump));
      }
    } else {
      // Add a change for the whole signal.
      auto valueDump = sig.dump();
      auto path = state->instances[root].path + '/' + sig.name;
      changes.push_back(std::make_tuple(path, valueDump));
    }
  }
}

void Trace::addChangeMerged(unsigned sigIndex) {
  auto sig = state->signals[sigIndex];
  auto time = state->time;
  // Add a change for all sub-elements
  if (sig.elements.size() > 0) {
    for (size_t i = 0, e = sig.elements.size(); i < e; ++i) {
      auto valueDump = sig.dump(i);
      mergedChanges[std::make_pair(sigIndex, i)] =
          std::make_pair(time, valueDump);
    }
  } else {
    // Add one change for the whole signal.
    auto valueDump = sig.dump();
    mergedChanges[std::make_pair(sigIndex, 0)] =
        std::make_pair(time, valueDump);
  }
}

void Trace::flush(bool force) {
  if (changes.size() > 0 || mergedChanges.size() > 0) {
    if (mode == TraceMode::full || mode == TraceMode::reduced)
      flushFull();
    else if (mode == TraceMode::merged || mode == TraceMode::mergedReduce)
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
    auto sig = state->signals[sigIndex];
    auto change = elem.second;
    // Add the changes for all signals.
    if (mode == merged) {
      for (auto inst : sig.triggers) {
        if (sig.elements.size() > 0) {
          std::string path;
          llvm::raw_string_ostream ss(path);
          ss << state->instances[inst].path << '/' << sig.name << '['
             << elem.first.second << ']';
          changes.push_back(std::make_pair(path, change.second));
        } else {
          auto path = state->instances[inst].path + '/' + sig.name;
          changes.push_back(std::make_pair(path, change.second));
        }
      }
    } else if (mode == mergedReduce) {
      // Add the changes for the top-level signals only.
      auto sigIndex = elem.first.first;
      auto sig = state->signals[sigIndex];
      auto root = state->root;
      if (sig.owner == root) {
        if (sig.elements.size() > 0) {
          std::string path;
          llvm::raw_string_ostream ss(path);
          ss << state->instances[root].path << '/' << sig.name << '['
             << elem.first.second << ']';
          changes.push_back(std::make_pair(path, change.second));
        } else {
          auto path = state->instances[root].path + '/' + sig.name;
          changes.push_back(std::make_pair(path, change.second));
        }
      }
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