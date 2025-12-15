//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_PENDINGCHANGES_H_
#define LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_PENDINGCHANGES_H_

#include "llvm/ADT/StringMap.h"
#include "llvm/Support/LSP/Protocol.h"
#include "llvm/Support/ThreadPool.h"

#include "circt/Tools/circt-verilog-lsp-server/CirctVerilogLspServerMain.h"

#include <chrono>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

namespace circt {
namespace lsp {

/// Build a pool strategy with a sensible minimum.
static llvm::ThreadPoolStrategy makeStrategy(unsigned maxThreads) {
  llvm::ThreadPoolStrategy s = llvm::hardware_concurrency();
  s.ThreadsRequested = (maxThreads == 0 ? 1u : maxThreads);
  return s;
}

/// Debounce tuning for document-change bursts.
struct DebounceOptions {
  /// If true, flush immediately (no sleep/check).
  bool disableDebounce = false;
  /// Minimum quiet time before we flush.
  uint64_t debounceMinMs = 0;
  /// Maximum total burst time (0 = no cap).
  uint64_t debounceMaxMs = 0;
  static DebounceOptions
  fromLSPOptions(const circt::lsp::LSPServerOptions &opts);
};

/// Accumulated edits + timing for a single document key.
struct PendingChanges {
  std::vector<llvm::lsp::TextDocumentContentChangeEvent> changes;
  int64_t version = 0;
  std::chrono::steady_clock::time_point firstChangeTime{};
  std::chrono::steady_clock::time_point lastChangeTime{};
};

/// Thread-safe accumulator + debouncer for text document changes.
class PendingChangesMap {
public:
  explicit PendingChangesMap(
      unsigned maxThreads = std::thread::hardware_concurrency())
      : pool(makeStrategy(maxThreads)), tasks(pool) {}

  /// Call during server shutdown; Erase all file changes, then clear file map.
  /// Thread-safe.
  void abort();

  /// Remove all pending edits for a document key.
  /// Safe to call from the LSP thread when a file closes.
  /// Thread-safe.
  void erase(llvm::StringRef key);

  // Convenience overload for using with an URI.
  void erase(const llvm::lsp::URIForFile &uri);

  /// Append new edits for a document key, then start a debounced update thread.
  /// Thread-safe.
  void
  debounceAndUpdate(const llvm::lsp::DidChangeTextDocumentParams &params,
                    DebounceOptions options,
                    std::function<void(std::unique_ptr<PendingChanges>)> cb);

  /// Append new edits for a document key.
  /// Thread-safe.
  void enqueueChange(const llvm::lsp::DidChangeTextDocumentParams &params);

  /// Schedule a debounce check on the internal pool and call `cb` when ready.
  /// If the task becomes obsolete (newer edits arrive before the quiet window,
  /// and max cap not reached), `cb(nullptr)` is invoked.
  /// Thread-safe.
  void debounceAndThen(const llvm::lsp::DidChangeTextDocumentParams &params,
                       DebounceOptions options,
                       std::function<void(std::unique_ptr<PendingChanges>)> cb);

private:
  /// NOT thread-safe; caller must hold mu.
  PendingChanges &getOrCreateEntry(std::string_view key);

  /// NOT thread-safe; caller must hold mu.
  std::unique_ptr<PendingChanges>
  takeAndErase(llvm::StringMap<PendingChanges>::iterator it);

  /// Per-document edit bursts, keyed by file string.
  /// Stored by value; safe to move out when flushing.
  llvm::StringMap<PendingChanges> pending;
  /// Guards `pending`.
  std::mutex mu;

  /// Internal concurrency used for sleeps + checks.
  llvm::StdThreadPool pool;
  llvm::ThreadPoolTaskGroup tasks;
};

} // namespace lsp
} // namespace circt

#endif // LIB_CIRCT_TOOLS_CIRCT_VERILOG_LSP_SERVER_PENDINGCHANGES_H_
