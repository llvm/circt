//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PendingChanges.h"

namespace circt {
namespace lsp {

/// Factory: build from server options. Keep mapping 1:1 for clarity.
DebounceOptions
DebounceOptions::fromLSPOptions(const circt::lsp::LSPServerOptions &opts) {
  DebounceOptions d;
  d.disableDebounce = opts.disableDebounce;
  d.debounceMinMs = opts.debounceMinMs;
  d.debounceMaxMs = opts.debounceMaxMs;
  return d;
}

void PendingChangesMap::abort() {
  std::scoped_lock lock(mu);
  pending.clear();
  pool.wait();
}

void PendingChangesMap::erase(llvm::StringRef key) {
  std::scoped_lock lock(mu);
  pending.erase(key);
}

void PendingChangesMap::erase(const llvm::lsp::URIForFile &uri) {
  auto file = uri.file();
  if (!file.empty())
    erase(file);
}

void PendingChangesMap::debounceAndUpdate(
    const llvm::lsp::DidChangeTextDocumentParams &params,
    DebounceOptions options,
    std::function<void(std::unique_ptr<PendingChanges>)> cb) {
  enqueueChange(params);
  debounceAndThen(params, options, std::move(cb));
}

void PendingChangesMap::enqueueChange(
    const llvm::lsp::DidChangeTextDocumentParams &params) {
  const auto now = nowFn();
  const std::string key = params.textDocument.uri.file().str();

  std::scoped_lock lock(mu);
  PendingChanges &pending = getOrCreateEntry(key);

  pending.changes.insert(pending.changes.end(), params.contentChanges.begin(),
                         params.contentChanges.end());
  pending.version = params.textDocument.version;
  pending.lastChangeTime = now;

  // If this was the first insert after a flush, record start of burst.
  if (pending.changes.size() == params.contentChanges.size())
    pending.firstChangeTime = now;
}

void PendingChangesMap::debounceAndThen(
    const llvm::lsp::DidChangeTextDocumentParams &params,
    DebounceOptions options,
    std::function<void(std::unique_ptr<PendingChanges>)> cb) {
  const std::string key = params.textDocument.uri.file().str();
  const auto scheduleTime = nowFn();

  // If debounce is disabled, run on main thread
  if (options.disableDebounce) {
    std::scoped_lock lock(mu);
    auto it = pending.find(key);
    if (it == pending.end())
      return cb(nullptr);
    return cb(takeAndErase(it));
  }

  // If debounced, run entirely on the pool; do not block the LSP thread.
  tasks.async([this, key, scheduleTime, options, cb = std::move(cb)]() {
    // Simple timer: sleep min-quiet before checking. We rely on the fact
    // that newer edits can arrive while we sleep, updating lastChangeTime.
    if (options.debounceMinMs > 0)
      waitForMinMs(options.debounceMinMs, scheduleTime);

    std::unique_ptr<PendingChanges>
        result; // decided under lock, callback after

    {
      std::scoped_lock lock(mu);
      auto it = pending.find(key);
      if (it != pending.end()) {
        PendingChanges &pc = it->second;
        const auto now = nowFn();

        // quietSinceSchedule: if no newer edits arrived after we scheduled
        // this task, then we consider the burst "quiet" and flush now.
        const bool quietSinceSchedule = (pc.lastChangeTime <= scheduleTime);

        // Apply max-burst cap if configured: force a flush once the total
        // time since first change exceeds the cap.
        bool maxWaitExpired = false;
        if (options.debounceMaxMs > 0) {
          const auto elapsedMs =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  now - pc.firstChangeTime)
                  .count();
          maxWaitExpired =
              static_cast<uint64_t>(elapsedMs) >= options.debounceMaxMs;
        }

        if (quietSinceSchedule || maxWaitExpired)
          result = takeAndErase(it); // flush now
        // else: newer edits arrived; obsolete -> result stays null
      }
    }

    // Invoke outside the lock to avoid deadlocks and allow heavy work.
    cb(std::move(result)); // nullptr => obsolete (no flush)
  });
}

PendingChanges &PendingChangesMap::getOrCreateEntry(std::string_view key) {
  auto it = pending.find(key);
  if (it != pending.end())
    return it->second;
  auto inserted = pending.try_emplace(key);
  return inserted.first->second;
}

std::unique_ptr<PendingChanges>
PendingChangesMap::takeAndErase(llvm::StringMap<PendingChanges>::iterator it) {
  auto out = std::make_unique<PendingChanges>(std::move(it->second));
  pending.erase(it);
  return out;
}

void PendingChangesMap::waitForMinMs(uint64_t ms,
                                     SteadyClock::time_point start) {
  if (!ms)
    return;
  if (!useManualClock) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
    return;
  }
  // Manual clock: busy-wait with yields until now() reaches start + ms.
  const auto target = start + std::chrono::milliseconds(ms);
  while (nowFn() < target) {
    std::this_thread::yield();
  }
}

} // namespace lsp
} // namespace circt
