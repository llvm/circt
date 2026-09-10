//===- Utils.h - ESI runtime utility code -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// DO NOT EDIT!
// This file is distributed as part of an ESI package. The source for this file
// should always be modified within CIRCT.
//
//===----------------------------------------------------------------------===//

// NOLINTNEXTLINE(llvm-header-guard)
#ifndef ESI_UTILS_H
#define ESI_UTILS_H

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <unordered_set>

namespace esi {
namespace utils {
// Very basic base64 encoding.
void encodeBase64(const void *data, size_t size, std::string &out);

/// C++'s stdlib doesn't have a hash_combine function. This is a simple one.
inline size_t hash_combine(size_t h1, size_t h2) {
  return h1 + 0x9e3779b9 + (h2 << 6) + (h2 >> 2);
}

/// Thread safe queue. Just wraps std::queue protected with a lock. Long term,
/// we need to avoid copying data. It has a lot of data copies currently.
///
/// Push operations notify a built-in condition variable so consumers can
/// block on `waitPop()` rather than poll. `requestShutdown()` permanently
/// wakes any waiters with `nullopt`, which is how owners cleanly retire a
/// drainer thread.
template <typename T>
class TSQueue {
  using Lock = std::lock_guard<std::mutex>;

  /// The queue and its mutex.
  mutable std::mutex qM;
  std::queue<T> q;

  /// CV signalled by `push` and `requestShutdown`. Waiters re-check the
  /// predicate under `qM` so missed notifications are recovered.
  std::condition_variable cv;
  bool shutdown = false;

  /// Optional external notifier invoked after `push` releases the queue lock.
  /// Set once at construction; never mutated afterwards.
  const std::function<void()> notifier;

  /// A mutex to ensure that only one 'pop' operation is happening at a time. It
  /// is critical that locks be obtained on this and `qM` same order in both pop
  /// methods. This lock should be obtained first since one of the pop methods
  /// must unlock `qM` then relock it.
  std::mutex popM;

public:
  /// Default constructor: no external notifier.
  TSQueue() = default;

  /// Construct a queue that calls `notifier` after every successful `push`,
  /// once the queue mutex has been released. Use this to ring an external
  /// doorbell (e.g. a server-wide "ports with pending data" set) without
  /// making every push site know about it. The notifier is fixed for the
  /// lifetime of the queue — there is no setter, because changing it while
  /// other threads are pushing would be a data race.
  explicit TSQueue(std::function<void()> notifier)
      : notifier(std::move(notifier)) {}

  /// Push onto the queue. Wakes one `waitPop()` waiter (if any) and rings the
  /// external notifier, if one was supplied to the constructor.
  template <typename... E>
  void push(E... t) {
    {
      Lock l(qM);
      q.emplace(t...);
    }
    if (notifier)
      notifier();
    cv.notify_one();
  }

  /// Pop something off the queue but return nullopt if the queue is empty. Why
  /// doesn't std::queue have anything like this?
  std::optional<T> pop() {
    Lock pl(popM);
    Lock ql(qM);
    if (q.size() == 0)
      return std::nullopt;
    auto t = q.front();
    q.pop();
    return t;
  }

  /// Block until an item is available or `requestShutdown()` is called.
  /// Returns `nullopt` only in the shutdown case. Intended for single-consumer
  /// drainer threads; do not interleave with `pop()` on the same queue.
  std::optional<T> waitPop() {
    std::unique_lock<std::mutex> l(qM);
    cv.wait(l, [this] { return shutdown || !q.empty(); });
    if (q.empty())
      return std::nullopt;
    auto t = q.front();
    q.pop();
    return t;
  }

  /// Call the callback for the front of the queue (if anything is there). Only
  /// pop it off the queue if the callback returns true.
  void pop(std::function<bool(const T &)> callback) {
    // Since we need to unlock the mutex to call the callback, the queue
    // could be pushed on to and its memory layout could thusly change,
    // invalidating the reference returned by `.front()`. The easy solution here
    // is to copy the data. TODO: Avoid copying the data.
    Lock pl(popM);
    T t;
    {
      Lock l(qM);
      if (q.size() == 0)
        return;
      t = q.front();
    }
    if (callback(t)) {
      Lock l(qM);
      q.pop();
    }
  }

  /// Is the queue empty?
  bool empty() const {
    Lock l(qM);
    return q.empty();
  }

  /// Permanently retire the queue: wake every current and future `waitPop()`
  /// with `nullopt` so consumer threads can exit cleanly. Pushes after this
  /// still enqueue, but no one is expected to be waiting.
  void requestShutdown() {
    {
      Lock l(qM);
      shutdown = true;
    }
    cv.notify_all();
  }

  /// True once `requestShutdown()` has been called.
  bool isShutdown() const {
    Lock l(qM);
    return shutdown;
  }
};

/// Multi-producer / single-consumer dirty-set of channel ids, with CV-style
/// blocking drain semantics. Producers call `markReady(id)`; the consumer
/// thread calls `waitDrain()` to atomically swap out the current set.
///
/// Use this when many independent per-channel producers feed a single
/// transport thread that needs to know which channels have work to do
/// without polling each in turn. The CV pattern (lock-around-shutdown-set,
/// notify-after-release, predicate captures both shutdown and !empty) is
/// fiddly enough that centralising it here avoids subtle copy/paste bugs
/// in each backend.
template <typename ID = uint16_t>
class ReadyIdSet {
public:
  /// Add `id` to the dirty set and wake the consumer (if any). Idempotent
  /// w.r.t. an id that is already in the set.
  void markReady(ID id) {
    {
      std::lock_guard<std::mutex> lock(mutex);
      ready.insert(id);
    }
    cv.notify_one();
  }

  /// Block until either `requestShutdown()` is called or the set is
  /// non-empty, then atomically swap the current set into `out`. With
  /// `backoff` non-nullopt, returns after the timeout even if neither
  /// condition holds (useful when the caller maintains its own retry set
  /// it wants to re-process periodically). Returns `false` once shutdown
  /// has been signalled, so consumer loops can write
  /// `while (set.waitDrain(ids, backoff)) { ... }`.
  bool waitDrain(std::unordered_set<ID> &out,
                 std::optional<std::chrono::milliseconds> backoff = {}) {
    std::unique_lock<std::mutex> lock(mutex);
    auto pred = [&] { return shutdown || !ready.empty(); };
    if (backoff)
      cv.wait_for(lock, *backoff, pred);
    else
      cv.wait(lock, pred);
    out.swap(ready);
    return !shutdown;
  }

  /// Signal a clean shutdown: wakes every current and future `waitDrain`
  /// caller, which will then observe `false`. Idempotent.
  void requestShutdown() {
    {
      std::lock_guard<std::mutex> lock(mutex);
      shutdown = true;
    }
    cv.notify_all();
  }

  /// True once `requestShutdown()` has been called.
  bool isShutdown() const {
    std::lock_guard<std::mutex> lock(mutex);
    return shutdown;
  }

private:
  mutable std::mutex mutex;
  std::condition_variable cv;
  std::unordered_set<ID> ready;
  bool shutdown = false;
};

/// Compute ceil(bits/8).
inline uint64_t bitsToBytes(uint64_t bits) { return (bits + 7) / 8; }

} // namespace utils
} // namespace esi

#endif // ESI_UTILS_H
