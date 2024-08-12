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

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <queue>
#include <string>

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
template <typename T>
class TSQueue {
  using Lock = std::lock_guard<std::mutex>;

  /// The queue and its mutex.
  mutable std::mutex qM;
  std::queue<T> q;

  /// A mutex to ensure that only one 'pop' operation is happening at a time. It
  /// is critical that locks be obtained on this and `qM` same order in both pop
  /// methods. This lock should be obtained first since one of the pop methods
  /// must unlock `qM` then relock it.
  std::mutex popM;

public:
  /// Push onto the queue.
  template <typename... E>
  void push(E... t) {
    Lock l(qM);
    q.emplace(t...);
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
};
} // namespace utils
} // namespace esi

#endif // ESI_UTILS_H
