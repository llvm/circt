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

/// Thread safe queue. Just wraps std::queue protected with a lock. Long term,
/// we need to avoid copying data. It has a lot of data copies currently.
template <typename T>
class TSQueue {
  using Lock = std::lock_guard<std::mutex>;

  mutable std::mutex m;
  std::queue<T> q;

public:
  /// Push onto the queue.
  template <typename... E>
  void push(E... t) {
    Lock l(m);
    q.emplace(t...);
  }

  /// Pop something off the queue but return nullopt if the queue is empty. Why
  /// doesn't std::queue have anything like this?
  std::optional<T> pop() {
    Lock l(m);
    if (q.size() == 0)
      return std::nullopt;
    auto t = q.front();
    q.pop();
    return t;
  }

  /// Call the callback for the front of the queue (if anything is there). Only
  /// pop it off the queue if the callback returns true.
  void pop(std::function<bool(const T &)> callback) {
    // TODO: since we need to unlock the mutex to call the callback, the queue
    // could be pushed on to and its memory layout could thusly change,
    // invalidating the reference returned by `.front()`. The easy solution here
    // is to copy the data. Avoid copying the data.
    T t;
    {
      Lock l(m);
      if (q.size() == 0)
        return;
      t = q.front();
    }
    if (callback(t)) {
      Lock l(m);
      q.pop();
    }
  }

  bool empty() const {
    Lock l(m);
    return q.empty();
  }
};
} // namespace utils
} // namespace esi

#endif // ESI_UTILS_H
